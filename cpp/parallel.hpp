// GuoFish C9 — the primitives the parallel search is built out of.
//
// Three things live here, and none of them knows anything about chess:
//
//   * the MPSC leaf queue (Vyukov's intrusive algorithm),
//   * CPU topology discovery and thread pinning, which on this machine is a
//     correctness-adjacent concern rather than a tuning one — see below,
//   * a latency histogram, used for the GIL-acquisition prediction the C9 brief
//     asks to be tested rather than merely recorded.
//
// They are separated from cpp/search.hpp because they are testable on their own
// and because a reader auditing the search should not have to read a lock-free
// queue to do it.
//
//
// WHY PINNING IS NOT A MICRO-OPTIMISATION HERE
// --------------------------------------------
// The target machine is an i5-12600K: 6 performance cores with SMT (12 logical)
// plus 4 efficiency cores (4 logical), 16 logical processors in total. Windows'
// Thread Director moves threads between the two classes on its own reading of
// what they are doing, and a search thread parked on an E-core is not merely
// slower at its own work — it holds the ROOT's contended atomics for longer,
// and the root is touched by every descent and every backup in the engine. A
// single slow thread in that path costs more than its own throughput.
//
// So the policy is explicit and measured (DECISIONS.md, C9) rather than left to
// the scheduler. `AffinityPolicy::PCorePhysical` puts one thread on each P-core
// and leaves the SMT siblings idle; `PCoreSmt` uses both siblings of each
// P-core. Which is better is genuinely open — descent is pointer-chasing and
// memory-latency-bound, which favours SMT, but the SoA arena makes sibling
// scans prefetch-friendly, which removes the stalls SMT would fill — so both
// are implemented and both are in the sweep.
//
// Everything degrades rather than fails: if the platform will not report its
// topology (WSL2 does not expose `cpu_capacity`, for one), `Topology::source`
// says so, the hybrid split is reported as unknown, and the policy falls back
// to `None` with the reason attached rather than pinning threads to a layout
// that was guessed.

#ifndef GUOFISH_PARALLEL_HPP
#define GUOFISH_PARALLEL_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#endif

namespace guofish {

// ---------------------------------------------------------------------------
// The MPSC leaf queue
//
// Dmitry Vyukov's intrusive multi-producer / single-consumer queue. Chosen over
// a mutex-protected deque for one reason that matters and one that does not:
//
//   matters      a producer's push is a single `exchange` plus a single store,
//                so a search thread that is preempted between them cannot block
//                any other producer. Under a mutex, a thread descheduled while
//                holding it stops every other worker — and with W threads pinned
//                across P-cores that a hybrid scheduler is free to park, "a
//                thread was descheduled at a bad moment" is a normal event, not
//                a rare one.
//   does not     raw throughput. There is one push per simulation, i.e. a few
//                thousand per search, against ~5.6 us of descent each. A mutex
//                would be fast enough. Blocking behaviour is the argument.
//
// FIFO is a requirement, not a nicety. Acceptance layer 2 (W=1, K=8, two runs,
// bit-identical trees) holds only if the dispatcher expands leaves in the order
// the worker submitted them, because expansion order determines the tree the
// next descent sees. A LIFO stack — which is the cheaper lock-free structure —
// would make that test fail for a reason that is not a bug.
//
// The stub node is Vyukov's: it keeps `pop` from having to special-case the
// empty queue, at the cost of one dummy element that is never handed out.
// ---------------------------------------------------------------------------

// A node in the queue. Any payload type embeds one of these; the queue only
// ever touches `next`.
struct MpscNode {
    std::atomic<MpscNode *> next{nullptr};
};

class MpscQueue {
public:
    MpscQueue() : head_(&stub_), tail_(&stub_) { stub_.next.store(nullptr, std::memory_order_relaxed); }

    MpscQueue(const MpscQueue &) = delete;
    MpscQueue &operator=(const MpscQueue &) = delete;

    // Any thread. Wait-free: one exchange, one store.
    void push(MpscNode *node) {
        node->next.store(nullptr, std::memory_order_relaxed);
        // Release: everything the producer wrote into the payload before this
        // call is visible to the consumer that acquires the same pointer.
        MpscNode *prev = head_.exchange(node, std::memory_order_acq_rel);
        prev->next.store(node, std::memory_order_release);
    }

    // The consumer thread only.
    //
    // Returns nullptr in two distinguishable-in-principle cases that the caller
    // does not need to distinguish: the queue is empty, or a producer is in the
    // window between its exchange and its store and the queue is momentarily
    // inconsistent. The dispatcher knows how many items it is owed (it counts
    // submissions), so it spins rather than concluding "empty" — see
    // `drain_exactly` in cpp/search.hpp.
    MpscNode *try_pop() {
        MpscNode *tail = tail_;
        MpscNode *next = tail->next.load(std::memory_order_acquire);

        if (tail == &stub_) {
            if (next == nullptr) {
                return nullptr;
            }
            tail_ = next;
            tail = next;
            next = tail->next.load(std::memory_order_acquire);
        }

        if (next != nullptr) {
            tail_ = next;
            return tail;
        }

        if (tail != head_.load(std::memory_order_acquire)) {
            // A producer has exchanged but not yet linked. Not empty.
            return nullptr;
        }

        // Re-arm the stub so the queue is never left with tail == head and no
        // way back to the empty state.
        push(&stub_);
        next = tail->next.load(std::memory_order_acquire);
        if (next != nullptr) {
            tail_ = next;
            return tail;
        }
        return nullptr;
    }

    // True only when the consumer has drained everything AND no producer is
    // mid-push. Used by asserts, not by the drain loop.
    bool empty_from_consumer() const {
        MpscNode *tail = tail_;
        return tail == &stub_ && tail->next.load(std::memory_order_acquire) == nullptr;
    }

private:
    std::atomic<MpscNode *> head_;
    // Consumer-private. Deliberately not atomic: nothing else reads it.
    MpscNode *tail_;
    MpscNode stub_;
};

// ---------------------------------------------------------------------------
// CPU topology
// ---------------------------------------------------------------------------

enum class AffinityPolicy : std::uint8_t {
    None = 0,          // let the OS place threads
    PCorePhysical,     // one thread per performance core, SMT siblings unused
    PCoreSmt,          // every logical processor of every performance core
    AllLogical,        // every logical processor, E-cores included
};

inline const char *affinity_policy_name(AffinityPolicy policy) noexcept {
    switch (policy) {
        case AffinityPolicy::None: return "none";
        case AffinityPolicy::PCorePhysical: return "pcore_physical";
        case AffinityPolicy::PCoreSmt: return "pcore_smt";
        case AffinityPolicy::AllLogical: return "all_logical";
    }
    return "none";
}

inline constexpr std::array<const char *, 4> kAffinityPolicyNames = {
    "none", "pcore_physical", "pcore_smt", "all_logical"};

// The inverse, for the Python surface.
//
// THE POLICY CROSSES THE BOUNDARY AS A STRING, NOT AS A BOUND ENUM, and the
// reason is a measured leak rather than taste. `py::enum_` was the obvious
// binding and it is the only one in this module; on pybind11 2.12 it leaks its
// registration — 4,402 bytes in 78 allocations at import, every one of them
// through `pybind11::enum_<AffinityPolicy>`, against zero for the same module
// without it (measured both ways, ASan on Linux/Clang).
//
// The number is trivial and one-time. What is not trivial is that
// README_BUILD.md's leak discriminator is "no leaked allocation's stack should
// mention guofish_core", and that is the only tool a non-C++ reader has for
// telling our leaks from CPython's. Spending it on an enum's ergonomics would
// blunt it for every later chunk, and `ParallelStats::affinity` already reports
// a string, so a string-valued config is the symmetric choice anyway.
inline AffinityPolicy affinity_policy_from_name(std::string_view name) {
    for (std::size_t i = 0; i < kAffinityPolicyNames.size(); ++i) {
        if (name == kAffinityPolicyNames[i]) {
            return static_cast<AffinityPolicy>(i);
        }
    }
    std::string valid;
    for (std::size_t i = 0; i < kAffinityPolicyNames.size(); ++i) {
        if (i != 0) {
            valid += ", ";
        }
        valid += kAffinityPolicyNames[i];
    }
    throw std::invalid_argument("guofish: unknown affinity policy '" + std::string(name) +
                                "'; expected one of: " + valid);
}

struct Topology {
    // Logical processor ids, in the order threads should be assigned to them.
    std::vector<int> pcore_physical;   // one per performance core
    std::vector<int> pcore_all;        // every logical processor on a P-core
    std::vector<int> ecore_all;        // every logical processor on an E-core
    std::vector<int> all_logical;
    // True when the platform reported two efficiency classes. On a homogeneous
    // machine every core is a "P-core" by construction and the two P lists are
    // just the physical and logical enumerations.
    bool hybrid = false;
    // Where the numbers came from, or why there are none. Reported to Python so
    // a benchmark table can say what it was actually running on.
    std::string source = "unavailable";
};

#if defined(_WIN32)

// GetLogicalProcessorInformationEx(RelationProcessorCore) reports one record per
// PHYSICAL core, carrying the mask of its logical processors and an
// EfficiencyClass. Microsoft documents higher EfficiencyClass as higher
// performance, so the P-cores are the records at the maximum class. On a
// non-hybrid part every record reports 0 and the maximum is 0, which makes the
// homogeneous case fall out of the same code with no branch.
inline Topology detect_topology() {
    Topology topo;
    DWORD length = 0;
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        topo.source = "GetLogicalProcessorInformationEx: size query failed";
        return topo;
    }

    std::vector<std::uint8_t> buffer(length);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            // The API writes a packed sequence of variable-length records into a
            // caller-supplied byte buffer and has no other calling convention;
            // the buffer is over-aligned for the struct because it came from
            // std::vector's allocator, and it is walked by the documented
            // `Size` field below rather than by pointer arithmetic on the type.
            reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buffer.data()),
            &length)) {
        topo.source = "GetLogicalProcessorInformationEx: query failed";
        return topo;
    }

    struct CoreRecord {
        std::vector<int> logical;
        int efficiency_class;
    };
    std::vector<CoreRecord> cores;
    int max_class = 0;

    DWORD offset = 0;
    while (offset < length) {
        // Same buffer, same reason: the sequence is heterogeneous and the only
        // way to read the next record's Size is through the common header.
        const auto *record = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(
            buffer.data() + offset);
        if (record->Size == 0) {
            break;
        }
        if (record->Relationship == RelationProcessorCore) {
            CoreRecord core;
            core.efficiency_class = static_cast<int>(record->Processor.EfficiencyClass);
            max_class = (core.efficiency_class > max_class) ? core.efficiency_class : max_class;
            for (WORD g = 0; g < record->Processor.GroupCount; ++g) {
                const GROUP_AFFINITY &group = record->Processor.GroupMask[g];
                // One processor group only. A 16-thread desktop part is always
                // in group 0; a machine with more than 64 logical processors
                // would need group-aware pinning, and pretending otherwise
                // would silently pin every thread to the wrong core, so the
                // extra groups are reported as unusable rather than guessed at.
                if (group.Group != 0) {
                    continue;
                }
                for (int bit = 0; bit < 64; ++bit) {
                    if ((group.Mask >> bit) & 1u) {
                        core.logical.push_back(bit);
                    }
                }
            }
            if (!core.logical.empty()) {
                cores.push_back(std::move(core));
            }
        }
        offset += record->Size;
    }

    if (cores.empty()) {
        topo.source = "GetLogicalProcessorInformationEx: no processor-core records in group 0";
        return topo;
    }

    for (const CoreRecord &core : cores) {
        const bool performance = core.efficiency_class == max_class;
        for (std::size_t i = 0; i < core.logical.size(); ++i) {
            topo.all_logical.push_back(core.logical[i]);
            if (performance) {
                topo.pcore_all.push_back(core.logical[i]);
                if (i == 0) {
                    topo.pcore_physical.push_back(core.logical[i]);
                }
            } else {
                topo.ecore_all.push_back(core.logical[i]);
            }
        }
    }
    topo.hybrid = !topo.ecore_all.empty();
    topo.source = "GetLogicalProcessorInformationEx";
    return topo;
}

inline bool pin_current_thread(int logical_cpu) {
    if (logical_cpu < 0 || logical_cpu >= 64) {
        return false;
    }
    const DWORD_PTR mask = static_cast<DWORD_PTR>(1) << logical_cpu;
    return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
}

#else  // POSIX

namespace detail {

inline bool read_int_file(const std::string &path, long &out) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    long value = 0;
    in >> value;
    if (!in) {
        return false;
    }
    out = value;
    return true;
}

// "0-1", "0,4", "0-1,8-9" -> the list of ids.
inline std::vector<int> parse_cpu_list(const std::string &text) {
    std::vector<int> out;
    std::stringstream stream(text);
    std::string chunk;
    while (std::getline(stream, chunk, ',')) {
        const std::size_t dash = chunk.find('-');
        if (dash == std::string::npos) {
            try {
                out.push_back(std::stoi(chunk));
            } catch (const std::exception &) {
                // A malformed sysfs field is not worth aborting over; the
                // caller falls back to no pinning and says why.
            }
        } else {
            try {
                const int lo = std::stoi(chunk.substr(0, dash));
                const int hi = std::stoi(chunk.substr(dash + 1));
                for (int i = lo; i <= hi; ++i) {
                    out.push_back(i);
                }
            } catch (const std::exception &) {
            }
        }
    }
    return out;
}

}  // namespace detail

// sysfs. `cpu_capacity` is how the kernel reports the hybrid split on Alder
// Lake and later; it is frequently ABSENT (notably under WSL2, which is where
// this project's ThreadSanitizer runs live), and when it is, this reports a
// homogeneous machine rather than inventing a split. `thread_siblings_list`
// gives the SMT pairing, which is available far more widely.
inline Topology detect_topology() {
    Topology topo;
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        topo.source = "std::thread::hardware_concurrency() == 0";
        return topo;
    }

    std::vector<long> capacity(hw, 0);
    bool have_capacity = true;
    for (unsigned cpu = 0; cpu < hw; ++cpu) {
        const std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                                 "/cpu_capacity";
        if (!detail::read_int_file(path, capacity[cpu])) {
            have_capacity = false;
            break;
        }
    }

    long max_capacity = 0;
    if (have_capacity) {
        for (unsigned cpu = 0; cpu < hw; ++cpu) {
            max_capacity = (capacity[cpu] > max_capacity) ? capacity[cpu] : max_capacity;
        }
    }

    // First logical processor of each SMT sibling group, in ascending order.
    std::vector<bool> claimed(hw, false);
    for (unsigned cpu = 0; cpu < hw; ++cpu) {
        topo.all_logical.push_back(static_cast<int>(cpu));
    }
    for (unsigned cpu = 0; cpu < hw; ++cpu) {
        if (claimed[cpu]) {
            continue;
        }
        std::vector<int> siblings;
        std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                         "/topology/thread_siblings_list");
        if (in) {
            std::string text;
            std::getline(in, text);
            siblings = detail::parse_cpu_list(text);
        }
        if (siblings.empty()) {
            siblings.push_back(static_cast<int>(cpu));
        }
        std::sort(siblings.begin(), siblings.end());

        const bool performance = !have_capacity || capacity[cpu] == max_capacity;
        for (std::size_t i = 0; i < siblings.size(); ++i) {
            const int id = siblings[i];
            if (id < 0 || static_cast<unsigned>(id) >= hw) {
                continue;
            }
            claimed[static_cast<unsigned>(id)] = true;
            if (performance) {
                topo.pcore_all.push_back(id);
                if (i == 0) {
                    topo.pcore_physical.push_back(id);
                }
            } else {
                topo.ecore_all.push_back(id);
            }
        }
    }

    topo.hybrid = have_capacity && !topo.ecore_all.empty();
    topo.source = have_capacity ? "sysfs (cpu_capacity + thread_siblings_list)"
                                : "sysfs (thread_siblings_list only; no cpu_capacity, "
                                  "hybrid split not reported by this kernel)";
    return topo;
}

inline bool pin_current_thread(int logical_cpu) {
    if (logical_cpu < 0) {
        return false;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<unsigned>(logical_cpu), &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

#endif

// The logical processors a policy wants, in assignment order. Empty means "do
// not pin" — either because the policy says so or because the platform did not
// report enough to pin honestly.
inline std::vector<int> affinity_slots(const Topology &topo, AffinityPolicy policy) {
    switch (policy) {
        case AffinityPolicy::None: return {};
        case AffinityPolicy::PCorePhysical: return topo.pcore_physical;
        case AffinityPolicy::PCoreSmt: return topo.pcore_all;
        case AffinityPolicy::AllLogical: return topo.all_logical;
    }
    return {};
}

// ---------------------------------------------------------------------------
// Latency histogram
//
// Raw samples in nanoseconds, plus the running extremes. Percentiles are
// computed in Python from the samples rather than here, because the C9 brief
// asks for a HISTOGRAM and not a mean: the tail is the whole question, and a
// mean hides it behind the many fast acquisitions.
//
// Single-threaded by contract — only the dispatcher writes one of these.
// ---------------------------------------------------------------------------

class LatencyHistogram {
public:
    void reserve(std::size_t n) { samples_.reserve(n); }

    void add(std::int64_t nanoseconds) {
        samples_.push_back(nanoseconds);
        total_ += nanoseconds;
        if (nanoseconds > max_ || count_ == 0) {
            max_ = nanoseconds;
        }
        if (nanoseconds < min_ || count_ == 0) {
            min_ = nanoseconds;
        }
        ++count_;
    }

    void clear() {
        samples_.clear();
        count_ = 0;
        total_ = 0;
        min_ = 0;
        max_ = 0;
    }

    const std::vector<std::int64_t> &samples() const noexcept { return samples_; }
    std::int64_t count() const noexcept { return count_; }
    std::int64_t total_ns() const noexcept { return total_; }
    std::int64_t min_ns() const noexcept { return min_; }
    std::int64_t max_ns() const noexcept { return max_; }

private:
    std::vector<std::int64_t> samples_;
    std::int64_t count_ = 0;
    std::int64_t total_ = 0;
    std::int64_t min_ = 0;
    std::int64_t max_ = 0;
};

// The dispatcher's hook into whatever the evaluator's host language needs doing
// once per batch. In C9 the evaluator is the replay dump and there is nothing to
// do, so the only implementation is the GIL probe in cpp/bindings.cpp — which
// exists to test scope 2.1's PREDICTION (acquisition stays near the uncontended
// floor because no Python bytecode runs during a search) on the real C9 thread
// topology, one chunk before C10 depends on it being true.
class BatchHook {
public:
    virtual ~BatchHook() = default;
    // Called by the dispatcher immediately before a batch is expanded, with the
    // batch size. Returns the time it spent waiting, in nanoseconds.
    virtual std::int64_t before_batch(std::size_t batch_size) = 0;
};

}  // namespace guofish

#endif  // GUOFISH_PARALLEL_HPP
