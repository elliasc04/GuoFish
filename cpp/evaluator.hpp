// GuoFish C10 — the live evaluation boundary, and the gather/softmax that sits
// on the C++ side of it.
//
// WHAT CROSSES THE BOUNDARY, AND WHAT DELIBERATELY DOES NOT
// ---------------------------------------------------------
// Three buffers, allocated once (scope §2.1). C++ writes 68 int32 tokens per
// leaf into `input_buffer`, calls one Python callable with a row count, and
// reads a bf16 `[max_batch, 4096]` policy block and an fp32 `[max_batch]` value
// block back. Nothing is copied at the boundary.
//
// What does NOT cross it is a 4096-wide row per node. A leaf has 26-38 legal
// moves; materialising the other ~4060 logits per expansion is the single
// largest avoidable cost in the Python evaluator, so the gather below indexes
// the buffer directly and touches only the entries it needs. At batch 256 that
// is the difference between reading 2 MiB and reading ~16 KiB per batch.
//
//
// WHY THE POLICY BUFFER IS bf16 AND NOT float
// -------------------------------------------
// Because Python's is. `policy_logits.cpu()` in the reference is a bf16 tensor
// and `MCTSNode.expand` gathers from it before `.float()`. Handing C++ an fp32
// buffer would round the logits through a wider type on a different schedule
// than the reference does, and Gate 2's whole claim is that the two gathers see
// the SAME BITS. So `policy_buffer` is bf16 and the widening happens here, in
// `bf16_to_float`, exactly where `.float()` happens over there.
//
// bf16 is the top 16 bits of an IEEE-754 binary32 with the same exponent field,
// so the widening is a shift and nothing else: no rounding, no lookup, exactly
// representable. That is why `.float()` on the Python side is also exact, and
// why the two sides can be compared at all.
//
//
// THE ORDER THE SOFTMAX REDUCES IN IS PART OF THE ANSWER
// ------------------------------------------------------
// `torch.softmax` is not permutation-invariant — scope §2.6 measured 109 of 200
// random permutations of one node's legal logits reproducing bit-identical
// priors, max Δ 3e-7 — so "gather the legal logits" and "order the children"
// are two operations and doing them in the wrong order changes the priors on a
// large fraction of nodes (26,569 of 51,927, measured end to end).
//
// The discipline, verified 300/300 bit-identical and implemented in
// `gather_softmax_canonical` below:
//
//     softmax over GENERATION-order logits, then permute the resulting
//     probabilities into canonical (from, to, promotion) order.
//
// Never gather in sorted order. The reduction order stays a property of
// whichever movegen produced the logits — python-chess's over there,
// chess-library's here — and only the STORAGE order is canonical. That the two
// generation orders differ is precisely why Gate 2's bound is 1e-6 and not a
// few ulps: permutation alone accounts for up to 3e-7 on a correct
// implementation.
//
// ---------------------------------------------------------------------------

#ifndef GUOFISH_EVALUATOR_HPP
#define GUOFISH_EVALUATOR_HPP

#include "arena.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace guofish {

// The flat policy head's width and its index. Both are the trained contract:
// `data/multiPV/labels.py:move_index` is this same `from_square * 64 +
// to_square`, and `core.mctsv4.require_v5_config` refuses to run a checkpoint
// whose `policy_size` disagrees. A 64x64 head cannot represent a promotion
// piece, so the four promotions from one square collide on one index and
// receive identical priors — a preserved defect (scope §1), not a bug to route
// around here.
inline constexpr std::size_t kPolicySize = 4096;

// "This leaf was not sent to the network." Used by the dispatcher to mark the
// leaves the transposition cache answered, which consume no row of the batch.
inline constexpr std::size_t kNoEvalRow = static_cast<std::size_t>(-1);

inline constexpr std::size_t policy_index(std::uint16_t packed) noexcept {
    return static_cast<std::size_t>(move_from(packed)) * 64u +
           static_cast<std::size_t>(move_to(packed));
}

// bf16 bits -> float. Exact: bf16 is binary32 truncated to its top 16 bits.
//
// memcpy rather than a union or a cast through `float*`: it is the only
// type-punning spelling that is defined behaviour in C++, and every compiler in
// the allowed set folds it to a single move. Rule 6 also bans an unexplained
// reinterpret_cast, and this way there is not one to explain.
inline float bf16_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
    float out = 0.0f;
    std::memcpy(&out, &widened, sizeof(out));
    return out;
}

// float -> bf16 bits, round-to-nearest-even. Only the test surface needs this
// (it lets a Python-side reference row be written through the same conversion
// the model's `.to(torch.bfloat16)` performs); the search never converts in
// this direction.
inline std::uint16_t float_to_bf16(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    // NaN: keep it a NaN rather than letting the rounding carry turn it into an
    // infinity. Payload is not preserved, which no caller depends on.
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) {
        return static_cast<std::uint16_t>((bits >> 16) | 0x0040u);
    }
    const std::uint32_t lsb = (bits >> 16) & 1u;
    const std::uint32_t rounded = bits + 0x7FFFu + lsb;
    return static_cast<std::uint16_t>(rounded >> 16);
}

// The softmax, in float, over `count` logits in place.
//
// max-shift then exp then divide, which is what ATen's `_softmax` does on both
// its CPU and its CUDA path. It is not bit-identical to either — ATen's CPU
// reduction goes through a vectorised exp (SLEEF/AVX) rather than libm's, and
// its CUDA path reduces in warp order — and it is not trying to be: Gate 2's
// bound is 1e-6 absolute with zero prior-ORDERING inversions, because the
// ordering is what PUCT reads and the last ulp is not.
//
// Accumulating the sum in double would be closer to nothing in particular: the
// reference sums in float (the tensor is float32 after `.float()`), so float is
// the choice that tracks it. Measured against ATen over the Gate 2 corpus in
// tests/test_c10_gate2.py.
inline void softmax_in_place(float *values, std::size_t count) noexcept {
    assert(values != nullptr);
    if (count == 0) {
        return;
    }
    float max_logit = values[0];
    for (std::size_t i = 1; i < count; ++i) {
        if (values[i] > max_logit) {
            max_logit = values[i];
        }
    }
    float sum = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = std::exp(values[i] - max_logit);
        sum += values[i];
    }
    // `sum >= 1` always: the max entry contributes exactly exp(0) = 1. So there
    // is no divide-by-zero to guard and no underflow branch to take, which is
    // the reason for the max-shift beyond the overflow one.
    const float inv = 1.0f / sum;
    for (std::size_t i = 0; i < count; ++i) {
        values[i] *= inv;
    }
}

// Gather this node's legal logits out of one 4096-wide bf16 policy row,
// softmax them in the order the moves were GENERATED, and write the resulting
// probabilities out in CANONICAL order. See the file header for why those are
// three separate sentences.
//
//   `packed`      the node's legal moves, canonical order (movegen.hpp).
//   `generation`  parallel to `packed`: `generation[k]` is the index move k
//                 held in chess-library's generation order.
//   `scratch`     caller-owned, resized here; lives across leaves so a steady
//                 state performs no allocation.
//   `priors`      out, canonical order, `count` floats.
inline void gather_softmax_canonical(const std::uint16_t *policy_row, const std::uint16_t *packed,
                                     const std::uint16_t *generation, std::size_t count,
                                     std::vector<float> &scratch, float *priors) {
    assert(policy_row != nullptr && packed != nullptr && generation != nullptr);
    assert(priors != nullptr);
    scratch.resize(count);

    // Scatter into generation order. `generation` is a permutation of
    // [0, count), so every slot is written exactly once and none is read
    // before it is written.
    for (std::size_t k = 0; k < count; ++k) {
        const std::size_t g = generation[k];
        assert(g < count);
        scratch[g] = bf16_to_float(policy_row[policy_index(packed[k])]);
    }

    softmax_in_place(scratch.data(), count);

    // And permute back. Now — and only now — the order is canonical.
    for (std::size_t k = 0; k < count; ++k) {
        priors[k] = scratch[generation[k]];
    }
}

// ---------------------------------------------------------------------------
// The dispatcher's view of the host-language evaluator
//
// Abstract because cpp/search.hpp holds no pybind11 include and must not: the
// search is compiled and tested without an interpreter (the Linux ASan/TSan
// runs, the C4 microbenchmarks), and a header that reached for `py::object`
// would end that. The one production implementation is `LiveEvaluator` in
// cpp/bindings.cpp; the tests add a deterministic in-process one so the live
// PATH can be exercised with no GIL and no model.
//
// The contract is deliberately narrow:
//
//   1. The dispatcher writes rows [0, count) of the token buffer.
//   2. It calls `run(count)` ONCE. Exactly one boundary crossing per batch.
//   3. It reads rows [0, count) of the policy and value buffers.
//
// `run` returns the nanoseconds it spent waiting to be allowed to run — the
// GIL acquire wait for the Python implementation. That number, and not the
// call's total duration, is the contention metric: torch and numpy re-acquire
// the GIL mid-call, so a handoff wait can land INSIDE the call and inflate a
// duration that has nothing to do with contention (scope §2.1).
// ---------------------------------------------------------------------------

// What one boundary crossing cost, split at the point where the two halves have
// different fixes. `acquire_wait_ns` is paid down only by bounding how long some
// OTHER thread holds the interpreter (sys.setswitchinterval, or moving that
// thread's work out of Python); `call_ns` is paid down by making the callback
// itself cheaper. Reporting one number for both is how a GIL problem gets
// misdiagnosed as a slow model.
struct EvalTiming {
    std::int64_t acquire_wait_ns = 0;
    std::int64_t call_ns = 0;
};

class BatchEvaluator {
public:
    virtual ~BatchEvaluator() = default;

    // Rows the three buffers hold. The dispatcher never submits more.
    virtual std::size_t max_batch() const noexcept = 0;

    // Row `i` of the int32 token buffer: 68 writable ints.
    virtual std::int32_t *token_row(std::size_t i) noexcept = 0;

    // Evaluate rows [0, count). One boundary crossing, no exceptions.
    virtual EvalTiming run(std::size_t count) = 0;

    // Row `i` of the bf16 policy buffer: kPolicySize raw bf16 bit patterns.
    virtual const std::uint16_t *policy_row(std::size_t i) const noexcept = 0;

    // The network's ABSOLUTE (White-POV) value for row `i`. The side-to-move
    // flip is search's job and happens in `mover_value`, exactly as it does in
    // the replay path.
    virtual float value_at(std::size_t i) const noexcept = 0;
};

}  // namespace guofish

#endif  // GUOFISH_EVALUATOR_HPP
