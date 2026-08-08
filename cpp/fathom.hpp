// GuoFish C7 — the native Syzygy backend, on jdart1/Fathom.
//
// cpp/tablebase.hpp defines what a backend must answer and what the engine does
// with the answer. This is the answer: real .rtbw/.rtbz decoding, no Python in
// the loop, usable from the search threads C9 introduces.
//
// Fathom is authorised as a third dependency for this chunk (Global Rule 7) and
// pinned to c9c6fef in CMakeLists.txt. The pin matters more here than for the
// other two: a tablebase probe is a lookup whose ANSWER is what is under test,
// so a floating revision could change the WDL a position reports through an
// en-passant convention or a fifty-move adjustment, with nothing in this
// repository changing, and the symptom would be an engine playing a different
// endgame move rather than a build error.
//
//
// FIVE PLACES FATHOM AND python-chess DO NOT SPEAK THE SAME LANGUAGE
// -----------------------------------------------------------------
// The reference probes through `chess.syzygy`. Every one of these is a place
// where a naive translation is quietly wrong, and each is handled below and
// then MEASURED against the reference over the real tables by
// tests/test_c7_cache.py — argument first, measurement second.
//
// 1. THE WDL SCALE IS DIFFERENT. Fathom returns TB_LOSS=0, TB_BLESSED_LOSS=1,
//    TB_DRAW=2, TB_CURSED_WIN=3, TB_WIN=4. Syzygy's own scale — the one
//    python-chess returns and the one `wdl_to_value` divides by two — is
//    -2..+2. The conversion is `syzygy = fathom - 2`, and getting it wrong maps
//    a draw to a cursed win rather than crashing.
//
// 2. `tb_probe_wdl` REFUSES A NON-ZERO HALFMOVE CLOCK. Its public wrapper is
//
//        if (_castling != 0) return TB_RESULT_FAILED;
//        if (_rule50 != 0)   return TB_RESULT_FAILED;
//
//    which is defensible — a WDL that ignores the fifty-move rule is not the
//    result of a position that is 60 plies into a shuffle — but it is NOT what
//    python-chess's `probe_wdl` does, and the reference's mode 2 probes leaves
//    at whatever clock they carry. A backend that passed the real clock through
//    would miss on almost every leaf and mode 2 would silently never fire.
//
//    So this passes rule50 = 0 deliberately, which makes `tb_probe_wdl`
//    reproduce python-chess's clock-independent probe exactly. THIS IS NOT THE
//    CACHE-POISONING DEFECT REAPPEARING. It is the same fact that causes it —
//    a raw WDL is clock-independent — handled in the one place where that is
//    correct. The value is still a `TablebaseValue`, it is still applied
//    tree-locally at the leaf, and it still cannot enter a position-keyed
//    cache. The clock is *why* it must not be cached; it is not an input to the
//    probe on either side of the comparison.
//
// 3. CASTLING RIGHTS ARE A MISS, NOT AN ERROR. Fathom returns
//    TB_RESULT_FAILED; python-chess raises. The reference catches and keeps the
//    neural value, so both become nullopt here. (No position with castling
//    rights is in a five-man table anyway — but a FEN can claim them.)
//
// 4. DTZ IS UNSIGNED IN FATHOM AND SIGNED IN python-chess. `TB_GET_DTZ` is a
//    magnitude; `probe_dtz` returns positive when the side to move wins and
//    negative when it loses. Mode 1's ranking subtracts DTZ, so an unsigned
//    value would make a losing side prefer the fastest loss. The sign is
//    restored from the WDL below.
//
// 5. DTZ COMES FROM A DIFFERENT ENTRY POINT. Fathom has no bare `probe_dtz` in
//    its public API — DTZ arrives inside `tb_probe_root`, which also generates
//    moves and is documented NOT thread safe. That is acceptable because mode 1
//    is a UCI-layer root bypass called once per move, and mode 2 — the one on
//    the search path — needs only WDL. `probe_dtz` below says so rather than
//    leaving a future reader to discover it under ThreadSanitizer.
//
//
// ONE OPEN TABLEBASE PER PROCESS
// ------------------------------
// `tb_init`/`tb_free` operate on file-scope state inside Fathom; there is no
// handle. Two `FathomProber`s would therefore share one set of tables, and the
// second's destructor would free the first's. Rather than document that and
// hope, the constructor refuses to open a second one and says why. This is a
// property of the library, not a design choice, and it is the reason
// `open_tablebase` returns something the caller must keep alive.

#ifndef GUOFISH_FATHOM_HPP
#define GUOFISH_FATHOM_HPP

#include "tablebase.hpp"
#include "terminal.hpp"
#include "tokens.hpp"

extern "C" {
#include <tbprobe.h>
}

#include <atomic>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace guofish {

// Fathom's TB_LOSS..TB_WIN is 0..4; Syzygy's, and python-chess's, is -2..+2.
inline constexpr int kFathomWdlOffset = 2;

class FathomProber final : public TablebaseProber {
public:
    // `path` is Fathom's tablebase PATH string — a directory, or several
    // separated by the platform's path separator.
    //
    // Throws if the tables cannot be opened, and — deliberately — if another
    // FathomProber is already alive. See the header comment: Fathom's state is
    // file-scope, so "another one" means "the same one".
    explicit FathomProber(const std::string &path) {
        bool expected = false;
        if (!open_.compare_exchange_strong(expected, true)) {
            throw std::runtime_error(
                "guofish::FathomProber: a tablebase is already open. Fathom keeps its "
                "state in file scope — tb_init and tb_free take no handle — so a second "
                "instance would share the first's tables and free them from under it. "
                "Keep one prober and hand it to whoever needs it, as the reference hands "
                "one chess.syzygy.Tablebase to the UCI layer and to every search.");
        }
        if (!tb_init(path.c_str())) {
            open_.store(false);
            throw std::runtime_error("guofish::FathomProber: tb_init failed for path: " + path);
        }
        // tb_init returns true with TB_LARGEST == 0 when the directory exists
        // but holds no tables. That is a configuration mistake rather than an
        // I/O error, and letting it through would produce a prober that misses
        // on everything while reporting itself open — which is exactly the
        // "tablebases are on but never fire" state that is hardest to notice.
        if (TB_LARGEST == 0) {
            tb_free();
            open_.store(false);
            throw std::runtime_error(
                "guofish::FathomProber: no tablebase files found under: " + path +
                "\n  tb_init succeeded but TB_LARGEST is 0, so every probe would miss.");
        }
        largest_ = static_cast<int>(TB_LARGEST);
        path_ = path;
    }

    ~FathomProber() override {
        tb_free();
        open_.store(false);
    }

    FathomProber(const FathomProber &) = delete;
    FathomProber &operator=(const FathomProber &) = delete;

    // The most men any loaded table covers. assets/syzygy is a 5-man set.
    int largest() const noexcept { return largest_; }
    const std::string &path() const noexcept { return path_; }

    std::string backend() const override {
        return "fathom(" + std::to_string(largest_) + "-man)";
    }

    // python-chess's `probe_wdl`: a raw Syzygy WDL in [-2, +2] from the side to
    // move's perspective, independent of the halfmove clock. nullopt on a miss.
    std::optional<int> probe_wdl(const ParsedFen &parsed, int halfmove_clock) const override {
        // The clock is accepted and not used — see note 2 in this file's header.
        // Naming it and ignoring it is deliberate: a silent omission here would
        // look like an oversight to the next reader, and the reason it is
        // correct is not obvious.
        (void)halfmove_clock;
        if (!probeable(parsed)) {
            return std::nullopt;
        }
        const unsigned result =
            tb_probe_wdl(white(parsed), black(parsed), parsed.placement.kings,
                         parsed.placement.queens, parsed.placement.rooks,
                         parsed.placement.bishops, parsed.placement.knights,
                         parsed.placement.pawns,
                         /*rule50=*/0u, /*castling=*/0u, ep_square(parsed),
                         parsed.white_to_move);
        if (result == TB_RESULT_FAILED) {
            return std::nullopt;
        }
        return static_cast<int>(result) - kFathomWdlOffset;
    }

    // python-chess's `probe_dtz`: SIGNED distance to zero from the side to
    // move's perspective — positive when it is winning, negative when losing,
    // zero on a draw. nullopt on a miss.
    //
    // NOT THREAD SAFE, because `tb_probe_root` is not: Fathom's own header says
    // "this function should only be called once at the root per search". Mode 1
    // is exactly that. Nothing on the search path calls this — mode 2 uses WDL
    // alone — and C9 must keep it that way.
    std::optional<int> probe_dtz(const ParsedFen &parsed, int halfmove_clock) const override {
        if (!probeable(parsed)) {
            return std::nullopt;
        }
        // The clock IS passed here. Unlike the WDL probe, tb_probe_root accepts
        // it and uses it to decide whether a win is still reachable inside the
        // fifty-move horizon, which is what a root bypass wants to know.
        const unsigned result = tb_probe_root(
            white(parsed), black(parsed), parsed.placement.kings, parsed.placement.queens,
            parsed.placement.rooks, parsed.placement.bishops, parsed.placement.knights,
            parsed.placement.pawns, static_cast<unsigned>(halfmove_clock < 0 ? 0 : halfmove_clock),
            /*castling=*/0u, ep_square(parsed), parsed.white_to_move, nullptr);
        if (result == TB_RESULT_FAILED) {
            return std::nullopt;
        }
        // Checkmate and stalemate come back as results with DTZ 0. python-chess
        // answers 0 for a stalemate too, and for a checkmate its caller never
        // asks (the reference's mode 1 tests is_checkmate first and skips the
        // probe entirely), so 0 is the agreeing answer in both cases.
        const int magnitude = static_cast<int>(TB_GET_DTZ(result));
        const int wdl = static_cast<int>(TB_GET_WDL(result)) - kFathomWdlOffset;
        if (wdl > 0) {
            return magnitude;
        }
        if (wdl < 0) {
            return -magnitude;
        }
        return 0;
    }

private:
    // Everything both probes must agree about before Fathom is called at all.
    bool probeable(const ParsedFen &parsed) const {
        if (piece_count(parsed.placement) > largest_) {
            return false;
        }
        // Fathom refuses a position with castling rights, and python-chess
        // raises on one; the reference catches and keeps the neural value. Both
        // become a miss, and checking here rather than passing the rights
        // through keeps the two probes' answers identical on the point.
        if (parsed.castling != 0) {
            return false;
        }
        return true;
    }

    static std::uint64_t white(const ParsedFen &parsed) noexcept {
        return parsed.placement.occupied_white;
    }
    static std::uint64_t black(const ParsedFen &parsed) noexcept {
        return parsed.placement.occupied_black;
    }

    // Fathom wants a square index, or 0 for "none". Square 0 is a1, which can
    // never be an en-passant square, so the sentinel is unambiguous — but the
    // conversion from our -1 still has to happen and is easy to forget.
    //
    // THE RAW EN-PASSANT SQUARE IS WHAT IS PASSED, which is the discipline this
    // port has kept since C2 and is also correct for the probe: Fathom, like
    // python-chess, generates en-passant captures itself when scoring the
    // position, so naming a square no pawn can act on costs nothing, while
    // omitting one that a pawn CAN act on would change the answer.
    static unsigned ep_square(const ParsedFen &parsed) noexcept {
        return parsed.ep_square < 0 ? 0u : static_cast<unsigned>(parsed.ep_square);
    }

    int largest_ = 0;
    std::string path_;

    // One open tablebase per process; see the header comment.
    static std::atomic<bool> open_;
};

// Defined inline so this stays a header-only unit like the rest of cpp/.
inline std::atomic<bool> FathomProber::open_{false};

}  // namespace guofish

#endif  // GUOFISH_FATHOM_HPP
