"""Phase 1 audit of the Lichess eval database (`lichess_db_eval.jsonl.zst`).

READ-ONLY. Streams the compressed file, never decompresses to disk, never holds
more than one line in memory. The only artifact written is a markdown report.

The headline output is the POV convention (section 1): whether `cp` scores are
side-to-move relative or White relative. Everything downstream (value sign,
policy softmax ordering) depends on getting this right, so it is determined
empirically and cross-checked two independent ways:

  (a) PV monotonicity. Engines emit multi-PV lists best-first. For a
      Black-to-move position, a side-to-move-relative score sequence must be
      non-increasing (best-for-Black first); a White-relative one must be
      non-decreasing (best-for-Black == worst-for-White == smallest cp).
      White-to-move positions are audited too as a CONTROL: both conventions
      predict non-increasing there, so if White-to-move is not overwhelmingly
      descending, the "best-first" premise itself is wrong and the whole test
      is void.

  (b) Mate replay. For PVs scored `mate`, the line is actually played out on a
      board and we observe who gets checkmated. Under a side-to-move-relative
      convention a positive mate means the side to move delivers it; under a
      White-relative convention a positive mate always means White delivers it.
      This test is independent of move ordering.

Usage:
    python data/multiPV/audit_eval_db.py --source path/to/lichess_db_eval.jsonl.zst
    python data/multiPV/audit_eval_db.py --source ... --limit 2000000 --report out.md
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import chess

# Import the project tokenizer rather than reimplementing it.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from data.pgn_parallel import _board_to_tokens, SEQ_LENGTH, VOCAB_SIZE  # noqa: E402

# Mate -> pseudo-centipawn mapping. Matches the Phase 2 rule so the audit
# measures the same ordering the converter will see: distance is preserved, so
# M1 outranks M8, and every mate outranks every non-mate cp score.
MATE_BASE = 10000
MATE_STEP = 100
MATE_MAX_DISTANCE = 50


def mate_to_cp(mate: int) -> int:
    """sign * (10000 - 100 * min(|mate|, 50)) - the mapping given in the brief."""
    sign = 1 if mate > 0 else -1
    return sign * (MATE_BASE - MATE_STEP * min(abs(mate), MATE_MAX_DISTANCE))


def mate_to_cp_disjoint(mate: int, cp_clamp: int) -> int:
    """Same distance ordering, but the mate band sits strictly ABOVE the clamped
    cp band, so no mate can ever sort below a large centipawn score.

    The brief's mapping bottoms out at 5000 (|mate| >= 50) while this database
    emits cp magnitudes well past that, which interleaves the two bands and
    breaks the `rel[0] == max(rel)` invariant. See section 6 of the report.
    """
    sign = 1 if mate > 0 else -1
    steps = MATE_MAX_DISTANCE + 1 - min(abs(mate), MATE_MAX_DISTANCE)
    return sign * (cp_clamp + MATE_STEP * steps)


def pv_score(pv: dict) -> int | None:
    """Single comparable score for a PV, or None if it carries neither field.

    `cp` and `mate` are read as a unit here so a malformed PV can never
    contribute a score without also contributing a move (the Phase 2 rule).
    """
    if "cp" in pv and pv["cp"] is not None:
        return int(pv["cp"])
    if "mate" in pv and pv["mate"] is not None:
        return mate_to_cp(int(pv["mate"]))
    return None


def classify_monotonic(scores: list[int]) -> str:
    """'descending' | 'ascending' | 'flat' | 'non-monotonic'."""
    if len(scores) < 2:
        return "flat"
    non_increasing = all(b <= a for a, b in zip(scores, scores[1:]))
    non_decreasing = all(b >= a for a, b in zip(scores, scores[1:]))
    if non_increasing and non_decreasing:
        return "flat"  # all equal - carries no directional information
    if non_increasing:
        return "descending"
    if non_decreasing:
        return "ascending"
    return "non-monotonic"


def stream_lines(source: Path, limit: int):
    """Yield decoded lines from a .zst (streaming) or plain .jsonl file."""
    if source.suffix == ".zst":
        import zstandard

        with open(source, "rb") as fh:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text = io.TextIOWrapper(reader, encoding="utf-8")
                for i, line in enumerate(text):
                    if limit and i >= limit:
                        return
                    yield line
    else:
        with open(source, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if limit and i >= limit:
                    return
                yield line


def first_move_of(board: chess.Board, line: str) -> chess.Move | None:
    """Parse the first UCI token of a PV line. Handles UCI_Chess960 castling
    (king-takes-rook, e.g. e1h1 -> e1g1) via board.parse_uci."""
    if not line:
        return None
    token = line.split(" ", 1)[0]
    try:
        return board.parse_uci(token)
    except Exception:
        return None


def replay_mate(board: chess.Board, line: str) -> chess.Color | None:
    """Play out a PV and return the color that DELIVERED checkmate, or None."""
    b = board.copy()
    try:
        for token in line.split():
            b.push(b.parse_uci(token))
            if b.is_checkmate():
                return not b.turn  # side that just moved
    except Exception:
        return None
    return None


def run_smoke_tests() -> list[str]:
    """Section 5. Self-contained; needs no eval database."""
    out: list[str] = []

    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6",
        "8/8/8/8/8/5k2/6q1/7K w - -",
    ]
    ok = 0
    for f in fens:
        try:
            chess.Board(f)
            ok += 1
        except Exception as exc:
            out.append(f"  - FAIL 4-field FEN `{f}`: {type(exc).__name__}: {exc}")
    out.append(f"- **4-field FEN accepted by `chess.Board`**: {ok}/{len(fens)} PASS "
               f"(missing halfmove/fullmove default to 0/1)")

    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq -")
    pairs = [("e1h1", "e1g1"), ("e1a1", "e1c1")]
    res = []
    for uci, expect in pairs:
        got = b.parse_uci(uci).uci()
        res.append(f"`{uci}`->`{got}`" + ("" if got == expect else f" (EXPECTED {expect})"))
    out.append(f"- **UCI_Chess960 castling normalization**: {', '.join(res)} "
               f"{'PASS' if all(b.parse_uci(u).uci() == e for u, e in pairs) else 'FAIL'}")

    rng = random.Random(0)
    board = chess.Board()
    bad = 0
    checked = 0
    for _ in range(3000):
        if board.is_game_over():
            board = chess.Board()
        board.push(rng.choice(list(board.legal_moves)))
        toks = _board_to_tokens(board)
        checked += 1
        if len(toks) != SEQ_LENGTH or not all(0 <= t < VOCAB_SIZE for t in toks):
            bad += 1
    out.append(f"- **`_board_to_tokens`**: {checked} random positions, "
               f"len=={SEQ_LENGTH} and all tokens in [0,{VOCAB_SIZE - 1}] -> "
               f"{bad} failures {'PASS' if bad == 0 else 'FAIL'}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1 audit of lichess_db_eval.jsonl.zst")
    ap.add_argument("--source", type=Path, required=True,
                    help="path to lichess_db_eval.jsonl.zst (or plain .jsonl)")
    ap.add_argument("--limit", type=int, default=2_000_000,
                    help="lines to stream (0 = whole file). Default 2000000")
    ap.add_argument("--report", type=Path,
                    default=Path(__file__).resolve().parent / "phase1_audit_report.md")
    ap.add_argument("--move-sample-rate", type=float, default=0.1,
                    help="fraction of positions given full move-level parsing "
                         "(promotion collisions, mate replay). Default 0.1")
    ap.add_argument("--value-min-depths", type=int, nargs="+", default=[16, 20, 24])
    ap.add_argument("--policy-min-depths", type=int, nargs="+", default=[12, 16, 20])
    ap.add_argument("--cp-clamp", type=int, default=10000,
                    help="cp magnitude clamp used by the disjoint-band mate mapping "
                         "in section 6. Default 10000")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.source.exists():
        print(f"ERROR: source not found: {args.source}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)

    # --- accumulators -------------------------------------------------------
    n_lines = 0
    n_parse_errors = 0
    n_no_evals = 0

    # 1. POV
    pov_black = Counter()   # classification -> count, black to move
    pov_white = Counter()   # control, white to move
    mate_replay = Counter()  # 'stm_relative' | 'white_relative' | 'inconclusive'

    # 2. histograms
    depth_hist = Counter()
    npv_hist = Counter()
    maxdepth_multipv_hist = Counter()
    n_deepest_has_1pv = 0
    n_positions_with_multipv = 0

    # 3. yield
    yield_counts: dict[tuple[int, int], list[int]] = {
        (v, p): [0, 0] for v in args.value_min_depths for p in args.policy_min_depths
    }

    # 4. mate / collisions
    n_pvs_total = 0
    n_pvs_mate = 0
    n_pvs_missing_score = 0
    n_move_sampled = 0
    n_collision_positions = 0
    n_pv_unparseable = 0

    # 6. Phase-2 pre-flight
    max_abs_cp = 0
    n_order_blocks = 0
    n_mixed_blocks = 0
    n_viol_spec = 0
    n_viol_spec_mixed = 0
    n_viol_disjoint = 0
    n_pos_chess960 = 0

    t0 = time.time()
    for raw in stream_lines(args.source, args.limit):
        n_lines += 1
        try:
            rec = json.loads(raw)
        except Exception:
            n_parse_errors += 1
            continue

        fen = rec.get("fen")
        evals = rec.get("evals") or []
        if not fen or not evals:
            n_no_evals += 1
            continue

        # side to move straight off the FEN string - far cheaper than building a Board
        parts = fen.split()
        black_to_move = len(parts) > 1 and parts[1] == "b"

        deepest_depth = -1
        deepest_npv = 0
        best_multipv_depth = -1
        has_multipv = False

        for ev in evals:
            depth = int(ev.get("depth", -1))
            pvs = ev.get("pvs") or []
            npv = len(pvs)
            depth_hist[depth] += 1
            npv_hist[npv] += 1

            if depth > deepest_depth:
                deepest_depth = depth
                deepest_npv = npv

            if npv >= 2:
                has_multipv = True
                if depth > best_multipv_depth:
                    best_multipv_depth = depth

                # --- sections 1a + 6: build both score vectors in one pass ---
                # s_spec uses the brief's mate mapping; s_disj uses the
                # disjoint-band mapping. Both are built PV-by-PV as a unit, so a
                # PV lacking a score aborts the whole block rather than
                # desynchronising the vectors.
                s_spec: list[int] = []
                s_disj: list[int] = []
                complete = True
                blk_mate = blk_cp = False
                for pv in pvs:
                    if pv.get("cp") is not None:
                        v = int(pv["cp"])
                        blk_cp = True
                        if abs(v) > max_abs_cp:
                            max_abs_cp = abs(v)
                        s_spec.append(v)
                        s_disj.append(max(-args.cp_clamp, min(args.cp_clamp, v)))
                    elif pv.get("mate") is not None:
                        m = int(pv["mate"])
                        blk_mate = True
                        s_spec.append(mate_to_cp(m))
                        s_disj.append(mate_to_cp_disjoint(m, args.cp_clamp))
                    else:
                        complete = False
                        break

                if complete:
                    cls = classify_monotonic(s_spec)
                    (pov_black if black_to_move else pov_white)[cls] += 1

                    # rel[0] == max(rel) invariant, in stm-relative space.
                    # Source is White-relative (section 1), so negate for Black.
                    n_order_blocks += 1
                    mixed = blk_mate and blk_cp
                    if mixed:
                        n_mixed_blocks += 1
                    r_spec = [-x for x in s_spec] if black_to_move else s_spec
                    r_disj = [-x for x in s_disj] if black_to_move else s_disj
                    if r_spec[0] != max(r_spec):
                        n_viol_spec += 1
                        if mixed:
                            n_viol_spec_mixed += 1
                    if r_disj[0] != max(r_disj):
                        n_viol_disjoint += 1

            for pv in pvs:
                n_pvs_total += 1
                if pv.get("mate") is not None:
                    n_pvs_mate += 1
                if pv_score(pv) is None:
                    n_pvs_missing_score += 1

        if deepest_depth >= 0:
            if deepest_npv == 1:
                n_deepest_has_1pv += 1
            if has_multipv:
                n_positions_with_multipv += 1
                maxdepth_multipv_hist[best_multipv_depth] += 1

        # --- section 3: yield ---
        for (vmin, pmin), cell in yield_counts.items():
            if deepest_depth >= vmin:
                cell[0] += 1
                if best_multipv_depth >= pmin:
                    cell[1] += 1

        # --- sections 1b + 4: move-level work on a subsample ---
        if rng.random() < args.move_sample_rate:
            try:
                board = chess.Board(fen)
            except Exception:
                n_pv_unparseable += 1
                continue
            n_move_sampled += 1
            collided = False
            is_960 = False
            board960: chess.Board | None = None
            for ev in evals:
                seen: set[tuple[int, int]] = set()
                for pv in ev.get("pvs") or []:
                    mv = first_move_of(board, pv.get("line", ""))
                    if mv is None:
                        n_pv_unparseable += 1
                        # A first move that is illegal on a standard board but
                        # legal on a chess960 one is king-takes-rook castling in
                        # a shuffled position - i.e. the record is Chess960, not
                        # standard chess. Size it here; Phase 2 must decide to
                        # skip or accommodate.
                        if board960 is None:
                            try:
                                board960 = chess.Board(fen, chess960=True)
                            except Exception:
                                board960 = board  # sentinel: cannot build
                        if board960 is not board and \
                                first_move_of(board960, pv.get("line", "")) is not None:
                            is_960 = True
                        continue
                    key = (mv.from_square, mv.to_square)
                    if key in seen:
                        collided = True
                    seen.add(key)

                    mate = pv.get("mate")
                    if mate is not None and int(mate) != 0:
                        winner = replay_mate(board, pv.get("line", ""))
                        if winner is None:
                            mate_replay["inconclusive"] += 1
                        else:
                            positive = int(mate) > 0
                            stm_delivers = winner == board.turn
                            white_delivers = winner == chess.WHITE
                            if positive == stm_delivers and positive != white_delivers:
                                mate_replay["stm_relative"] += 1
                            elif positive == white_delivers and positive != stm_delivers:
                                mate_replay["white_relative"] += 1
                            elif positive == stm_delivers and positive == white_delivers:
                                mate_replay["consistent_with_both"] += 1
                            else:
                                mate_replay["consistent_with_neither"] += 1
            if collided:
                n_collision_positions += 1
            if is_960:
                n_pos_chess960 += 1

        if n_lines % 250_000 == 0:
            rate = n_lines / max(time.time() - t0, 1e-9)
            print(f"  {n_lines:,} lines  ({rate:,.0f}/s)", file=sys.stderr, flush=True)

    elapsed = time.time() - t0
    report = build_report(args, locals())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report written to {args.report}]  ({elapsed:.1f}s)", file=sys.stderr)
    return 0


def _pct(num: int, den: int) -> str:
    return f"{100.0 * num / den:.3f}%" if den else "n/a"


def _hist_table(counter: Counter, label: str, top: int = 40) -> str:
    if not counter:
        return f"_(no {label} data)_\n"
    total = sum(counter.values())
    rows = [f"| {label} | count | share |", "|---:|---:|---:|"]
    for key in sorted(counter)[:top]:
        rows.append(f"| {key} | {counter[key]:,} | {_pct(counter[key], total)} |")
    if len(counter) > top:
        rows.append(f"| _({len(counter) - top} more)_ | | |")
    return "\n".join(rows) + "\n"


def build_report(args, ns: dict) -> str:
    g = ns.get
    n_lines = g("n_lines")
    pov_black: Counter = g("pov_black")
    pov_white: Counter = g("pov_white")
    mate_replay: Counter = g("mate_replay")

    desc = pov_black.get("descending", 0)
    asc = pov_black.get("ascending", 0)
    flat = pov_black.get("flat", 0)
    nonmono = pov_black.get("non-monotonic", 0)
    decisive = desc + asc
    total_black = decisive + flat + nonmono

    if decisive == 0:
        verdict = "**INCONCLUSIVE** - no directionally informative Black-to-move blocks."
    elif desc / decisive > 0.99:
        verdict = ("**SIDE-TO-MOVE RELATIVE**: Black-to-move multi-PV scores are "
                   "overwhelmingly DESCENDING.")
    elif asc / decisive > 0.99:
        verdict = ("**WHITE RELATIVE (absolute)**: Black-to-move multi-PV scores are "
                   "overwhelmingly ASCENDING.")
    else:
        verdict = (f"**INCONCLUSIVE - STOP.** Split is descending={_pct(desc, decisive)} / "
                   f"ascending={_pct(asc, decisive)}. Neither convention dominates; "
                   f"do NOT guess. Investigate before Phase 2.")

    wdesc = pov_white.get("descending", 0)
    wasc = pov_white.get("ascending", 0)
    wdecisive = wdesc + wasc
    control = (f"descending={_pct(wdesc, wdecisive)}, ascending={_pct(wasc, wdecisive)} "
               f"(of {wdecisive:,} directional blocks)")
    control_ok = wdecisive > 0 and wdesc / wdecisive > 0.99

    lines = [
        "# Phase 1 audit - Lichess eval DB",
        "",
        f"- source: `{args.source}`",
        f"- lines streamed: **{n_lines:,}** (limit {args.limit or 'none'})",
        f"- JSON parse errors: {g('n_parse_errors'):,} | records with no evals: {g('n_no_evals'):,}",
        f"- move-level subsample rate: {args.move_sample_rate} "
        f"({g('n_move_sampled'):,} positions fully parsed)",
        "",
        "## 1. POV convention (headline)",
        "",
        f"### Verdict: {verdict}",
        "",
        "**Test (a) - multi-PV monotonicity, Black to move**",
        "",
        "| pattern | count | share of directional |",
        "|---|---:|---:|",
        f"| descending (=> stm-relative) | {desc:,} | {_pct(desc, decisive)} |",
        f"| ascending (=> White-relative) | {asc:,} | {_pct(asc, decisive)} |",
        f"| flat (all equal, uninformative) | {flat:,} | - |",
        f"| non-monotonic (unordered) | {nonmono:,} | - |",
        "",
        f"- directional blocks: **{decisive:,}**",
        f"- ambiguity rate (flat + non-monotonic) / all Black multi-PV blocks: "
        f"**{_pct(flat + nonmono, total_black)}**",
        "",
        f"**Control - White to move** (both conventions predict descending): {control} "
        f"-> {'OK, best-first ordering holds' if control_ok else 'WARNING: best-first ordering NOT confirmed; test (a) is void'}",
        "",
        "**Test (b) - mate replay** (independent of move ordering)",
        "",
        _hist_table(mate_replay, "outcome") if mate_replay else "_(no mate PVs in subsample)_\n",
        "## 2. Histograms",
        "",
        "### depth (all eval blocks)",
        _hist_table(g("depth_hist"), "depth"),
        "### PVs per eval block",
        _hist_table(g("npv_hist"), "n_pvs"),
        "### max depth among evals with >=2 PVs (per position)",
        _hist_table(g("maxdepth_multipv_hist"), "depth"),
        f"- positions whose DEEPEST eval has exactly 1 PV: **{g('n_deepest_has_1pv'):,}** "
        f"({_pct(g('n_deepest_has_1pv'), n_lines)})",
        f"- positions with at least one >=2-PV block: **{g('n_positions_with_multipv'):,}** "
        f"({_pct(g('n_positions_with_multipv'), n_lines)})",
        "",
        "## 3. Yield estimates",
        "",
        "`value` = positions whose deepest eval meets value_min_depth (a sample is emitted). ",
        "`policy` = subset that also has a >=2-PV block at policy_min_depth (has_policy=1).",
        "",
        "| value_min_depth | policy_min_depth | value samples | share | with policy | policy share of emitted |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for (vmin, pmin), (vc, pc) in sorted(g("yield_counts").items()):
        lines.append(f"| {vmin} | {pmin} | {vc:,} | {_pct(vc, n_lines)} | {pc:,} | {_pct(pc, vc)} |")

    n_pvs_total = g("n_pvs_total")
    lines += [
        "",
        "## 4. Mate scores and promotion collisions",
        "",
        f"- total PVs seen: **{n_pvs_total:,}**",
        f"- PVs scored `mate`: **{g('n_pvs_mate'):,}** ({_pct(g('n_pvs_mate'), n_pvs_total)})",
        f"- PVs with NEITHER `cp` nor `mate`: **{g('n_pvs_missing_score'):,}** "
        f"({_pct(g('n_pvs_missing_score'), n_pvs_total)}) - these must drop move+score together",
        f"- PV first-moves that failed to parse: **{g('n_pv_unparseable'):,}**",
        f"- positions where two PVs in one block share a (from,to) pair "
        f"(**promotion collision**): **{g('n_collision_positions'):,}** "
        f"({_pct(g('n_collision_positions'), g('n_move_sampled'))} of the move-parsed subsample)",
        "",
        "## 5. Smoke tests",
        "",
    ]
    lines += run_smoke_tests()

    # --- section 6 ---
    nb = g("n_order_blocks") or 1
    vs_ = g("n_viol_spec")
    vd = g("n_viol_disjoint")
    clamp = args.cp_clamp
    spec_rate = 100.0 * vs_ / nb
    disj_rate = 100.0 * vd / nb
    lines += [
        "",
        "## 6. Phase 2 pre-flight",
        "",
        "### 6a. `rel[0] == max(rel)` violation rate",
        "",
        "Phase 2 asserts that, after conversion to stm-relative, PV 0 is the best "
        "move, skipping the sample on violation and hard-failing the job above 0.1%.",
        "",
        f"- multi-PV blocks checked: **{g('n_order_blocks'):,}**",
        f"- max |cp| observed in the DB: **{g('max_abs_cp'):,}**",
        f"- brief's mate mapping occupies the band "
        f"**[{MATE_BASE - MATE_STEP * MATE_MAX_DISTANCE:,}, {MATE_BASE - MATE_STEP:,}]** "
        f"-> it OVERLAPS the cp range, so a large cp can outrank a mate",
        "",
        "| mate mapping | violations | rate | 0.1% gate |",
        "|---|---:|---:|---|",
        f"| brief: `sign*(10000-100*min(|m|,50))` | {vs_:,} | **{spec_rate:.4f}%** | "
        f"**{'PASS' if spec_rate <= 0.1 else 'TRIPS'}** |",
        f"| disjoint bands (cp clamped to +-{clamp:,}, mates above) | {vd:,} | "
        f"**{disj_rate:.4f}%** | **{'PASS' if disj_rate <= 0.1 else 'TRIPS'}** |",
        "",
        f"- blocks mixing `mate` and `cp` PVs: **{g('n_mixed_blocks'):,}** "
        f"({_pct(g('n_mixed_blocks'), g('n_order_blocks'))})",
        f"- share of the brief-mapping violations that occur in mixed blocks: "
        f"**{_pct(g('n_viol_spec_mixed'), vs_)}** - this is the root cause",
        "",
        "### 6b. Chess960 contamination",
        "",
        f"- positions whose PV first move is illegal on a standard board but legal "
        f"with `chess960=True`: **{g('n_pos_chess960'):,}** "
        f"({_pct(g('n_pos_chess960'), g('n_move_sampled'))} of the move-parsed subsample)",
        "- these are king-takes-rook castling moves in shuffled positions; the record "
        "is Chess960, not standard chess",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
