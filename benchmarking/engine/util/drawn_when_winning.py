"""Find games GuoFish drew by repetition / 50-move rule while it was winning.

Given a PGN of GuoFish-vs-Stockfish games, this replays each drawn game,
classifies *how* the draw happened on the board (threefold repetition,
fifty-move rule, stalemate, insufficient material), and uses the Stockfish
executable to evaluate whether GuoFish was actually in a winning position when
the draw was sealed.

"Winning" is an arbitrary centipawn threshold (default >= +0.50 pawns from
GuoFish's point of view). Because a repetition/50-move draw tends to flatten
the final eval to ~0, we scan a window of plies at the end of the game and take
the *peak* GuoFish eval in that window -- i.e. the best position GuoFish reached
and then failed to convert.

The opponent is always named "Stockfish"; whichever side is not "Stockfish" is
treated as the GuoFish engine, so this works regardless of the GuoFish label.

Usage:
    python benchmarking/engine/util/drawn_when_winning.py \
        --pgn benchmarking/engine/games/v4/guofishv4_2_2.5_80_2800sf.pgn
"""

import argparse
import os
import sys

import chess
import chess.engine
import chess.pgn


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_STOCKFISH = os.path.join(REPO_ROOT, "stockfish-windows-x86-64-avx2.exe")
DEFAULT_PGN = os.path.join(
    REPO_ROOT, "benchmarking", "engine", "games", "v4",
    "guofishv4_2_2.5_80_2800sf.pgn",
)
OPPONENT_NAME = "Stockfish"

# Draw types we care about for "threw away a win". Stalemate / insufficient
# material are board draws too but are not "repetition/50-move", so they are
# classified separately and excluded from the headline count.
RELEVANT_DRAWS = {"repetition", "fifty_move"}

MATE_SCORE_CP = 100_000  # cp value assigned to a forced mate when scoring


def classify_draw(board: chess.Board) -> str:
    """Classify how the final position is a draw on the board."""
    if board.is_checkmate():
        return "checkmate"  # not a draw; defensive
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient_material"
    # Repetition before fifty-move: repetition is the more common "throw away".
    if board.is_repetition(3):
        return "repetition"
    if board.is_fifty_moves():
        return "fifty_move"
    return "other"


def guofish_color(game: chess.pgn.Game) -> chess.Color | None:
    """Return the color GuoFish played, or None if both/neither is Stockfish."""
    white = game.headers.get("White", "")
    black = game.headers.get("Black", "")
    if white == OPPONENT_NAME and black != OPPONENT_NAME:
        return chess.BLACK
    if black == OPPONENT_NAME and white != OPPONENT_NAME:
        return chess.WHITE
    return None


def score_cp_white(info_score: chess.engine.PovScore) -> int:
    """Centipawns from White's POV, with mates mapped to large values."""
    white = info_score.white()
    return white.score(mate_score=MATE_SCORE_CP)


def peak_guofish_eval(engine: chess.engine.SimpleEngine, board_states: list,
                      gf_color: chess.Color, window: int, depth: int):
    """Peak GuoFish-POV eval over the last `window` plies.

    `board_states` is the list of board positions reached after each move
    (excluding the very final, already-drawn position, which has no legal
    continuation to weigh). Returns (best_cp, best_index) where best_cp is from
    GuoFish's perspective (positive == good for GuoFish), or (None, None).
    """
    if not board_states:
        return None, None
    start = max(0, len(board_states) - window)
    best_cp = None
    best_idx = None
    for idx in range(start, len(board_states)):
        board = board_states[idx]
        if board.is_game_over():
            continue
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        cp_white = score_cp_white(info["score"])
        cp_gf = cp_white if gf_color == chess.WHITE else -cp_white
        if best_cp is None or cp_gf > best_cp:
            best_cp = cp_gf
            best_idx = idx
    return best_cp, best_idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgn", default=DEFAULT_PGN,
                    help=f"PGN file to analyze (default: {DEFAULT_PGN})")
    ap.add_argument("--stockfish", default=DEFAULT_STOCKFISH,
                    help=f"Stockfish executable (default: {DEFAULT_STOCKFISH})")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Winning eval threshold in pawns, GuoFish POV (default: 0.5)")
    ap.add_argument("--depth", type=int, default=18,
                    help="Stockfish search depth per position (default: 18)")
    ap.add_argument("--window", type=int, default=20,
                    help="Plies from game end to scan for peak GuoFish eval (default: 20)")
    ap.add_argument("--threads", type=int, default=1,
                    help="Stockfish Threads option (default: 1)")
    ap.add_argument("--hash", type=int, default=256,
                    help="Stockfish Hash (MB) option (default: 256)")
    ap.add_argument("--out", default=None,
                    help="Optional path to also write the report as markdown")
    args = ap.parse_args()

    if not os.path.exists(args.stockfish):
        sys.exit(f"ERROR: Stockfish not found at {args.stockfish}")
    if not os.path.exists(args.pgn):
        sys.exit(f"ERROR: PGN not found at {args.pgn}")

    threshold_cp = round(args.threshold * 100)

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    engine.configure({"Threads": args.threads, "Hash": args.hash})

    total_games = 0
    total_draws = 0
    skipped_no_gf = 0
    draw_type_counts: dict[str, int] = {}
    # Records for relevant (repetition / fifty-move) draws.
    flagged = []        # GuoFish was winning at peak
    not_winning = []    # relevant draw but never reached the winning threshold

    try:
        with open(args.pgn, "r", encoding="utf-8", errors="replace") as fh:
            while True:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                total_games += 1

                if game.headers.get("Result", "") != "1/2-1/2":
                    continue
                total_draws += 1

                gf_color = guofish_color(game)
                if gf_color is None:
                    skipped_no_gf += 1
                    continue

                # Replay, recording every position reached.
                board = game.board()
                states = [board.copy()]
                for move in game.mainline_moves():
                    board.push(move)
                    states.append(board.copy())

                final_board = states[-1]
                dtype = classify_draw(final_board)
                draw_type_counts[dtype] = draw_type_counts.get(dtype, 0) + 1

                if dtype not in RELEVANT_DRAWS:
                    continue

                # Scan positions up to (but not including) the final drawn one.
                best_cp, best_idx = peak_guofish_eval(
                    engine, states[:-1], gf_color, args.window, args.depth)

                rnd = game.headers.get("Round", "?")
                gf_side = "White" if gf_color == chess.WHITE else "Black"
                peak_ply = best_idx if best_idx is not None else -1
                rec = {
                    "round": rnd,
                    "side": gf_side,
                    "dtype": dtype,
                    "peak_cp": best_cp,
                    "peak_ply": peak_ply,
                    "plycount": game.headers.get("PlyCount", str(len(states) - 1)),
                }
                if best_cp is not None and best_cp >= threshold_cp:
                    flagged.append(rec)
                else:
                    not_winning.append(rec)
    finally:
        engine.quit()

    # ---- Report ----
    lines: list[str] = []
    lines.append(f"# Draws-while-winning analysis: {os.path.basename(args.pgn)}\n")
    lines.append(f"- Total games: **{total_games}**")
    lines.append(f"- Drawn games: **{total_draws}**")
    if skipped_no_gf:
        lines.append(f"- Draws skipped (could not identify GuoFish side): {skipped_no_gf}")
    lines.append(f"- Winning threshold: GuoFish peak eval >= **+{args.threshold:.2f}** "
                 f"pawns ({threshold_cp} cp), scanned over the last {args.window} plies "
                 f"at depth {args.depth}.\n")

    lines.append("## Draw breakdown by board reason\n")
    for dtype in sorted(draw_type_counts):
        lines.append(f"- {dtype}: {draw_type_counts[dtype]}")
    lines.append("")

    rel_total = len(flagged) + len(not_winning)
    lines.append("## Repetition / 50-move draws\n")
    lines.append(f"- Total repetition/50-move draws: **{rel_total}**")
    lines.append(f"- ...of which GuoFish was WINNING (peak >= +{args.threshold:.2f}): "
                 f"**{len(flagged)}**")
    lines.append(f"- ...of which GuoFish was NOT winning: {len(not_winning)}\n")

    def fmt_rows(records: list) -> None:
        if not records:
            lines.append("_(none)_\n")
            return
        lines.append("| Round | GuoFish side | Draw type | Peak GuoFish eval | "
                     "Peak ply | PlyCount |")
        lines.append("|-------|--------------|-----------|-------------------|"
                     "----------|----------|")
        for r in sorted(records, key=lambda x: (x["peak_cp"] is None, -(x["peak_cp"] or 0))):
            if r["peak_cp"] is None:
                ev = "n/a"
            elif abs(r["peak_cp"]) >= MATE_SCORE_CP - 1000:
                ev = "#(mate)" if r["peak_cp"] > 0 else "-#(mate)"
            else:
                ev = f"{r['peak_cp']/100:+.2f}"
            lines.append(f"| {r['round']} | {r['side']} | {r['dtype']} | {ev} | "
                         f"{r['peak_ply']} | {r['plycount']} |")
        lines.append("")

    lines.append("### GuoFish drew from a winning position\n")
    fmt_rows(flagged)
    lines.append("### GuoFish drew but was not winning (for reference)\n")
    fmt_rows(not_winning)

    report = "\n".join(lines)
    print(report)
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
