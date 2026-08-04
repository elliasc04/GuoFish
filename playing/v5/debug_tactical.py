"""Phase 1 tactical-blindness instrument (READ-ONLY; no engine behavior change).

Runs a deterministic single-worker MCTS search on a position and dumps, for the
relevant subtree, each node's children with:
    - move (SAN) and whether it GIVES CHECK
    - whether the side-to-move at that node IS IN CHECK (evasion context)
    - policy prior
    - visit count (and share of parent visits)
    - Q (mover's perspective: + = good for whoever moved INTO the node)
    - terminal flags (is_terminal / terminal_value) so we can tell a starved
      checking move (0 visits, never expanded) from a visited-but-not-backing-up
      mate (the latter would re-scope us to a terminal-propagation bug).

It does NOT import/enable the opening book or tablebase, and calls search()
directly, so what you see is the pure policy/PUCT selection behavior.

Usage:
    python playing/v5/debug_tactical.py --fen "<FEN>" \
        [--line "Kd3 Rd6+ Ke4 Re7+ Kf3 Ne5+ Ke4 Nd7+ Kf3 Rd3+ Kg4 Nf6#"] \
        [--sims 3000] [--workers 1] [--top-k 8] [--pv-depth 12] \
        [checkpoint]

`--line` is the continuation FROM THE ROOT in SAN (the leading 1.Nc4+ in the
report is the move that REACHES the root, so the root line starts with Kd3).
"""
import argparse
import sys
from pathlib import Path

import chess

# Importing playv5 sets up sys.path for `core` and gives us load_model.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import playv5
from core.mctsv3 import ParallelMCTS

import torch


def _fmt_child(b: chess.Board, move: chess.Move, child, parent_visits: int) -> str:
    """One formatted row for a child edge, read-only from board + node."""
    try:
        san = b.san(move)
    except Exception:
        san = move.uci()
    gives_check = b.gives_check(move)
    share = (child.visit_count / parent_visits * 100.0) if parent_visits else 0.0
    flags = []
    if gives_check:
        flags.append("CHECK")
    if child.is_terminal:
        flags.append(f"TERM={child.terminal_value:+.1f}")
    if child.visit_count == 0:
        flags.append("UNVISITED")
    if not child.is_expanded and child.visit_count > 0:
        flags.append("leaf")
    flagstr = (" [" + ",".join(flags) + "]") if flags else ""
    return (f"{san:<7} prior={child.prior * 100:5.1f}%  "
            f"visits={child.visit_count:>7} ({share:4.1f}%)  "
            f"Q={child.q_value:+.3f}{flagstr}")


def dump_node_children(node, board: chess.Board, top_k: int,
                       force_moves: list[chess.Move] | None = None) -> None:
    """Print all (capped) children of one node, sorted by visits, ALWAYS including
    any checking move and any move in `force_moves` even if outside the top-k."""
    if not node.children:
        print("    (no children -- node was never expanded / never searched)")
        return
    parent_visits = sum(c.visit_count for c in node.children.values())
    kids = sorted(node.children.items(), key=lambda kv: kv[1].visit_count, reverse=True)

    force = set(force_moves or [])
    shown = set()
    # Top-k by visits.
    for move, child in kids[:top_k]:
        print("    " + _fmt_child(board, move, child, parent_visits))
        shown.add(move)
    # Always surface checking moves and forced (mate-line) moves below the cut.
    extra = [(m, c) for m, c in kids
             if m not in shown and (board.gives_check(m) or m in force)]
    if extra:
        print(f"    --- below top-{top_k}: checking / line moves ---")
        for move, child in extra:
            print("    " + _fmt_child(board, move, child, parent_visits))


def dump_visit_pv(root, root_board: chess.Board, depth: int) -> None:
    """Follow the most-visited child down the tree -- the line the engine actually
    believes / spent its search on."""
    print("\n========== VISIT-COUNT PV (where the search actually went) ==========")
    board = root_board.copy()
    node = root
    for ply in range(depth):
        if not node.children:
            print(f"  ply {ply}: (leaf -- no children)")
            break
        move, child = max(node.children.items(), key=lambda kv: kv[1].visit_count)
        if child.visit_count == 0:
            print(f"  ply {ply}: (best child unvisited -- stop)")
            break
        stm = "W" if board.turn else "B"
        chk = "+" if board.gives_check(move) else " "
        print(f"  ply {ply:2} [{stm}] {board.san(move):<7}{chk} "
              f"visits={child.visit_count:>7} prior={child.prior*100:5.1f}% Q={child.q_value:+.3f}"
              f"{'  MATE' if child.is_terminal and abs(child.terminal_value)==1.0 else ''}")
        board.push(move)
        node = child


def dump_line(root, root_board: chess.Board, line_sans: list[str], top_k: int) -> None:
    """Follow the provided mate line from the root; at each ply print the FULL
    sibling distribution so we can see the line move's prior/visits vs. what the
    search preferred."""
    print("\n========== MATE-LINE WALK (is the forcing line starved?) ==========")
    board = root_board.copy()
    node = root
    for ply, expected_san in enumerate(line_sans):
        stm = "White" if board.turn else "Black"
        in_check = board.is_check()
        print(f"\n--- ply {ply}: {stm} to move | side-to-move in_check={in_check} "
              f"| next line move: {expected_san} ---")
        if node is None:
            print("    (no tree node here -- line left the searched tree)")
            break
        # Parse the expected move so we can force-show it and descend.
        try:
            mv = board.parse_san(expected_san)
        except Exception as e:
            print(f"    !! could not parse '{expected_san}' on this board "
                  f"({type(e).__name__}); board FEN: {board.fen()}")
            break
        dump_node_children(node, board, top_k, force_moves=[mv])
        child = node.children.get(mv) if node.children else None
        if child is None:
            print(f"    >> line move {expected_san} is NOT a child in the tree "
                  f"(never generated/expanded at this node).")
            # We can still keep walking the board, but there's no subtree to show.
            board.push(mv)
            node = None
            continue
        if child.visit_count == 0:
            print(f"    >> line move {expected_san}: prior={child.prior*100:.1f}%, "
                  f"VISITS=0  <-- STARVED")
        board.push(mv)
        node = child


DEFAULT_LINE = "Kd3 Rd6+ Ke4 Re7+ Kf3 Ne5+ Ke4 Nd7+ Kf3 Rd3+ Kg4 Nf6#"


def main():
    ap = argparse.ArgumentParser(description="Phase 1 tactical-blindness tree dump")
    ap.add_argument("checkpoint", type=Path, nargs="?",
                    default=playv5._PROJECT_ROOT / "models" / "guofish4" / "guofish4_25.6M_policy_final.pt")
    ap.add_argument("--fen", required=True, help="Root position (engine to move).")
    ap.add_argument("--line", default=DEFAULT_LINE,
                    help="Continuation from the root in SAN (space-separated).")
    ap.add_argument("--sims", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=1,
                    help="Force 1 for determinism (default).")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--pv-depth", type=int, default=14)
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: checkpoint {args.checkpoint} not found")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = playv5.load_model(args.checkpoint, device)
    mcts = ParallelMCTS(model, device, num_workers=args.workers)
    # NOTE: deliberately leaving mcts.tablebase = None and not touching the book.

    board = chess.Board(args.fen)
    print(f"\nRoot FEN: {board.fen()}")
    print(f"Side to move: {'White' if board.turn else 'Black'} | "
          f"in_check={board.is_check()} | sims={args.sims} | workers={args.workers}")

    best = mcts.search(board, num_simulations=args.sims)
    print(f"\nEngine bestmove: {board.san(best) if best else None}  "
          f"(root_q={mcts.last_root_q:+.3f}, best_child_q={mcts.last_best_child_q:+.3f})")

    root = mcts.root
    print("\n========== ROOT CHILDREN (engine's move choice at the root) ==========")
    print(f"Root: {'White' if board.turn else 'Black'} to move, "
          f"in_check={board.is_check()}")
    dump_node_children(root, board, args.top_k)

    dump_visit_pv(root, board, args.pv_depth)

    line_sans = args.line.split()
    dump_line(root, board, line_sans, args.top_k)

    mcts.evaluator.stop()


if __name__ == "__main__":
    main()
