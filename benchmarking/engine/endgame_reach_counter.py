"""
Count how many games in a PGN file reached positions with <= X pieces remaining,
for X in [5, 6, 7]. Piece count is tracked by counting captures from the start.
"""

import argparse
import chess.pgn
import io


def count_pieces_from_board(board: chess.Board) -> int:
    return len(board.piece_map())


def games_reaching_endgame(pgn_path: str, thresholds: list[int]) -> dict[int, int]:
    counts = {x: 0 for x in thresholds}

    with open(pgn_path, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            board = game.board()
            reached = {x: False for x in thresholds}

            for move in game.mainline_moves():
                board.push(move)
                piece_count = count_pieces_from_board(board)
                for x in thresholds:
                    if not reached[x] and piece_count <= x:
                        reached[x] = True

            for x in thresholds:
                if reached[x]:
                    counts[x] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Count games reaching endgame thresholds in a PGN file."
    )
    parser.add_argument("pgn", help="Path to the PGN file")
    args = parser.parse_args()

    thresholds = [5, 6, 7]
    counts = games_reaching_endgame(args.pgn, thresholds)

    total_games = None
    with open(args.pgn, "r", encoding="utf-8") as f:
        total_games = sum(1 for line in f if line.startswith("[Event "))

    print(f"PGN: {args.pgn}")
    print(f"Total games: {total_games}")
    for x in thresholds:
        pct = 100 * counts[x] / total_games if total_games else 0
        print(f"  Games reaching <= {x} pieces: {counts[x]} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
