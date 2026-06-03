"""Parse cutechess PGN output from the Phase 3 C_PUCT sweep into a markdown report.

Pipeline:
  1. Walk every cpuct_*.pgn AND phase2_baseline_vs_sf.pgn in --pgn-dir and
     tally W/L/D per (engine, opponent) pair.
  2. Build a combined PGN-list file and invoke ordo with Stockfish anchored at
     the calibration ELO (--sf-anchor, default 2500). Every other engine's
     rating is then on an absolute scale, with proper CIs from -s simulations.
  3. Report per candidate:
        - W/L/D & score vs Phase 2 reference
        - W/L/D & score vs Stockfish
        - Absolute ELO (ordo, SF-anchored) +/- 95% CI half-width
        - ELO diff vs Phase 2 ref measured two ways:
            (a) Head-to-head: directly from the candidate-vs-ref score
            (b) SF-inferred:  (cand_vs_sf_elo) - (ref_vs_sf_elo)
          plus a discrepancy column; flag >50 ELO disagreement.
  4. Plot absolute ELO vs C_PUCT (matplotlib PNG) for visual curve inspection.
  5. Recommend top candidates within the best candidate's CI for a follow-up
     200-game formal benchmark.

Phase 2 ref's absolute ELO is also reported as a drift check against the
prior v1 sweep's measurement.
"""

import argparse
import csv
import glob
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict


CANDIDATE_NAME_RE = re.compile(r"Phase3_cpuct_([0-9.]+)")
BASELINE_NAME = "Phase2_ref"
STOCKFISH_NAME = "Stockfish"
DISCREPANCY_WARN_ELO = 50.0
FOLLOWUP_GAME_COUNT = 200

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_PGN_DIR = os.path.join(REPO_ROOT, "benchmarking", "engine", "games", "cpuct_phase3_tuning")
DEFAULT_OUT = os.path.join(REPO_ROOT, "benchmarking", "engine", "logs", "cpuct_phase3_tuning_results.md")
DEFAULT_PLOT = os.path.join(REPO_ROOT, "benchmarking", "engine", "logs", "cpuct_phase3_elo_curve.png")
DEFAULT_ORDO = os.path.join(REPO_ROOT, "ordo-win64.exe")


def tally_pgn(path: str) -> dict[tuple[str, str], tuple[int, int, int]]:
    """Return {(engine_name, opponent_name): (wins, losses, draws)} for one PGN."""
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    white = black = result = None

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("[White "):
                m = re.match(r'\[White\s+"([^"]*)"\]', line)
                white = m.group(1) if m else None
            elif line.startswith("[Black "):
                m = re.match(r'\[Black\s+"([^"]*)"\]', line)
                black = m.group(1) if m else None
            elif line.startswith("[Result "):
                m = re.match(r'\[Result\s+"([^"]*)"\]', line)
                result = m.group(1) if m else None
            elif line == "" and white and black and result:
                if result == "1/2-1/2":
                    counts[(white, black)][2] += 1
                    counts[(black, white)][2] += 1
                elif result == "1-0":
                    counts[(white, black)][0] += 1
                    counts[(black, white)][1] += 1
                elif result == "0-1":
                    counts[(white, black)][1] += 1
                    counts[(black, white)][0] += 1
                white = black = result = None

    return {k: (v[0], v[1], v[2]) for k, v in counts.items()}


def run_ordo(ordo_path: str, pgn_files: list[str], csv_out: str,
             sf_anchor: float, simulations: int = 1000) -> None:
    """Invoke ordo with Stockfish anchored at the calibration ELO."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        pgn_list_path = fh.name
        for p in pgn_files:
            fh.write(p + "\n")

    try:
        cmd = [
            ordo_path,
            "-q",
            "-a", str(sf_anchor),
            "-A", STOCKFISH_NAME,
            "-D",
            "-W",
            "-s", str(simulations),
            "-P", pgn_list_path,
            "-c", csv_out,
        ]
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(pgn_list_path)


def parse_ordo_csv(path: str) -> dict[str, tuple[float, float]]:
    """Return {player_name: (rating, error)} from an ordo CSV file."""
    out: dict[str, tuple[float, float]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        name_key = next((k for k in (reader.fieldnames or []) if k.strip().lower() == "player"), None)
        rating_key = next((k for k in (reader.fieldnames or []) if k.strip().lower() == "rating"), None)
        error_key = next((k for k in (reader.fieldnames or []) if k.strip().lower() == "error"), None)
        if not (name_key and rating_key):
            raise RuntimeError(f"Unexpected ordo CSV header: {reader.fieldnames}")
        for row in reader:
            name = (row[name_key] or "").strip()
            try:
                rating = float(row[rating_key])
            except (TypeError, ValueError):
                continue
            error = math.nan
            if error_key:
                raw_err = (row[error_key] or "").strip()
                try:
                    error = float(raw_err)
                except ValueError:
                    pass
            out[name] = (rating, error)
    return out


def fmt_signed(x: float, decimals: int = 1) -> str:
    if math.isnan(x):
        return "n/a"
    if math.isinf(x):
        return "+inf" if x > 0 else "-inf"
    return f"{x:+.{decimals}f}"


def fmt_unsigned(x: float, decimals: int = 1) -> str:
    if math.isnan(x):
        return "n/a"
    if math.isinf(x):
        return "+inf" if x > 0 else "-inf"
    return f"{x:.{decimals}f}"


def score_str(wins: int, losses: int, draws: int) -> str:
    n = wins + losses + draws
    if n == 0:
        return "n/a"
    return f"{(wins + 0.5 * draws) / n:.3f}"


def score_to_elo(score: float) -> float:
    if score <= 0.0:
        return float("-inf")
    if score >= 1.0:
        return float("inf")
    return -400.0 * math.log10(1.0 / score - 1.0)


def score_elo_se(wins: int, losses: int, draws: int) -> float:
    n = wins + losses + draws
    if n == 0:
        return float("nan")
    score = (wins + 0.5 * draws) / n
    if score <= 0.0 or score >= 1.0:
        return float("inf")
    p_w, p_d, p_l = wins / n, draws / n, losses / n
    var_per_game = (p_w * (1.0 - score) ** 2
                    + p_d * (0.5 - score) ** 2
                    + p_l * (0.0 - score) ** 2)
    se_score = math.sqrt(var_per_game / n) if var_per_game > 0 else 0.0
    d_elo_d_score = 400.0 / (math.log(10) * score * (1.0 - score))
    return se_score * d_elo_d_score


def cpuct_from_engine_name(name: str) -> str | None:
    m = CANDIDATE_NAME_RE.match(name)
    return m.group(1) if m else None


def plot_elo_curve(plot_path: str, diag_rows: list[tuple],
                   baseline_elo: float, sf_anchor: float) -> None:
    """Save a matplotlib PNG of absolute ELO vs C_PUCT with error bars."""
    finite = [(float(c), e, err) for (c, e, err, _, _) in diag_rows
              if not math.isnan(e) and not math.isinf(e)]
    if not finite:
        print("[plot] no finite ELO entries; skipping plot.", file=sys.stderr)
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; falling back to ASCII plot.", file=sys.stderr)
        _ascii_elo_curve(finite, baseline_elo, sf_anchor)
        return

    finite.sort(key=lambda r: r[0])
    xs   = [r[0] for r in finite]
    ys   = [r[1] for r in finite]
    yerr = [r[2] if not math.isnan(r[2]) else 0.0 for r in finite]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(xs, ys, yerr=yerr, marker="o", linestyle="-", color="tab:blue",
                capsize=4, label="Phase 3 candidate (abs ELO, ordo)")
    if not math.isnan(baseline_elo):
        ax.axhline(baseline_elo, color="tab:orange", linestyle="--",
                   label=f"Phase 2 ref ({baseline_elo:.0f} ELO)")
    ax.axhline(sf_anchor, color="gray", linestyle=":",
               label=f"Stockfish anchor ({sf_anchor:.0f} ELO)")
    ax.set_xlabel("C_PUCT")
    ax.set_ylabel("Absolute ELO (Stockfish-anchored)")
    ax.set_title("Phase 3 C_PUCT sweep")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Wrote ELO curve plot: {plot_path}")


def _ascii_elo_curve(finite: list[tuple[float, float, float]],
                     baseline_elo: float, sf_anchor: float) -> None:
    """Fallback if matplotlib is missing. Single-line per candidate."""
    elos = [y for (_, y, _) in finite]
    lo, hi = min(elos), max(elos)
    span = max(1.0, hi - lo)
    print("\nELO vs C_PUCT (ASCII):")
    for (x, y, e) in finite:
        col = int(60 * (y - lo) / span)
        bar = " " * col + "*"
        err = f" +/- {e:.0f}" if not math.isnan(e) else ""
        print(f"  C_PUCT={x:>5.2f}  ELO={y:7.1f}{err}   {bar}")
    if not math.isnan(baseline_elo):
        print(f"  (Phase 2 ref baseline: {baseline_elo:.0f} ELO)")
    print(f"  (Stockfish anchor:    {sf_anchor:.0f} ELO)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse Phase 3 C_PUCT sweep PGNs and emit a markdown report."
    )
    ap.add_argument("--pgn-dir", default=DEFAULT_PGN_DIR,
                    help=f"Directory of sweep PGN files (default: {DEFAULT_PGN_DIR})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"Markdown report output path (default: {DEFAULT_OUT})")
    ap.add_argument("--plot", default=DEFAULT_PLOT,
                    help=f"ELO-vs-C_PUCT plot output (PNG) (default: {DEFAULT_PLOT})")
    ap.add_argument("--ordo", default=DEFAULT_ORDO,
                    help=f"Path to ordo executable (default: {DEFAULT_ORDO})")
    ap.add_argument("--simulations", type=int, default=1000,
                    help="Ordo bootstrap simulations for error margins (default: 1000)")
    ap.add_argument("--sf-anchor", type=float, default=2500.0,
                    help="Stockfish UCI_Elo anchor (default: 2500, matches the sweep .bat)")
    args = ap.parse_args()

    if not os.path.exists(args.ordo):
        print(f"ERROR: ordo executable not found at {args.ordo}", file=sys.stderr)
        sys.exit(1)

    pgn_files = sorted(
        set(glob.glob(os.path.join(args.pgn_dir, "cpuct_*.pgn"))) |
        set(glob.glob(os.path.join(args.pgn_dir, "phase2_baseline_vs_sf.pgn")))
    )
    if not pgn_files:
        raise SystemExit(f"No PGN files matched in {args.pgn_dir}")

    # === Per-pair W/L/D tally ===
    pair_counts: dict[tuple[str, str], tuple[int, int, int]] = defaultdict(lambda: (0, 0, 0))
    for pgn in pgn_files:
        for key, wld in tally_pgn(pgn).items():
            prev = pair_counts[key]
            pair_counts[key] = (prev[0] + wld[0], prev[1] + wld[1], prev[2] + wld[2])

    candidates = sorted(
        {name for (name, _opp) in pair_counts.keys() if CANDIDATE_NAME_RE.match(name)},
        key=lambda n: float(cpuct_from_engine_name(n) or "0"),
    )
    if not candidates:
        raise SystemExit("No Phase3_cpuct_* engines found in any PGN.")

    # === Run ordo for absolute ratings (SF-anchored) ===
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
        ordo_csv = fh.name
    try:
        run_ordo(args.ordo, pgn_files, ordo_csv,
                 sf_anchor=args.sf_anchor, simulations=args.simulations)
        ratings = parse_ordo_csv(ordo_csv)
    finally:
        if os.path.exists(ordo_csv):
            os.unlink(ordo_csv)

    baseline_rating, baseline_err = ratings.get(BASELINE_NAME, (math.nan, math.nan))
    sf_rating, _sf_err = ratings.get(STOCKFISH_NAME, (math.nan, math.nan))

    # === Baseline score vs SF for the (b) inference baseline ===
    b_vs_sf = pair_counts.get((BASELINE_NAME, STOCKFISH_NAME), (0, 0, 0))
    b_w_sf, b_l_sf, b_d_sf = b_vs_sf
    b_n_sf = b_w_sf + b_l_sf + b_d_sf
    if b_n_sf > 0:
        b_score_vs_sf = (b_w_sf + 0.5 * b_d_sf) / b_n_sf
        b_elo_vs_sf = score_to_elo(b_score_vs_sf)
        b_elo_vs_sf_se = score_elo_se(b_w_sf, b_l_sf, b_d_sf)
    else:
        b_elo_vs_sf = math.nan
        b_elo_vs_sf_se = math.nan

    # === Render markdown ===
    lines: list[str] = []
    lines.append("# Phase 3 C_PUCT Tuning Results\n")
    lines.append("Phase 3 model (`guofish4`) at varying C_PUCT plays two opponents per sweep step:")
    lines.append(f"- **Phase 2 reference** (`Phase2_ref`, C_PUCT=1.5, {800} sims/move) -- isolates the C_PUCT effect on Phase 3.")
    lines.append(f"- **Stockfish** (tc=10+0.1) -- absolute ELO calibration, anchored at {args.sf_anchor:.0f}.")
    lines.append(f"- **Phase 2 reference also plays Stockfish** -- pins the Phase 2 baseline on today's hardware.\n")
    lines.append(f"Ordo ratings use {args.simulations} bootstrap simulations for 95% CIs.\n")

    # --- Calibration drift line ---
    lines.append("## Calibration drift check\n")
    if not math.isnan(baseline_rating):
        lines.append(f"- Phase 2 ref absolute ELO (ordo, SF anchored at {args.sf_anchor:.0f}): "
                     f"**{fmt_unsigned(baseline_rating)} +/- {fmt_unsigned(baseline_err)}**.")
    if not math.isnan(b_elo_vs_sf):
        lines.append(f"- Phase 2 ref score vs SF (raw): {b_w_sf}W/{b_l_sf}L/{b_d_sf}D "
                     f"-> ELO_vs_SF = {fmt_signed(b_elo_vs_sf)} "
                     f"-> absolute ELO ~ {fmt_unsigned(args.sf_anchor + b_elo_vs_sf)}.")
    lines.append("- Compare against the v1 sweep's Phase 2 measurement; if these diverge by >50 ELO,"
                 " the hardware/SF version has drifted and head-to-head deltas are the more"
                 " reliable signal.\n")

    # --- Main per-candidate table ---
    lines.append("## Per-candidate results\n")
    lines.append("| C_PUCT | vs Phase 2 (W/L/D, score) | vs SF (W/L/D, score) | Absolute ELO | 95% CI | "
                 "ELO diff vs Phase 2 (a: head-to-head) | ELO diff vs Phase 2 (b: via SF) | |a-b| |")
    lines.append("|--------|---------------------------|----------------------|--------------|--------|"
                 "--------------------------------------|---------------------------------|--------|")

    diag_rows = []  # (cpuct_str, abs_elo, abs_err, delta_a, delta_b)
    for cand in candidates:
        cpuct = cpuct_from_engine_name(cand) or "?"
        w1, l1, d1 = pair_counts.get((cand, BASELINE_NAME), (0, 0, 0))
        ws, ls, ds = pair_counts.get((cand, STOCKFISH_NAME), (0, 0, 0))
        elo, err = ratings.get(cand, (math.nan, math.nan))

        ci_str = (f"[{fmt_unsigned(elo - err)}, {fmt_unsigned(elo + err)}]"
                  if not math.isnan(elo) and not math.isnan(err) else "n/a")

        n1 = w1 + l1 + d1
        if n1 > 0:
            score_a = (w1 + 0.5 * d1) / n1
            delta_a = score_to_elo(score_a)
            delta_a_se = score_elo_se(w1, l1, d1)
            delta_a_str = f"{fmt_signed(delta_a)} +/- {fmt_unsigned(delta_a_se)}"
        else:
            delta_a = math.nan
            delta_a_se = math.nan
            delta_a_str = "n/a"

        ns = ws + ls + ds
        if ns > 0 and not math.isnan(b_elo_vs_sf):
            score_cand_vs_sf = (ws + 0.5 * ds) / ns
            cand_elo_vs_sf = score_to_elo(score_cand_vs_sf)
            cand_elo_vs_sf_se = score_elo_se(ws, ls, ds)
            delta_b = cand_elo_vs_sf - b_elo_vs_sf
            delta_b_se = math.sqrt(cand_elo_vs_sf_se ** 2 + b_elo_vs_sf_se ** 2)
            delta_b_str = f"{fmt_signed(delta_b)} +/- {fmt_unsigned(delta_b_se)}"
        else:
            delta_b = math.nan
            delta_b_str = "n/a"

        if not math.isnan(delta_a) and not math.isnan(delta_b) and \
           not math.isinf(delta_a) and not math.isinf(delta_b):
            disc = abs(delta_a - delta_b)
            disc_str = fmt_unsigned(disc)
            if disc > DISCREPANCY_WARN_ELO:
                disc_str += " !"
        else:
            disc_str = "n/a"

        vs_b = f"{w1}/{l1}/{d1}, {score_str(w1, l1, d1)}"
        vs_sf = f"{ws}/{ls}/{ds}, {score_str(ws, ls, ds)}"
        lines.append(f"| {cpuct:>6} | {vs_b:<25} | {vs_sf:<20} | {fmt_unsigned(elo):>12} | "
                     f"{ci_str} | {delta_a_str} | {delta_b_str} | {disc_str} |")
        diag_rows.append((cpuct, elo, err, delta_a, delta_b))

    # --- Recommendation: candidates within best CI for follow-up ---
    lines.append("\n## Recommendation for formal benchmark\n")
    finite_rows = [r for r in diag_rows if not math.isnan(r[1]) and not math.isinf(r[1])]
    if finite_rows:
        best = max(finite_rows, key=lambda r: r[1])
        best_cpuct, best_elo, best_err, _, _ = best
        ci_width = best_err if not math.isnan(best_err) else 0.0
        threshold = best_elo - ci_width
        within = [r for r in finite_rows if r[1] >= threshold]
        within.sort(key=lambda r: r[1], reverse=True)

        lines.append(f"- Best candidate: **C_PUCT={best_cpuct}** at "
                     f"{fmt_unsigned(best_elo)} ELO (+/- {fmt_unsigned(ci_width)}).")
        lines.append(f"- Candidates within one CI-width of the best "
                     f"(ELO >= {fmt_unsigned(threshold)}):")
        for c, e, er, _, _ in within:
            lines.append(f"    - C_PUCT={c}: {fmt_unsigned(e)} ELO +/- {fmt_unsigned(er)}")

        top_n = min(3, len(within))
        top_cpucts = [str(r[0]) for r in within[:top_n]]
        lines.append("")
        lines.append(f"**Recommend: run {top_n} candidate(s) ({', '.join(top_cpucts)}) "
                     f"at {FOLLOWUP_GAME_COUNT} games vs Stockfish each to identify the "
                     "formal best.** These are statistically indistinguishable from the "
                     "current best at this sample size; the follow-up tightens their CIs "
                     "by ~2x and separates them.")

        if len(finite_rows) >= 3:
            edge_cpucts = [r[0] for r in finite_rows]
            if best_cpuct in (edge_cpucts[0], edge_cpucts[-1]):
                lines.append("")
                lines.append(f"NOTE: best candidate ({best_cpuct}) is at a sweep endpoint -- "
                             "the peak may lie beyond the swept range. Consider extending "
                             "the sweep before locking in the optimum.")
    else:
        lines.append("- No finite ELO rows available; check ordo output and PGN integrity.")

    # --- Discrepancy diagnostics ---
    flagged = [(c, da, db) for (c, _, _, da, db) in diag_rows
               if not math.isnan(da) and not math.isnan(db)
               and not math.isinf(da) and not math.isinf(db)
               and abs(da - db) > DISCREPANCY_WARN_ELO]
    if flagged:
        lines.append("")
        lines.append(f"### Head-to-head vs SF-inferred discrepancies > {DISCREPANCY_WARN_ELO:.0f} ELO\n")
        for cpuct, da, db in flagged:
            lines.append(f"- C_PUCT={cpuct}: head-to-head says {fmt_signed(da)}, "
                         f"SF-inferred says {fmt_signed(db)} (diff {fmt_unsigned(abs(da - db))}). "
                         "Usually means the SF leg has too few games, or matched-sims vs clock-time "
                         "are measuring different strength aspects.")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {args.out}")
    print("\n".join(lines))

    # --- Plot ---
    plot_elo_curve(args.plot, diag_rows, baseline_rating, args.sf_anchor)


if __name__ == "__main__":
    main()
