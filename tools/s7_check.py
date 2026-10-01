#!/usr/bin/env python
"""S7 — a v6 export runs in the engine (design doc §12 S7, §11.1). Needs the GPU.

    python tools/s7_check.py <export.pt> [--sims 1600] [--limit N]

Run it in a gap in any GPU queue: every part below takes the device.

  1. UCI. `playing/uci_wrapper_v6.py --model <export> --no-book --no-syzygy
     --max-batch 128`. That covers:
       - uci/uciok, then isready/readyok (load, Inductor compile, capture of the
         ladder to 128);
       - `go nodes 800` on the 20 Gate 1 positions, where every bestmove must be legal.
     The [init] capture and architecture lines are recorded.
  2. Forward, reported only. The export as the engine holds it (bf16 Linear storage,
     policy narrowed to bf16), uncaptured under bf16 autocast, against the
     training-side forward (`load_for_inference`, fp32 weights, same autocast).
     It runs on the C++ tokenizer's rows for the c10 corpus. Expected: 0
     differing words.
  3. Contract-A numerics (§11.1), the C12b precedent. The same export, eager-captured
     (compile=False) against Inductor-compiled and captured (compile=True):
       - a W=1 K=1 search at --sims with Gate 2b's recorded search config and
         C12b's cache size, on the 500 positions of golden/c10_corpus.json;
       - pass: move agreement >= 98.75%. Disagreements are printed with both
         top-two margins.

Writes runs/s7/<export stem>.json. Exits 1 if criterion 1 or 3 fails.
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import chess  # noqa: E402
import guofish_core  # noqa: E402
import torch  # noqa: E402

from core.guofish_net import load_for_inference  # noqa: E402
from playing.v6 import evaluator as ev  # noqa: E402

CORPUS = REPO / "golden" / "c10_corpus.json"
GATE1 = REPO / "golden" / "gate1_manifest.json"
GATE2B_MANIFEST = REPO / "golden" / "c10_gate2b_manifest.json"
CACHE_SLOTS = 400_000            # tools/gen_c12b_baseline.py
MIN_AGREEMENT = 0.9875           # design doc §11.1
UCI_NODES = 800


def fens(path: Path) -> list[str]:
    return [p["fen"] for p in json.loads(path.read_text(encoding="utf-8"))["positions"]]


# ---------------------------------------------------------------- 1. UCI

def uci_smoke(export: Path, log_path: Path) -> dict:
    cmd = [sys.executable, "-u", str(REPO / "playing" / "uci_wrapper_v6.py"), "--model", str(export),
           "--no-book", "--no-syzygy", "--max-batch", "128"]
    with open(log_path, "w", encoding="utf-8") as err:
        p = subprocess.Popen(cmd, cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=err, text=True, bufsize=1)
        lines: queue.Queue = queue.Queue()
        threading.Thread(target=lambda: [lines.put(ln.rstrip("\n")) for ln in p.stdout],
                         daemon=True).start()

        def send(s: str) -> None:
            p.stdin.write(s + "\n")
            p.stdin.flush()

        def wait_for(prefix: str, timeout: float) -> tuple[str, list[str]]:
            seen, deadline = [], time.time() + timeout
            while time.time() < deadline:
                try:
                    ln = lines.get(timeout=max(0.1, deadline - time.time()))
                except queue.Empty:
                    break
                seen.append(ln)
                if ln.startswith(prefix):
                    return ln, seen
            raise TimeoutError(f"no '{prefix}' within {timeout:.0f}s (exit {p.poll()}); see {log_path}")

        out = {"cmd": cmd[1:], "positions": []}
        try:
            t0 = time.time()
            send("uci")
            wait_for("uciok", 60)
            send("isready")
            wait_for("readyok", 900)
            out["ready_seconds"] = round(time.time() - t0, 1)
            for fen in fens(GATE1):
                send(f"position fen {fen}")
                send(f"go nodes {UCI_NODES}")
                line, seen = wait_for("bestmove", 300)
                move = line.split()[1]
                board = chess.Board(fen)
                legal = move in {m.uci() for m in board.legal_moves}
                info = [s for s in seen if s.startswith("info") and " nodes " in s]
                out["positions"].append({"fen": fen, "bestmove": move, "legal": legal,
                                         "last_info": info[-1] if info else None})
            send("quit")
            p.wait(timeout=60)
        finally:
            if p.poll() is None:
                p.kill()
    text = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out["init"] = [ln for ln in text if ln.startswith(("[init] capture", "[init] architecture",
                                                        "[load_model]", "[eval] value_scale"))]
    out["exit_code"] = p.returncode
    out["passed"] = (len(out["positions"]) == 20 and all(r["legal"] for r in out["positions"])
                     and any("capture=" in ln and "sizes" in ln for ln in out["init"]))
    return out


# ------------------------------------------------------------ 2. forward

@torch.no_grad()
def forward_check(export: Path, device: torch.device, positions: list[str]) -> dict:
    reference, contract = load_for_inference(export)
    reference = reference.to(device)
    engine, _ = ev.load_default_model(export, device)
    # the core's own rows for this contract; the net reads the first seq_length slots
    rows = torch.stack([torch.from_numpy(guofish_core.eval_row(f, contract)["tokens"])
                        for f in positions]).long()
    amp = lambda: torch.autocast("cuda", dtype=ev.AUTOCAST_DTYPE)  # noqa: E731
    out = {"rows": len(rows), "policy_words_differing": 0, "value_max_abs_diff": 0.0}
    for s in range(0, len(rows), 128):
        t = rows[s:s + 128].to(device)
        with amp():
            p_ref, v_ref = reference(t[:, :reference.seq_length])
            p_eng, v_eng = engine(t)
        out["policy_words_differing"] += int((p_eng.float() != p_ref).sum())
        out["value_max_abs_diff"] = max(out["value_max_abs_diff"], float((v_eng - v_ref).abs().max()))
    return out


# ------------------------------------------------------------ 3. numerics

def search_arm(export: Path, device: torch.device, compiled: bool, positions: list[str],
               sims: int) -> tuple[list[dict], dict]:
    recorded = json.loads(GATE2B_MANIFEST.read_text(encoding="utf-8"))["search_config"]
    config = guofish_core.SearchConfig()
    for k in ("c_init", "c_base", "fpu_root", "fpu_tree", "virtual_loss", "max_tree_depth"):
        setattr(config, k, recorded[k])
    config.cache_slots = CACHE_SLOTS
    model, _ = ev.load_default_model(export, device)
    evaluator = ev.TorchEvaluator(model, device, 1, compile=compiled)
    search = guofish_core.ReplaySearchDouble(config)
    search.set_evaluator(evaluator.core)
    parallel = guofish_core.ParallelConfig(workers=1, in_flight=1, max_batch=1)
    records, started = [], time.perf_counter()
    try:
        for i, fen in enumerate(positions):
            search.set_position(fen)
            stats = search.search_parallel(sims, parallel)
            arrays = search.dump_tree_arrays(0)
            visits = {guofish_core.move_to_uci(int(m)): int(c)
                      for d, m, c in zip(arrays["depth"], arrays["move"], arrays["visits"]) if d == 1}
            counts = sorted(visits.values(), reverse=True)
            total = sum(counts)
            margin = 1.0 if len(counts) == 1 else ((counts[0] - counts[1]) / total if total else 0.0)
            records.append({"fen": fen, "best_move": stats["best_move"], "margin": margin,
                            "root_visits": int(stats["root_visits"]), "visits": visits})
            if (i + 1) % 50 == 0:
                rate = (i + 1) / (time.perf_counter() - started)
                print(f"  {'inductor' if compiled else 'eager'} {i + 1}/{len(positions)} "
                      f"{rate:.2f} pos/s", flush=True)
        evaluator.assert_no_recompilation("over the S7 sweep")
    finally:
        search.set_evaluator(None)
        meta = {"capture": evaluator.graph_report.describe() if evaluator.graph_report else None,
                "seconds": round(time.perf_counter() - started, 1)}
        evaluator.close()
    return records, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", type=Path)
    ap.add_argument("--sims", type=int, default=1600)
    ap.add_argument("--limit", type=int, default=None, help="first N corpus positions (a pilot)")
    args = ap.parse_args()
    if not ev.is_v6_export(args.export):
        raise SystemExit(f"{args.export} is not a training/v6 export")
    device = torch.device("cuda")
    out_dir = REPO / "runs" / "s7"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"export": str(args.export), "gpu": torch.cuda.get_device_name(0),
              "torch": str(torch.__version__), "build": guofish_core.build_info()}

    print("1. UCI smoke", flush=True)
    report["uci"] = uci_smoke(args.export, out_dir / f"{args.export.stem}_uci.log")
    print(f"   ready in {report['uci'].get('ready_seconds')} s; legal "
          f"{sum(r['legal'] for r in report['uci']['positions'])}/20; passed {report['uci']['passed']}")
    for ln in report["uci"]["init"]:
        print("   " + ln[:200])

    positions = fens(CORPUS)[:args.limit]
    print("2. forward vs training-side", flush=True)
    report["forward"] = forward_check(args.export, device, positions)
    print(f"   {report['forward']}")

    print(f"3. eager vs Inductor, {len(positions)} positions at {args.sims} sims, W=1 K=1", flush=True)
    eager, eager_meta = search_arm(args.export, device, False, positions, args.sims)
    inductor, ind_meta = search_arm(args.export, device, True, positions, args.sims)
    dis = [{"fen": a["fen"], "eager": a["best_move"], "inductor": b["best_move"],
            "eager_margin": a["margin"], "inductor_margin": b["margin"]}
           for a, b in zip(eager, inductor) if a["best_move"] != b["best_move"]]
    rate = 1 - len(dis) / len(positions)
    report["numerics"] = {"sims": args.sims, "positions": len(positions), "agreement": rate,
                          "min_agreement": MIN_AGREEMENT, "passed": rate >= MIN_AGREEMENT,
                          "disagreements": dis, "eager": eager_meta, "inductor": ind_meta,
                          "near_ties_under_2pct": sum(1 for r in eager if r["margin"] < 0.02)}
    print(f"   agreement {len(positions) - len(dis)}/{len(positions)} = {rate:.4%} "
          f"(criterion >= {MIN_AGREEMENT:.2%}); eager {eager_meta}; inductor {ind_meta}")
    for d in dis:
        print(f"   {d['fen']}: eager {d['eager']} ({d['eager_margin']:.2%}) vs "
              f"inductor {d['inductor']} ({d['inductor_margin']:.2%})")

    report["passed"] = report["uci"]["passed"] and report["numerics"]["passed"]
    path = out_dir / f"{args.export.stem}.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"S7 {'PASS' if report['passed'] else 'FAIL'}; wrote {path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
