"""Minimal CUDA load generator for exercising the PCIe link under sustained
GPU utilisation -- paired with tools/monitor_pcie_health.ps1 to reproduce the
P7 D14 diagnosis conditions (docs/tuning/RESULTS.md) without needing a full
cutechess match running.

    python tools/gpu_load_stress.py --seconds 120
"""
import argparse
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--size", type=int, default=8192)
    args = parser.parse_args()

    device = torch.device("cuda")
    a = torch.randn(args.size, args.size, device=device, dtype=torch.float16)
    b = torch.randn(args.size, args.size, device=device, dtype=torch.float16)

    print(f"[stress] running matmul loop on {torch.cuda.get_device_name(0)} for {args.seconds}s", flush=True)
    start = time.monotonic()
    iters = 0
    while time.monotonic() - start < args.seconds:
        c = a @ b
        torch.cuda.synchronize()
        iters += 1
    print(f"[stress] done: {iters} iterations", flush=True)


if __name__ == "__main__":
    main()
