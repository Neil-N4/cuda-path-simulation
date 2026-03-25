#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CPU vs GPU benchmark results")
    parser.add_argument("--csv", default=Path("results/benchmark_results.csv"), type=Path)
    parser.add_argument("--out", default=Path("results/runtime_comparison.png"), type=Path)
    args = parser.parse_args()

    runs, cpu, gpu = [], [], []
    with args.csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            runs.append(int(row["run"]))
            cpu.append(float(row["cpu_ms"]))
            gpu.append(float(row["gpu_ms"]))

    plt.figure(figsize=(8, 5))
    plt.plot(runs, cpu, marker="o", label="CPU baseline")
    plt.plot(runs, gpu, marker="o", label="CUDA")
    plt.xlabel("Run")
    plt.ylabel("Runtime (ms)")
    plt.title("Monte Carlo Runtime: CPU vs CUDA")
    plt.grid(alpha=0.3)
    plt.legend()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
