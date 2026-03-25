#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
from pathlib import Path


def parse_kv(stdout: str) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for line in stdout.strip().splitlines():
        if "=" not in line:
            continue
        k, v = line.strip().split("=", 1)
        if k in {"engine"}:
            out[k] = v
        else:
            out[k] = float(v)
    return out


def run_binary(path: Path, args: list[str]) -> dict[str, float | str]:
    proc = subprocess.run([str(path), *args], capture_output=True, text=True, check=True)
    return parse_kv(proc.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU vs GPU Monte Carlo engines")
    parser.add_argument("--build-dir", default="build", type=Path)
    parser.add_argument("--paths", type=int, default=10_000_000)
    parser.add_argument("--steps", type=int, default=365)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", default=Path("results/benchmark_results.csv"), type=Path)
    args = parser.parse_args()

    cpu_bin = args.build_dir / "mc_cpu"
    gpu_bin = args.build_dir / "mc_gpu"

    common = ["--paths", str(args.paths), "--steps", str(args.steps)]
    rows: list[dict[str, float | str]] = []

    for i in range(args.runs):
      cpu = run_binary(cpu_bin, common)
      gpu = run_binary(gpu_bin, common)
      speedup = float(cpu["runtime_ms"]) / float(gpu["runtime_ms"])
      price_diff = abs(float(cpu["price"]) - float(gpu["price"]))
      row = {
          "run": i + 1,
          "paths": args.paths,
          "steps": args.steps,
          "cpu_ms": float(cpu["runtime_ms"]),
          "gpu_ms": float(gpu["runtime_ms"]),
          "speedup": speedup,
          "cpu_price": float(cpu["price"]),
          "gpu_price": float(gpu["price"]),
          "price_abs_diff": price_diff,
      }
      rows.append(row)
      print(json.dumps(row, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg_speedup = sum(r["speedup"] for r in rows) / len(rows)
    avg_cpu_ms = sum(r["cpu_ms"] for r in rows) / len(rows)
    avg_gpu_ms = sum(r["gpu_ms"] for r in rows) / len(rows)
    print("\nSummary")
    print(f"avg_cpu_ms={avg_cpu_ms:.3f}")
    print(f"avg_gpu_ms={avg_gpu_ms:.3f}")
    print(f"avg_speedup={avg_speedup:.2f}x")


if __name__ == "__main__":
    main()
