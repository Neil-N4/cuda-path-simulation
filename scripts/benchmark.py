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
    parser.add_argument(
        "--payoff",
        choices=["european", "asian", "upout"],
        default="european",
        help="Payoff model to evaluate",
    )
    parser.add_argument("--barrier", type=float, default=130.0)
    parser.add_argument(
        "--antithetic",
        action="store_true",
        help="Enable antithetic variates in both CPU and GPU engines",
    )
    parser.add_argument(
        "--control-variate",
        action="store_true",
        help="Enable control variate estimator in both CPU and GPU engines",
    )
    parser.add_argument("--out", default=Path("results/benchmark_results.csv"), type=Path)
    args = parser.parse_args()

    cpu_bin = args.build_dir / "mc_cpu"
    gpu_bin = args.build_dir / "mc_gpu"

    common = [
        "--paths", str(args.paths),
        "--steps", str(args.steps),
        "--payoff", args.payoff,
        "--barrier", str(args.barrier),
    ]
    if args.antithetic:
        common.append("--antithetic")
    if args.control_variate:
        common.append("--control-variate")
    rows: list[dict[str, float | str]] = []

    for i in range(args.runs):
      cpu = run_binary(cpu_bin, common)
      gpu = run_binary(gpu_bin, common)
      speedup = float(cpu["runtime_ms"]) / float(gpu["runtime_ms"])
      price_key = "price_cv" if args.control_variate else "price"
      price_diff = abs(float(cpu[price_key]) - float(gpu[price_key]))
      row = {
          "run": i + 1,
          "paths": args.paths,
          "steps": args.steps,
          "payoff": args.payoff,
          "barrier": args.barrier,
          "cpu_ms": float(cpu["runtime_ms"]),
          "gpu_ms": float(gpu["runtime_ms"]),
          "speedup": speedup,
          "cpu_price": float(cpu["price"]),
          "gpu_price": float(gpu["price"]),
          "cpu_price_cv": float(cpu["price_cv"]),
          "gpu_price_cv": float(gpu["price_cv"]),
          "cpu_ci95_low": float(cpu["ci95_low"]),
          "cpu_ci95_high": float(cpu["ci95_high"]),
          "gpu_ci95_low": float(gpu["ci95_low"]),
          "gpu_ci95_high": float(gpu["ci95_high"]),
          "cpu_ci95_low_cv": float(cpu["ci95_low_cv"]),
          "cpu_ci95_high_cv": float(cpu["ci95_high_cv"]),
          "gpu_ci95_low_cv": float(gpu["ci95_low_cv"]),
          "gpu_ci95_high_cv": float(gpu["ci95_high_cv"]),
          "cpu_cv_beta": float(cpu["cv_beta"]),
          "gpu_cv_beta": float(gpu["cv_beta"]),
          "cpu_delta": float(cpu["delta"]),
          "gpu_delta": float(gpu["delta"]),
          "cpu_vega": float(cpu["vega"]),
          "gpu_vega": float(gpu["vega"]),
          "cpu_abs_error_delta_bs": float(cpu["abs_error_delta_bs"]),
          "gpu_abs_error_delta_bs": float(gpu["abs_error_delta_bs"]),
          "cpu_abs_error_vega_bs": float(cpu["abs_error_vega_bs"]),
          "gpu_abs_error_vega_bs": float(gpu["abs_error_vega_bs"]),
          "cpu_abs_error_bs": float(cpu["abs_error_bs"]),
          "gpu_abs_error_bs": float(gpu["abs_error_bs"]),
          "price_abs_diff": price_diff,
          "antithetic": 1 if args.antithetic else 0,
          "control_variate": 1 if args.control_variate else 0,
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
