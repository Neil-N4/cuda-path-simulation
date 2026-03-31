#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
from pathlib import Path


STRING_KEYS = {"engine", "rng_mode", "math_mode"}


def parse_kv(stdout: str) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for line in stdout.strip().splitlines():
        if "=" not in line:
            continue
        k, v = line.strip().split("=", 1)
        out[k] = v if k in STRING_KEYS else float(v)
    return out


def run_binary(path: Path, args: list[str]) -> dict[str, float | str]:
    proc = subprocess.run([str(path), *args], capture_output=True, text=True, check=True)
    return parse_kv(proc.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CUDA engine across RNG/math modes.")
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--paths", type=int, default=2_000_000)
    parser.add_argument("--steps", type=int, default=365)
    parser.add_argument("--payoff", choices=["european", "asian", "upout"], default="european")
    parser.add_argument("--barrier", type=float, default=130.0)
    parser.add_argument("--antithetic", action="store_true")
    parser.add_argument("--control-variate", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/modes/mode_sweep.csv"))
    args = parser.parse_args()

    cpu_bin = args.build_dir / "mc_cpu"
    gpu_bin = args.build_dir / "mc_gpu"

    base_args = [
        "--paths", str(args.paths),
        "--steps", str(args.steps),
        "--payoff", args.payoff,
        "--barrier", str(args.barrier),
    ]
    if args.antithetic:
        base_args.append("--antithetic")
    if args.control_variate:
        base_args.append("--control-variate")

    modes = [
        ("baseline", "philox", "fp32"),
        ("sobol", "sobol", "fp32"),
        ("mixed", "philox", "mixed"),
        ("sobol_mixed", "sobol", "mixed"),
    ]

    rows: list[dict[str, float | str]] = []
    for name, rng, math in modes:
        if rng == "sobol" and math == "mixed":
            row = {
                "mode": name,
                "rng_mode": rng,
                "math_mode": math,
                "status": "unsupported",
                "cpu_ms": "",
                "gpu_ms": "",
                "speedup": "",
                "cpu_price": "",
                "gpu_price": "",
                "price_abs_diff": "",
                "cpu_ci95_low": "",
                "cpu_ci95_high": "",
                "gpu_ci95_low": "",
                "gpu_ci95_high": "",
                "ci_overlap": "",
            }
            rows.append(row)
            print(json.dumps(row, indent=2))
            continue
        mode_args = [*base_args, "--rng", rng, "--math", math]
        cpu = run_binary(cpu_bin, mode_args)
        gpu = run_binary(gpu_bin, mode_args)
        price_key = "price_cv" if args.control_variate else "price"
        ci_l_key = "ci95_low_cv" if args.control_variate else "ci95_low"
        ci_h_key = "ci95_high_cv" if args.control_variate else "ci95_high"
        cpu_price = float(cpu[price_key])
        gpu_price = float(gpu[price_key])
        cpu_l = float(cpu[ci_l_key])
        cpu_h = float(cpu[ci_h_key])
        gpu_l = float(gpu[ci_l_key])
        gpu_h = float(gpu[ci_h_key])
        overlap = not (cpu_h < gpu_l or gpu_h < cpu_l)
        row = {
            "mode": name,
            "rng_mode": rng,
            "math_mode": math,
            "status": "ok",
            "cpu_ms": float(cpu["runtime_ms"]),
            "gpu_ms": float(gpu["runtime_ms"]),
            "speedup": float(cpu["runtime_ms"]) / float(gpu["runtime_ms"]),
            "cpu_price": cpu_price,
            "gpu_price": gpu_price,
            "price_abs_diff": abs(cpu_price - gpu_price),
            "cpu_ci95_low": cpu_l,
            "cpu_ci95_high": cpu_h,
            "gpu_ci95_low": gpu_l,
            "gpu_ci95_high": gpu_h,
            "ci_overlap": 1 if overlap else 0,
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSummary")
    for row in rows:
        if row["status"] != "ok":
            print(f"{row['mode']}: status={row['status']}")
            continue
        print(
            f"{row['mode']}: "
            f"gpu_ms={float(row['gpu_ms']):.3f} "
            f"speedup={float(row['speedup']):.2f}x "
            f"price_abs_diff={float(row['price_abs_diff']):.6f} "
            f"ci_overlap={int(row['ci_overlap'])}"
        )


if __name__ == "__main__":
    main()
