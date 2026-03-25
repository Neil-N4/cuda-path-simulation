#!/usr/bin/env python3
import argparse
import csv
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


def parse_kv(stdout: str) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for line in stdout.strip().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k] = v if k == "engine" else float(v)
    return out


def run_binary(bin_path: Path, args: list[str]) -> dict[str, float | str]:
    proc = subprocess.run([str(bin_path), *args], capture_output=True, text=True, check=True)
    return parse_kv(proc.stdout)


def run_series(
    bin_path: Path,
    paths_grid: list[int],
    steps: int,
    antithetic: bool,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for paths in paths_grid:
        args = ["--paths", str(paths), "--steps", str(steps)]
        if antithetic:
            args.append("--antithetic")
        out = run_binary(bin_path, args)
        row = {
            "paths": float(paths),
            "price": float(out["price"]),
            "bs_price": float(out["bs_price"]),
            "abs_error_bs": float(out["abs_error_bs"]),
            "ci95_low": float(out["ci95_low"]),
            "ci95_high": float(out["ci95_high"]),
            "runtime_ms": float(out["runtime_ms"]),
        }
        rows.append(row)
    return rows


def plot_convergence(
    out_dir: Path,
    paths_grid: list[int],
    baseline: list[dict[str, float]],
    anti: list[dict[str, float]],
) -> None:
    x = paths_grid
    b_err = [r["abs_error_bs"] for r in baseline]
    a_err = [r["abs_error_bs"] for r in anti]
    b_ci = [r["ci95_high"] - r["ci95_low"] for r in baseline]
    a_ci = [r["ci95_high"] - r["ci95_low"] for r in anti]

    plt.figure(figsize=(8.5, 5))
    plt.loglog(x, b_err, marker="o", label="No antithetic")
    plt.loglog(x, a_err, marker="o", label="Antithetic")
    plt.xlabel("Trajectories")
    plt.ylabel("Absolute Error vs Black-Scholes")
    plt.title("Monte Carlo Convergence Error")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "convergence_error.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 5))
    plt.loglog(x, b_ci, marker="o", label="No antithetic")
    plt.loglog(x, a_ci, marker="o", label="Antithetic")
    plt.xlabel("Trajectories")
    plt.ylabel("95% CI Width")
    plt.title("Estimator Confidence Interval Width")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "convergence_ci_width.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate convergence report (error + CI width) vs Black-Scholes."
    )
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--engine", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--steps", type=int, default=365)
    parser.add_argument(
        "--paths-grid",
        type=str,
        default="250000,500000,1000000,2000000,4000000,8000000",
        help="Comma-separated trajectory counts",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/convergence"))
    args = parser.parse_args()

    bin_name = "mc_gpu" if args.engine == "gpu" else "mc_cpu"
    bin_path = args.build_dir / bin_name
    paths_grid = [int(x.strip()) for x in args.paths_grid.split(",") if x.strip()]

    baseline = run_series(bin_path, paths_grid, args.steps, antithetic=False)
    anti = run_series(bin_path, paths_grid, args.steps, antithetic=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.engine}_convergence.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "paths",
                "variant",
                "price",
                "bs_price",
                "abs_error_bs",
                "ci95_low",
                "ci95_high",
                "ci95_width",
                "runtime_ms",
            ]
        )
        for row in baseline:
            writer.writerow(
                [
                    int(row["paths"]),
                    "baseline",
                    row["price"],
                    row["bs_price"],
                    row["abs_error_bs"],
                    row["ci95_low"],
                    row["ci95_high"],
                    row["ci95_high"] - row["ci95_low"],
                    row["runtime_ms"],
                ]
            )
        for row in anti:
            writer.writerow(
                [
                    int(row["paths"]),
                    "antithetic",
                    row["price"],
                    row["bs_price"],
                    row["abs_error_bs"],
                    row["ci95_low"],
                    row["ci95_high"],
                    row["ci95_high"] - row["ci95_low"],
                    row["runtime_ms"],
                ]
            )

    plot_convergence(args.out_dir, paths_grid, baseline, anti)

    avg_ratio = sum(
        (b["ci95_high"] - b["ci95_low"]) / max(a["ci95_high"] - a["ci95_low"], 1e-12)
        for b, a in zip(baseline, anti)
    ) / len(paths_grid)
    print(f"engine={args.engine}")
    print(f"rows={2 * len(paths_grid)}")
    print(f"avg_ci_width_reduction_factor={avg_ratio:.3f}")
    print(f"csv={csv_path}")
    print(f"plot_error={args.out_dir / 'convergence_error.png'}")
    print(f"plot_ci={args.out_dir / 'convergence_ci_width.png'}")


if __name__ == "__main__":
    main()
