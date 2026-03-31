#!/usr/bin/env python3
import argparse
import csv
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


def parse_kv(stdout: str) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for line in stdout.strip().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k] = v if k in {"engine", "rng_mode", "math_mode"} else float(v)
    return out


def run_binary(bin_path: Path, args: list[str]) -> dict[str, float | str]:
    proc = subprocess.run([str(bin_path), *args], capture_output=True, text=True, check=True)
    return parse_kv(proc.stdout)


def run_variant(
    bin_path: Path,
    paths_grid: list[int],
    steps: int,
    payoff: str,
    barrier: float,
    antithetic: bool,
    control_variate: bool,
    rng: str,
    math: str,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for paths in paths_grid:
        args = [
            "--paths", str(paths),
            "--steps", str(steps),
            "--payoff", payoff,
            "--barrier", str(barrier),
            "--rng", rng,
            "--math", math,
        ]
        if antithetic:
            args.append("--antithetic")
        if control_variate:
            args.append("--control-variate")
        out = run_binary(bin_path, args)
        price_key = "price_cv" if control_variate else "price"
        ci_l_key = "ci95_low_cv" if control_variate else "ci95_low"
        ci_h_key = "ci95_high_cv" if control_variate else "ci95_high"
        rows.append(
            {
                "paths": float(paths),
                "price": float(out[price_key]),
                "bs_price": float(out["bs_price"]),
                "abs_error_bs": float(out["abs_error_bs"]),
                "ci95_low": float(out[ci_l_key]),
                "ci95_high": float(out[ci_h_key]),
                "runtime_ms": float(out["runtime_ms"]),
            }
        )
    return rows


def plot_convergence(out_dir: Path, paths_grid: list[int], series: dict[str, list[dict[str, float]]]) -> None:
    x = paths_grid

    plt.figure(figsize=(8.5, 5))
    for name, rows in series.items():
        plt.loglog(x, [r["abs_error_bs"] for r in rows], marker="o", label=name)
    plt.xlabel("Trajectories")
    plt.ylabel("Absolute Error vs Black-Scholes")
    plt.title("Convergence Error Across Estimators")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "convergence_error.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 5))
    for name, rows in series.items():
        plt.loglog(x, [r["ci95_high"] - r["ci95_low"] for r in rows], marker="o", label=name)
    plt.xlabel("Trajectories")
    plt.ylabel("95% CI Width")
    plt.title("CI Width Across Estimators")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "convergence_ci_width.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate convergence report for baseline/antithetic/control variants."
    )
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--engine", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--steps", type=int, default=365)
    parser.add_argument("--payoff", choices=["european", "asian", "upout"], default="european")
    parser.add_argument("--barrier", type=float, default=130.0)
    parser.add_argument("--rng", choices=["philox", "sobol"], default="philox")
    parser.add_argument("--math", choices=["fp32", "mixed"], default="fp32")
    parser.add_argument(
        "--paths-grid",
        type=str,
        default="250000,500000,1000000,2000000,4000000,8000000",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/convergence"))
    args = parser.parse_args()
    if args.payoff != "european":
        raise SystemExit("convergence_report currently supports --payoff european only (uses BS reference).")

    bin_name = "mc_gpu" if args.engine == "gpu" else "mc_cpu"
    bin_path = args.build_dir / bin_name
    paths_grid = [int(x.strip()) for x in args.paths_grid.split(",") if x.strip()]

    variants = {
        "baseline": (False, False),
        "antithetic": (True, False),
        "control": (False, True),
        "antithetic+control": (True, True),
    }
    series = {
        name: run_variant(
            bin_path,
            paths_grid,
            args.steps,
            args.payoff,
            args.barrier,
            antithetic,
            control_variate,
            args.rng,
            args.math,
        )
        for name, (antithetic, control_variate) in variants.items()
    }

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
        for name, rows in series.items():
            for row in rows:
                writer.writerow(
                    [
                        int(row["paths"]),
                        name,
                        row["price"],
                        row["bs_price"],
                        row["abs_error_bs"],
                        row["ci95_low"],
                        row["ci95_high"],
                        row["ci95_high"] - row["ci95_low"],
                        row["runtime_ms"],
                    ]
                )

    plot_convergence(args.out_dir, paths_grid, series)

    baseline = series["baseline"]
    anti = series["antithetic"]
    control = series["control"]
    anti_control = series["antithetic+control"]
    anti_factor = sum(
        (b["ci95_high"] - b["ci95_low"]) / max(a["ci95_high"] - a["ci95_low"], 1e-12)
        for b, a in zip(baseline, anti)
    ) / len(paths_grid)
    control_factor = sum(
        (b["ci95_high"] - b["ci95_low"]) / max(c["ci95_high"] - c["ci95_low"], 1e-12)
        for b, c in zip(baseline, control)
    ) / len(paths_grid)
    combo_factor = sum(
        (b["ci95_high"] - b["ci95_low"]) / max(ac["ci95_high"] - ac["ci95_low"], 1e-12)
        for b, ac in zip(baseline, anti_control)
    ) / len(paths_grid)

    print(f"engine={args.engine}")
    print(f"rows={len(paths_grid) * len(variants)}")
    print(f"avg_ci_width_reduction_antithetic={anti_factor:.3f}")
    print(f"avg_ci_width_reduction_control={control_factor:.3f}")
    print(f"avg_ci_width_reduction_antithetic_control={combo_factor:.3f}")
    print(f"csv={csv_path}")
    print(f"plot_error={args.out_dir / 'convergence_error.png'}")
    print(f"plot_ci={args.out_dir / 'convergence_ci_width.png'}")


if __name__ == "__main__":
    main()
