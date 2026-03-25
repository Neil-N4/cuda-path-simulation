#!/usr/bin/env python3
import argparse
import math
import subprocess
from pathlib import Path


def parse_kv(stdout: str) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for line in stdout.strip().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k] = v if k == "engine" else float(v)
    return out


def run(bin_path: Path, args: list[str]) -> dict[str, float | str]:
    p = subprocess.run([str(bin_path), *args], capture_output=True, text=True, check=True)
    return parse_kv(p.stdout)


def assert_finite(out: dict[str, float | str], keys: list[str], name: str) -> None:
    for k in keys:
        val = float(out[k])
        if not math.isfinite(val):
            raise RuntimeError(f"{name}: non-finite {k}={val}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness scenarios for CPU/GPU engines.")
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--paths", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=365)
    args = parser.parse_args()

    cpu = args.build_dir / "mc_cpu"
    gpu = args.build_dir / "mc_gpu"

    scenarios = [
        ("eur_nominal", ["--payoff", "european", "--vol", "0.2"]),
        ("eur_high_vol", ["--payoff", "european", "--vol", "1.2"]),
        ("eur_low_rate", ["--payoff", "european", "--rate", "-0.01"]),
        ("asian", ["--payoff", "asian", "--vol", "0.35"]),
        ("upout", ["--payoff", "upout", "--barrier", "120", "--vol", "0.25"]),
    ]

    for name, extra in scenarios:
        common = [
            "--paths", str(args.paths),
            "--steps", str(args.steps),
            "--antithetic",
            "--control-variate",
            *extra,
        ]
        out_cpu = run(cpu, common)
        out_gpu = run(gpu, common)

        key = "price_cv"
        diff = abs(float(out_cpu[key]) - float(out_gpu[key]))
        assert_finite(out_cpu, [key, "runtime_ms", "ci95_low_cv", "ci95_high_cv"], f"{name}/cpu")
        assert_finite(out_gpu, [key, "runtime_ms", "ci95_low_cv", "ci95_high_cv"], f"{name}/gpu")
        if diff > 0.15:
            raise RuntimeError(f"{name}: cpu/gpu {key} diff too large: {diff:.6f}")

        if extra[1] == "european":
            if float(out_cpu["abs_error_bs"]) > 0.08 or float(out_gpu["abs_error_bs"]) > 0.08:
                raise RuntimeError(f"{name}: BS error exceeded threshold")

        print(
            f"{name}: ok "
            f"cpu_{key}={float(out_cpu[key]):.6f} "
            f"gpu_{key}={float(out_gpu[key]):.6f} "
            f"diff={diff:.6f}"
        )

    print("stress_suite=passed")


if __name__ == "__main__":
    main()
