#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def parse_kv(stdout: str) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for line in stdout.strip().splitlines():
        if "=" not in line:
            continue
        k, v = line.strip().split("=", 1)
        out[k] = v if k == "engine" else float(v)
    return out


def run(bin_path: Path, args: list[str]) -> dict[str, float | str]:
    p = subprocess.run([str(bin_path), *args], capture_output=True, text=True, check=True)
    return parse_kv(p.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check CPU/GPU output parity")
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--paths", type=int, default=2_000_000)
    parser.add_argument("--steps", type=int, default=365)
    parser.add_argument("--payoff", choices=["european", "asian", "upout"], default="european")
    parser.add_argument("--barrier", type=float, default=130.0)
    parser.add_argument("--price-tol", type=float, default=0.05)
    parser.add_argument("--require-ci-overlap", action="store_true")
    parser.add_argument("--antithetic", action="store_true")
    parser.add_argument("--control-variate", action="store_true")
    args = parser.parse_args()

    common = [
        "--paths",
        str(args.paths),
        "--steps",
        str(args.steps),
        "--payoff",
        args.payoff,
        "--barrier",
        str(args.barrier),
    ]
    if args.antithetic:
        common.append("--antithetic")
    if args.control_variate:
        common.append("--control-variate")

    cpu = run(args.build_dir / "mc_cpu", common)
    gpu = run(args.build_dir / "mc_gpu", common)

    price_key = "price_cv" if args.control_variate else "price"
    ci_l_key = "ci95_low_cv" if args.control_variate else "ci95_low"
    ci_h_key = "ci95_high_cv" if args.control_variate else "ci95_high"
    diff = abs(float(cpu[price_key]) - float(gpu[price_key]))
    print(f"price_key={price_key}")
    print(f"cpu_price={cpu[price_key]}")
    print(f"gpu_price={gpu[price_key]}")
    print(f"abs_diff={diff:.6f}")
    print(f"tolerance={args.price_tol:.6f}")

    if diff > args.price_tol:
        raise SystemExit("Parity check failed")

    if args.require_ci_overlap:
        cpu_l, cpu_h = float(cpu[ci_l_key]), float(cpu[ci_h_key])
        gpu_l, gpu_h = float(gpu[ci_l_key]), float(gpu[ci_h_key])
        overlap = not (cpu_h < gpu_l or gpu_h < cpu_l)
        print(f"ci_overlap={1 if overlap else 0}")
        if not overlap:
            raise SystemExit("CI overlap check failed")

    print("Parity check passed")


if __name__ == "__main__":
    main()
