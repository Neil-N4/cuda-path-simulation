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
    parser.add_argument("--price-tol", type=float, default=0.05)
    args = parser.parse_args()

    common = ["--paths", str(args.paths), "--steps", str(args.steps)]

    cpu = run(args.build_dir / "mc_cpu", common)
    gpu = run(args.build_dir / "mc_gpu", common)

    diff = abs(float(cpu["price"]) - float(gpu["price"]))
    print(f"cpu_price={cpu['price']}")
    print(f"gpu_price={gpu['price']}")
    print(f"abs_diff={diff:.6f}")
    print(f"tolerance={args.price_tol:.6f}")

    if diff > args.price_tol:
        raise SystemExit("Parity check failed")

    print("Parity check passed")


if __name__ == "__main__":
    main()
