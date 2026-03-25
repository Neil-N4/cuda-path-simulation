#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU/GPU engine from JSON config.")
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    engine = cfg.get("engine", "gpu")
    bin_name = "mc_gpu" if engine == "gpu" else "mc_cpu"
    bin_path = args.build_dir / bin_name

    cli = [str(bin_path)]
    for key in ("paths", "steps", "s0", "strike", "rate", "vol", "maturity", "barrier"):
        if key in cfg:
            cli.extend([f"--{key}", str(cfg[key])])
    if "payoff" in cfg:
      cli.extend(["--payoff", str(cfg["payoff"])])
    if cfg.get("antithetic", False):
      cli.append("--antithetic")
    if cfg.get("control_variate", False):
      cli.append("--control-variate")

    print(" ".join(cli))
    subprocess.run(cli, check=True)


if __name__ == "__main__":
    main()
