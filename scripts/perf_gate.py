#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Performance regression gate for benchmark/convergence outputs.")
    parser.add_argument("--benchmark-csv", type=Path, default=Path("results/benchmark_results.csv"))
    parser.add_argument("--convergence-csv", type=Path, default=Path("results/convergence/gpu_convergence.csv"))
    parser.add_argument("--thresholds", type=Path, default=Path("configs/perf_gate_thresholds.json"))
    args = parser.parse_args()

    th = json.loads(args.thresholds.read_text())
    bench = load_rows(args.benchmark_csv)
    conv = load_rows(args.convergence_csv)

    speedups = [float(r["speedup"]) for r in bench]
    gpu_ms = [float(r["gpu_ms"]) for r in bench]
    price_diff = [float(r["price_abs_diff"]) for r in bench]

    avg_speedup = mean(speedups)
    avg_gpu_ms = mean(gpu_ms)
    max_diff = max(price_diff)

    print(f"avg_speedup={avg_speedup:.3f}")
    print(f"avg_gpu_ms={avg_gpu_ms:.3f}")
    print(f"max_price_abs_diff={max_diff:.6f}")

    bth = th["benchmark"]
    failures: list[str] = []
    if avg_speedup < float(bth["min_avg_speedup"]):
        failures.append(
            f"avg_speedup {avg_speedup:.3f} < min_avg_speedup {float(bth['min_avg_speedup']):.3f}"
        )
    if avg_gpu_ms > float(bth["max_avg_gpu_ms"]):
        failures.append(f"avg_gpu_ms {avg_gpu_ms:.3f} > max_avg_gpu_ms {float(bth['max_avg_gpu_ms']):.3f}")
    if max_diff > float(bth["max_price_abs_diff"]):
        failures.append(
            f"max_price_abs_diff {max_diff:.6f} > max_price_abs_diff {float(bth['max_price_abs_diff']):.6f}"
        )

    by_variant: dict[str, list[dict[str, str]]] = {}
    for r in conv:
        by_variant.setdefault(r["variant"], []).append(r)

    baseline = by_variant["baseline"]
    anti = by_variant["antithetic"]
    control = by_variant["control"]
    combo = by_variant["antithetic+control"]

    def avg_reduction(target: list[dict[str, str]]) -> float:
        vals = []
        for b, t in zip(baseline, target):
            bw = float(b["ci95_width"])
            tw = float(t["ci95_width"])
            vals.append(bw / max(tw, 1e-12))
        return mean(vals)

    anti_red = avg_reduction(anti)
    control_red = avg_reduction(control)
    combo_red = avg_reduction(combo)
    print(f"avg_ci_width_reduction_antithetic={anti_red:.3f}")
    print(f"avg_ci_width_reduction_control={control_red:.3f}")
    print(f"avg_ci_width_reduction_antithetic_control={combo_red:.3f}")

    cth = th["convergence"]
    if anti_red < float(cth["min_antithetic_reduction"]):
        failures.append(
            f"antithetic reduction {anti_red:.3f} < {float(cth['min_antithetic_reduction']):.3f}"
        )
    if control_red < float(cth["min_control_reduction"]):
        failures.append(f"control reduction {control_red:.3f} < {float(cth['min_control_reduction']):.3f}")
    if combo_red < float(cth["min_combo_reduction"]):
        failures.append(
            f"antithetic+control reduction {combo_red:.3f} < {float(cth['min_combo_reduction']):.3f}"
        )

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit("perf_gate=failed")
    print("perf_gate=passed")


if __name__ == "__main__":
    main()
