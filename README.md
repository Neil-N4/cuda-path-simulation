# CUDA-Accelerated Path Simulation Engine

High-performance Monte Carlo pricing engine with matched C++/CUDA implementations, variance-reduction estimators, Greeks validation, and hard regression gates.

## TL;DR

- 10M paths benchmarked on NVIDIA T4
- 67.85 s CPU vs 37.65 ms GPU
- 1852.66x average speedup
- CPU/GPU parity passed (`abs_diff=0.002882`, CI overlap=1)
- CI-width reduction vs baseline:
  - Antithetic: 1.342x
  - Control variate: 2.434x
  - Antithetic + control: 4.854x
- Stress suite passed across nominal/off-nominal scenarios

## Why This Exists

Monte Carlo pricing systems are both compute-heavy and numerically sensitive.
This project targets both goals:

- Throughput: production-scale GPU acceleration
- Reliability: reproducible validation, parity checks, and performance gates

## Features

- C++20 CPU baseline + CUDA GPU engine
- On-device RNG (`curand`) with vectorized draws (`curand_normal4`)
- Shared-memory reductions + coalesced writes + stream-overlapped async copies
- Payoffs:
  - `european`
  - `asian`
  - `upout` (up-and-out barrier)
- Variance reduction:
  - `--antithetic`
  - `--control-variate`
- Quant outputs:
  - 95% confidence intervals
  - Black-Scholes error (European)
  - Pathwise Greeks (`delta`, `vega`) + BS reference errors
- Engineering gates:
  - CPU/GPU parity checks
  - stress suite
  - performance regression gate

## Architecture

- `src/cpu_main.cpp`: CPU estimator + reference checks
- `src/gpu_main.cu`: CUDA kernel path + stream orchestration
- `scripts/benchmark.py`: benchmark harness
- `scripts/validate_parity.py`: parity + CI overlap checks
- `scripts/convergence_report.py`: convergence and CI-width analysis
- `scripts/stress_suite.py`: robustness scenarios
- `scripts/perf_gate.py`: threshold-based regression gate
- `scripts/profile_nsight.sh`, `scripts/profile_ncu.sh`, `scripts/profile_ncu_csv.sh`: profiling workflows

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j
```

For other GPUs: T4=`75`, RTX30=`86`, RTX40=`89`.

## Quick Run

```bash
./build/mc_gpu --paths 10000000 --steps 365 --payoff european --antithetic --control-variate
```

## Reproduce Main Results

```bash
python3 scripts/benchmark.py --build-dir build --paths 10000000 --steps 365 --runs 3 --payoff european --antithetic --control-variate
python3 scripts/validate_parity.py --build-dir build --paths 2000000 --steps 365 --payoff european --antithetic --control-variate --price-tol 0.05 --require-ci-overlap
python3 scripts/convergence_report.py --build-dir build --engine gpu --steps 365 --payoff european
python3 scripts/stress_suite.py --build-dir build
python3 scripts/perf_gate.py --benchmark-csv results/benchmark_results.csv --convergence-csv results/convergence/gpu_convergence.csv --thresholds configs/perf_gate_thresholds.json
```

## Output Fields

Binaries emit key-value metrics including:

- estimator:
  - `price`, `price_cv`, `cv_beta`
  - `ci95_low`, `ci95_high`, `ci95_low_cv`, `ci95_high_cv`
- quant reference (European):
  - `bs_price`, `abs_error_bs`
  - `delta`, `vega`, `bs_delta`, `bs_vega`
  - `abs_error_delta_bs`, `abs_error_vega_bs`
- runtime and scale:
  - `runtime_ms`, `sample_count`, `trajectory_count`

## Profiling Workflow

```bash
bash scripts/profile_nsight.sh build 10000000 365 european
bash scripts/profile_ncu.sh build 10000000 365 european
bash scripts/profile_ncu_csv.sh build results/ncu_metrics_after.csv 10000000 365
```

For before/after analysis:

```bash
python3 scripts/compare_ncu_csv.py --before results/ncu_metrics_before.csv --after results/ncu_metrics_after.csv
```

Use `docs/NSIGHT_REPORT_TEMPLATE.md` and `docs/PERF_GATES.md` for reporting.
See `docs/VERIFICATION_REPORT.md` for a one-page summary.

## CI

- `.github/workflows/cpu_ci.yml`: CPU gate on push/PR (`BUILD_CUDA=OFF`)
- `.github/workflows/gpu_perf_gate.yml`: manual GPU perf gate for self-hosted NVIDIA runners

## Project Structure

```text
src/
  cpu_main.cpp
  gpu_main.cu

scripts/
  benchmark.py
  validate_parity.py
  convergence_report.py
  stress_suite.py
  perf_gate.py
  profile_nsight.sh
  profile_ncu.sh
  profile_ncu_csv.sh
  compare_ncu_csv.py
  run_from_config.py

configs/
  nominal_european.json
  high_vol_upout.json
  perf_gate_thresholds.json

docs/
  NSIGHT_REPORT_TEMPLATE.md
  PERF_GATES.md
  VERIFICATION_REPORT.md
  RESUME_BULLETS.md
```

## Resume Translation

Built a C++/CUDA Monte Carlo pricing engine processing 10M paths with 1852.66x speedup, validated via Black-Scholes/Greeks checks, CI-width reductions up to 4.854x, and automated performance regression gates.
