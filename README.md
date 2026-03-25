# CUDA-Accelerated Path Simulation Engine

Quant-oriented Monte Carlo pricing engine with matched CPU/CUDA implementations, variance-reduction estimators, pathwise Greeks, convergence reporting, stress tests, and profiling hooks.

## Tech Stack

C++20, CUDA, Python, CMake, Nsight Systems, Nsight Compute, Valgrind

## Core Capabilities

- Native C++/CUDA pricing engine with on-device `curand` RNG
- CUDA kernel with vectorized random draws (`curand_normal4`), shared-memory reductions, coalesced writes, and stream-overlapped async copies
- Multiple payoff models:
  - `european` call
  - `asian` arithmetic-average call
  - `upout` up-and-out barrier call
- Variance-reduction estimators:
  - antithetic variates (`--antithetic`)
  - control variate using discounted terminal price (`--control-variate`)
- Quant validation outputs:
  - 95% confidence intervals (`ci95_*`)
  - Black-Scholes price error (`abs_error_bs`) for European calls
  - pathwise Greeks (`delta`, `vega`) + BS reference error
- Engineering gates:
  - CPU/GPU parity checks
  - robustness stress suite
  - convergence report across estimator variants

## Project Layout

- `src/cpu_main.cpp`: CPU engine + pricing/Greeks estimators
- `src/gpu_main.cu`: CUDA engine + stream orchestration
- `include/`: config + CLI parsing
- `scripts/benchmark.py`: benchmark harness (CPU vs GPU)
- `scripts/validate_parity.py`: parity + CI-overlap checks
- `scripts/convergence_report.py`: convergence + CI-width reduction report
- `scripts/stress_suite.py`: robustness scenarios
- `scripts/perf_gate.py`: hard performance regression gate
- `scripts/profile_nsight.sh`: Nsight Systems capture
- `scripts/profile_ncu.sh`: Nsight Compute capture
- `scripts/profile_ncu_csv.sh`: Nsight Compute CSV capture
- `scripts/compare_ncu_csv.py`: before/after Nsight CSV comparison
- `docs/NSIGHT_REPORT_TEMPLATE.md`: profiler report template

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

Set `CMAKE_CUDA_ARCHITECTURES` for your GPU (e.g. T4=`75`, RTX30=`86`, RTX40=`89`).

## Run

### European call (baseline)

```bash
./build/mc_gpu --paths 10000000 --steps 365 --payoff european
```

### Antithetic + control variate (recommended)

```bash
./build/mc_gpu --paths 10000000 --steps 365 --payoff european --antithetic --control-variate
```

### Model variants

```bash
./build/mc_gpu --paths 10000000 --steps 365 --payoff asian --antithetic --control-variate
./build/mc_gpu --paths 10000000 --steps 365 --payoff upout --barrier 120 --antithetic --control-variate
```

## Output Fields

Both CPU and GPU binaries emit key-value metrics including:

- estimator:
  - `price`, `price_cv`, `cv_beta`
  - `ci95_low`, `ci95_high`, `ci95_low_cv`, `ci95_high_cv`
- quant reference (European):
  - `bs_price`, `abs_error_bs`
  - `delta`, `vega`, `bs_delta`, `bs_vega`
  - `abs_error_delta_bs`, `abs_error_vega_bs`
- runtime and sample counts:
  - `runtime_ms`, `sample_count`, `trajectory_count`

## Benchmark

```bash
python3 scripts/benchmark.py --build-dir build --paths 10000000 --steps 365 --runs 3 --payoff european
python3 scripts/benchmark.py --build-dir build --paths 10000000 --steps 365 --runs 3 --payoff european --antithetic --control-variate
python3 scripts/plot_benchmark.py --csv results/benchmark_results.csv --out results/runtime_comparison.png
```

## Validation Gates

```bash
python3 scripts/validate_parity.py --build-dir build --paths 2000000 --steps 365 --payoff european --price-tol 0.05
python3 scripts/validate_parity.py --build-dir build --paths 2000000 --steps 365 --payoff european --antithetic --control-variate --price-tol 0.05 --require-ci-overlap
python3 scripts/stress_suite.py --build-dir build
```

## Performance Gate

Run after benchmark + convergence artifacts are generated:

```bash
python3 scripts/perf_gate.py \
  --benchmark-csv results/benchmark_results.csv \
  --convergence-csv results/convergence/gpu_convergence.csv \
  --thresholds configs/perf_gate_thresholds.json
```

## Convergence Report

Runs four estimators: baseline, antithetic, control, antithetic+control.

```bash
python3 scripts/convergence_report.py --build-dir build --engine gpu --steps 365 --payoff european
```

Artifacts:

- `results/convergence/gpu_convergence.csv`
- `results/convergence/convergence_error.png`
- `results/convergence/convergence_ci_width.png`

## Profiling

### Nsight Systems

```bash
bash scripts/profile_nsight.sh build 10000000 365
```

### Nsight Compute

```bash
bash scripts/profile_ncu.sh build 10000000 365 european
bash scripts/profile_ncu_csv.sh build results/ncu_metrics_after.csv 10000000 365
python3 scripts/compare_ncu_csv.py --before results/ncu_metrics_before.csv --after results/ncu_metrics_after.csv
```

Use `docs/NSIGHT_REPORT_TEMPLATE.md` to capture occupancy/memory/warp-stall evidence.

## Make Targets

```bash
make build
make benchmark-anti-cv
make validate-cv
make convergence-gpu
make stress
make perf-gate
make nsight-compute
make nsight-csv
```

## CI

- `.github/workflows/cpu_ci.yml`: CPU-only gate on push/PR.
- `.github/workflows/gpu_perf_gate.yml`: manual GPU perf gate for self-hosted NVIDIA runners.

## Resume Evidence Workflow

1. Run `benchmark.py` with/without antithetic/control
2. Run `convergence_report.py` and capture CI-width reduction factors
3. Run `validate_parity.py` + `stress_suite.py` for reliability proof
4. Capture Nsight metrics and complete `docs/NSIGHT_REPORT_TEMPLATE.md`
5. Attach generated plots/tables to portfolio
