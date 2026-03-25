# CUDA-Accelerated Path Simulation

Monte Carlo path simulation engine for European call pricing with a native C++20 CPU baseline and a CUDA implementation tuned for throughput, statistical rigor, and quant-style validation.

## Tech Stack

C++20, CUDA, Python, Nsight Systems, CMake, Valgrind

## Highlights

- Native C++/CUDA implementation with **on-device RNG (`curand`)**
- GPU kernel uses vectorized RNG (`curand_normal4`) and **shared-memory reduction**
- Multi-stream execution with **asynchronous device-to-host copies**
- CPU baseline for parity and speedup benchmarking
- Reproducible benchmark and plotting scripts
- **Antithetic variates** support for variance reduction (`--antithetic`)
- 95% confidence intervals and **Black-Scholes error tracking** in binary outputs
- Convergence report generator (error + CI width vs trajectory count)

## Model

Simulates Geometric Brownian Motion paths:

- `S(t + dt) = S(t) * exp((r - 0.5*sigma^2) * dt + sigma * sqrt(dt) * Z)`
- Payoff: `max(S(T) - K, 0)`
- Price: `exp(-rT) * E[payoff]`

## Project Structure

- `src/gpu_main.cu`: CUDA kernel + stream orchestration
- `src/cpu_main.cpp`: CPU baseline engine
- `include/`: common config and argument parsing
- `scripts/benchmark.py`: CPU vs GPU benchmark
- `scripts/validate_parity.py`: CPU/GPU price parity gate
- `scripts/profile_nsight.sh`: Nsight Systems profiling command

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

Adjust `CMAKE_CUDA_ARCHITECTURES` for your GPU.

## Run

```bash
./build/mc_cpu --paths 1000000 --steps 365
./build/mc_gpu --paths 10000000 --steps 365
./build/mc_gpu --paths 10000000 --steps 365 --antithetic
```

Each binary emits:

- `price`, `ci95_low`, `ci95_high`
- `bs_price`, `abs_error_bs`
- `sample_count`, `trajectory_count`
- `payoff_stddev`, `payoff_stderr`

## Benchmark

```bash
python3 scripts/benchmark.py --build-dir build --paths 10000000 --steps 365 --runs 3
python3 scripts/benchmark.py --build-dir build --paths 10000000 --steps 365 --runs 3 --antithetic
python3 scripts/plot_benchmark.py --csv results/benchmark_results.csv --out results/runtime_comparison.png
```

## Validation Gate

```bash
python3 scripts/validate_parity.py --build-dir build --paths 2000000 --steps 365 --price-tol 0.05
python3 scripts/validate_parity.py --build-dir build --paths 2000000 --steps 365 --price-tol 0.05 --antithetic --require-ci-overlap
```

## Convergence Report (Quant Validation)

```bash
python3 scripts/convergence_report.py --build-dir build --engine gpu --steps 365
python3 scripts/convergence_report.py --build-dir build --engine cpu --steps 365
```

Outputs in `results/convergence/`:

- `*_convergence.csv`
- `convergence_error.png` (abs error vs Black-Scholes)
- `convergence_ci_width.png` (95% CI width trend)

## Nsight Systems Profiling

```bash
bash scripts/profile_nsight.sh build 10000000 365
```

## Valgrind (CPU baseline)

```bash
valgrind --leak-check=full ./build/mc_cpu --paths 500000 --steps 365
```

## Resume-Ready Metrics Workflow

1. Run `benchmark.py` with and without `--antithetic`
2. Run `convergence_report.py` and capture CI-width reduction and error curves
3. Use speedup + CI/error reductions in your bullets
4. Attach `results/runtime_comparison.png` and convergence plots in portfolio
