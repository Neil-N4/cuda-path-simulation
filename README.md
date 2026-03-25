# CUDA-Accelerated Path Simulation

Monte Carlo path simulation engine for European call pricing with a native C++20 CPU baseline and a CUDA implementation tuned for throughput.

## Tech Stack

C++20, CUDA, Python, Nsight Systems, CMake, Valgrind

## Highlights

- Native C++/CUDA implementation with **on-device RNG (`curand`)**
- GPU kernel uses vectorized RNG (`curand_normal4`) and **shared-memory reduction**
- Multi-stream execution with **asynchronous device-to-host copies**
- CPU baseline for parity and speedup benchmarking
- Reproducible benchmark and plotting scripts

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
```

## Benchmark

```bash
python3 scripts/benchmark.py --build-dir build --paths 10000000 --steps 365 --runs 3
python3 scripts/plot_benchmark.py --csv results/benchmark_results.csv --out results/runtime_comparison.png
```

## Validation Gate

```bash
python3 scripts/validate_parity.py --build-dir build --paths 2000000 --steps 365 --price-tol 0.05
```

## Nsight Systems Profiling

```bash
bash scripts/profile_nsight.sh build 10000000 365
```

## Valgrind (CPU baseline)

```bash
valgrind --leak-check=full ./build/mc_cpu --paths 500000 --steps 365
```

## Resume-Ready Metrics Workflow

1. Run `benchmark.py` to generate `results/benchmark_results.csv`
2. Use average speedup from script output in your bullets
3. Attach `results/runtime_comparison.png` in portfolio/project page
4. Keep parity check output as correctness evidence

