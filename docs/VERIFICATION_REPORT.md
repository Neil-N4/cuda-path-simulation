# Verification Report (One Page)

## Project

CUDA-Accelerated Path Simulation Engine  
C++20, CUDA, Python, Nsight Systems/Compute, CMake

## Objective

Build a high-throughput Monte Carlo pricing system that is both fast and numerically trustworthy under real workloads.

## Test Environment

- Platform: Google Colab
- GPU: NVIDIA Tesla T4 (CUDA 12.8 toolchain)
- Workload: 10,000,000 paths, 365 steps, European payoff
- Estimators: antithetic variates + control variate

## Performance Results

- CPU runtime (avg, 3 runs): **67,850.168 ms**
- GPU runtime (avg, 3 runs): **37.650 ms**
- Speedup (avg): **1852.66x**

## Numerical Validation

- CPU/GPU CV estimator parity:
  - `abs_diff = 0.002882`
  - CI overlap: **passed**
- Black-Scholes consistency (European):
  - CPU abs error: `6.52e-4`
  - GPU abs error: `3.75e-4`
- Greeks consistency (European, pathwise vs BS):
  - Delta abs error (GPU): `9e-6`
  - Vega abs error (GPU): `4.368e-3`

## Variance-Reduction Impact (CI Width vs Baseline)

- Antithetic: **1.342x**
- Control variate: **2.434x**
- Antithetic + control: **4.854x**

## Robustness Validation

Stress suite status: **passed**

Scenarios validated:

- European nominal
- European high volatility
- European low rate
- Asian payoff
- Up-and-out barrier payoff

## Regression Protection

Automated gate (`scripts/perf_gate.py`) enforces:

- minimum speedup
- maximum GPU runtime
- maximum CPU/GPU price deviation
- minimum CI-width reduction factors

Latest gate status: **perf_gate=passed**

## Conclusion

The engine meets performance, numerical-consistency, and robustness objectives with reproducible command-line workflows and explicit pass/fail thresholds.
