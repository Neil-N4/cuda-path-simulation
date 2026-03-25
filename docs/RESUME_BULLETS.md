# Resume Bullets (Copy/Paste)

## Full-Length Version

**CUDA-Accelerated Path Simulation Engine | C++20, CUDA, Python, Nsight Systems, CMake, Valgrind**

- Built a C++/CUDA Monte Carlo pricing engine (European/Asian/Up-and-Out), processing 10M paths per run on NVIDIA T4.
- Optimized kernels with shared-memory reductions, coalesced writes, and stream-overlapped transfers; achieved 1852.66x average speedup.
- Implemented antithetic + control variate estimators; reduced CI width by 1.342x, 2.434x, and 4.854x combined.
- Added Black-Scholes/Greeks validation, CPU-GPU parity gates, stress scenarios, and performance regression thresholds.

## 114-Character Version

- Built C++/CUDA Monte Carlo engine; processed 10M paths/run on NVIDIA T4 with validated quant outputs.
- Tuned shared memory, coalesced writes, and streams; achieved 1852.66x avg speedup vs CPU baseline.
- Added antithetic + control variates; cut CI width 1.342x, 2.434x, and 4.854x for combined estimator.
- Implemented BS/Greeks checks, parity/stress gates, and perf regression thresholds with pass/fail automation.
