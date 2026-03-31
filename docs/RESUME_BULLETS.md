# Resume Bullets (Copy/Paste)

## Full-Length Version

**CUDA-Accelerated Path Simulation Engine | C++20, CUDA, Python, Nsight Systems, CMake, Valgrind**

- Built a C++/CUDA Monte Carlo pricing engine (European/Asian/Up-and-Out), processing 10M paths per run on NVIDIA T4.
- Replaced block reductions with warp-shuffle reductions and tuned CUDA execution; achieved 1588.91x average speedup on 10M-path runs.
- Implemented antithetic + control variate estimators; reduced CI width by 1.342x, 2.434x, and 4.854x combined.
- Added Sobol low-discrepancy sampling and mixed-precision path propagation; validated both modes with CI-overlap and sub-0.001 pricing drift.

## 114-Character Version

- Built C++/CUDA Monte Carlo engine; processed 10M paths/run on NVIDIA T4 with validated quant outputs.
- Tuned warp-shuffle reductions, memory access, and streams; achieved 1588.91x avg speedup vs CPU baseline.
- Added antithetic + control variates; cut CI width 1.342x, 2.434x, and 4.854x for combined estimator.
- Added Sobol + mixed modes, parity/stress gates, and validation tooling that rejected unsupported combinations.
