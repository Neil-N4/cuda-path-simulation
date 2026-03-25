# Nsight Compute Report Template

Use this template after running:

```bash
make nsight-compute
```

## Environment

- GPU:
- CUDA toolkit:
- Driver:
- Paths / Steps:

## Kernel Summary

- Primary kernel:
- Avg duration:
- Achieved occupancy:
- SM throughput:
- DRAM throughput:
- Warp stall breakdown (top 3):

## Tuning Changes

1. Change:
   - Before:
   - After:
   - Impact:
2. Change:
   - Before:
   - After:
   - Impact:

## Final Metrics

- Runtime (ms):
- Throughput (paths/s):
- CPU vs GPU speedup:
- CI width reduction factors:
  - antithetic:
  - control:
  - antithetic+control:
