# Performance Gate Policy

Thresholds are defined in `configs/perf_gate_thresholds.json`.

## Benchmark Thresholds

- `min_avg_speedup`: minimum required CPU/GPU speedup.
- `max_avg_gpu_ms`: maximum allowed average GPU runtime.
- `max_price_abs_diff`: maximum allowed CPU/GPU estimator mismatch.

## Convergence Thresholds

- `min_antithetic_reduction`: minimum CI-width reduction factor for antithetic vs baseline.
- `min_control_reduction`: minimum CI-width reduction factor for control variate vs baseline.
- `min_combo_reduction`: minimum CI-width reduction factor for antithetic+control vs baseline.

## Gate Command

```bash
python3 scripts/perf_gate.py \
  --benchmark-csv results/benchmark_results.csv \
  --convergence-csv results/convergence/gpu_convergence.csv \
  --thresholds configs/perf_gate_thresholds.json
```
