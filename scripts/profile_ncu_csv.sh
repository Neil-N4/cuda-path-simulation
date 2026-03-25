#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR=${1:-build}
OUT_CSV=${2:-results/ncu_metrics.csv}
PATHS=${3:-10000000}
STEPS=${4:-365}

mkdir -p results

ncu \
  --csv \
  --page raw \
  --target-processes all \
  --metrics \
sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warp_issue_stalled_memory_dependency_per_warp_active.avg,\
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.avg,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.avg \
  "${BUILD_DIR}/mc_gpu" \
  --paths "${PATHS}" \
  --steps "${STEPS}" \
  --payoff european \
  --antithetic \
  --control-variate > "${OUT_CSV}"

echo "ncu_metrics_csv=${OUT_CSV}"
