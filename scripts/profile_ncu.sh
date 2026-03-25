#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR=${1:-build}
PATHS=${2:-10000000}
STEPS=${3:-365}
PAYOFF=${4:-european}

mkdir -p results

ncu \
  --set full \
  --target-processes all \
  --export results/ncu_profile \
  "${BUILD_DIR}/mc_gpu" \
  --paths "${PATHS}" \
  --steps "${STEPS}" \
  --payoff "${PAYOFF}" \
  --antithetic \
  --control-variate

echo "NVIDIA Nsight Compute report written to results/ncu_profile.ncu-rep"
