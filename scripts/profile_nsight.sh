#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR=${1:-build}
PATHS=${2:-10000000}
STEPS=${3:-365}

mkdir -p results

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --output=results/nsight_gpu_profile \
  "${BUILD_DIR}/mc_gpu" --paths "${PATHS}" --steps "${STEPS}"

echo "Nsight profile written to results/nsight_gpu_profile.qdrep"
