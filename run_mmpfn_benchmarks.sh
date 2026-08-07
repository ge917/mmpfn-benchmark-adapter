#!/usr/bin/env bash
set -euo pipefail

# Everything created by this wrapper stays under the user-owned work root.
WORK_ROOT="${MMPFN_WORK_ROOT:-/mnt/hdd/zhangyg/projects/tab}"
GPU_ID="${MMPFN_GPU:-1}"
PYTHON_BIN="${MMPFN_PYTHON:-/home/debian/miniconda/envs/tabopen/bin/python}"

cd "${WORK_ROOT}/MultiModalPFN-main"

exec "${PYTHON_BIN}" -u -m mmpfn.run_benchmark_suite \
  --gpu "${GPU_ID}" \
  --data-root "${WORK_ROOT}/benchmark_data" \
  --output-dir "${WORK_ROOT}/results/mmpfn_benchmark_suite" \
  --dino-checkpoint "${WORK_ROOT}/models/dinov2_vitb14_pretrain.pth" \
  --download-multabench \
  "$@"
