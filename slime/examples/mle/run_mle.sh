#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${PROJECT_ROOT}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TRITON_CACHE_DIR="${PROJECT_ROOT}/outputs/mle/triton-cache"
PYTHON=${MLE_PYTHON:-python3}
exec "${PYTHON}" -m mle.launch \
  --config "${MLE_ROUTER_CONFIG:-${SCRIPT_DIR}/router.json}" \
  --experts "${MLE_EXPERT_CONFIG:-${SCRIPT_DIR}/experts.json}" \
  --output "${MLE_OUTPUT:-${PROJECT_ROOT}/outputs/mle/services}" \
  --gpus "${MLE_GPUS:-0,1}" "$@"
