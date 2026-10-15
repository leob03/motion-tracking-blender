#!/usr/bin/env bash
set -euo pipefail
bridge_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
comfy_dir="${MOCAP_COMFY_ROOT:-/home/innovation/dev-lbringer/ComfyUI}"
export MOCAP_STUDY_ROOT="${MOCAP_STUDY_ROOT:-$bridge_dir/../comparison_study}"
export MOCAP_PYTHON="${MOCAP_PYTHON:-$MOCAP_STUDY_ROOT/.venv/bin/python}"
runtime_dir="${MOCAP_RUNTIME_DIR:-/tmp/mocap-comfy}"
mkdir -p "$runtime_dir/input" "$runtime_dir/output" "$runtime_dir/user"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
exec "$MOCAP_PYTHON" "$comfy_dir/main.py" \
  --listen 127.0.0.1 --port "${MOCAP_PORT:-8188}" \
  --disable-all-custom-nodes --whitelist-custom-nodes ComfyUI-MocapProduction \
  --disable-api-nodes --disable-auto-launch --cpu --cache-none \
  --max-upload-size 2048 \
  --input-directory "$runtime_dir/input" --output-directory "$runtime_dir/output" \
  --user-directory "$runtime_dir/user" --database-url "sqlite:///$runtime_dir/comfyui.db"
