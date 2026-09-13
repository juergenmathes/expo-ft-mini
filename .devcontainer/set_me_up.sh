#!/usr/bin/env bash
# Idempotent workspace setup. Runs as postCreateCommand; safe to re-run any time (`just setup`).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

D3IL_REPO="${D3IL_REPO:-https://github.com/allenzren/d3il}"
D3IL_REF="${D3IL_REF:-139dbf9b114d0f6192e5433ebcffeb0fc17098f4}"   # keep in sync with Dockerfile ARG
TORCH_EXTRA="${TORCH_EXTRA:-cpu}"                                  # cpu | cu124 | cu128 (Blackwell/sm_120)
DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"                                # 1 -> pull avoid_m1 data (~gdown)

log() { printf '\033[1;34m[set_me_up]\033[0m %s\n' "$*"; }

# 1) D3IL fork (editable path dep of pyproject.toml). The bind mount hides the copy baked into the image.
if [ ! -f third_party/d3il/environments/d3il/setup.py ]; then
  log "cloning D3IL fork -> third_party/d3il @ ${D3IL_REF:0:8}"
  rm -rf third_party/d3il
  git clone --filter=blob:none "$D3IL_REPO" third_party/d3il
  git -C third_party/d3il checkout -q "$D3IL_REF"
else
  log "third_party/d3il present"
fi

# 2) venv: /opt/venv already holds the deps from the image; this (re)links the editable packages + project.
log "uv sync --extra ${TORCH_EXTRA} --extra dev"
if [ -f uv.lock ]; then uv sync --frozen --extra "$TORCH_EXTRA" --extra dev; else uv sync --extra "$TORCH_EXTRA" --extra dev; fi

# 3) MuJoCo GL backend: EGL needs an NVIDIA driver; otherwise fall back to OSMesa (software) for offscreen renders.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  log "GPU visible -> MUJOCO_GL=egl"
else
  log "no GPU -> MUJOCO_GL=osmesa"
  grep -q 'MUJOCO_GL=osmesa' ~/.bashrc 2>/dev/null || printf '\nexport MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa\n' >> ~/.bashrc
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
fi

# 4) data (optional; ~ seconds): DOWNLOAD_DATA=1 bash .devcontainer/set_me_up.sh
if [ "$DOWNLOAD_DATA" = "1" ]; then
  bash scripts/download_data.sh m1
fi

# 5) smoke test: builds one env, runs one 100-step episode (~5 s on CPU)
log "smoke test"
uv run d3il-flow-smoke || log "WARNING: smoke test failed - check the traceback above"

log "done. next:  just data && just train && just eval"
