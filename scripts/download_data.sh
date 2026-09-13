#!/usr/bin/env bash
# Download DPPO's pre-processed D3IL Avoiding data (train.npz + normalization.npz) for a mode split.
#   bash scripts/download_data.sh m1     # d56_r12: desired modes 5,6 / required 1,2
#   bash scripts/download_data.sh m2     # d57_r12
#   bash scripts/download_data.sh m3     # d58_r12
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODE="${1:-m1}"
DATA_DIR="${DPPO_DATA_DIR:-data}/d3il/avoid_${MODE}"

case "$MODE" in
  m1) FOLDER="https://drive.google.com/drive/u/1/folders/1ZAPvLQwv2y4Q98UDVKXFT4fvGF5yhD_o"
      NORM="https://drive.google.com/file/d/1PubKaPabbiSdWYpGmouDhYfXp4QwNHFG/view?usp=drive_link" ;;
  m2) FOLDER="https://drive.google.com/drive/u/1/folders/1wyJi1Zbnd6JNy4WGszHBH40A0bbl-vkd"
      NORM="https://drive.google.com/file/d/1Hoohw8buhsLzXoqivMA6IzKS5Izlj07_/view?usp=drive_link" ;;
  m3) FOLDER="https://drive.google.com/drive/u/1/folders/1mNXCIPnCO_FDBlEj95InA9eWJM2XcEEj"
      NORM="https://drive.google.com/file/d/1qt7apV52C9Tflsc-A55J6uDMHzaFa1wN/view?usp=drive_link" ;;
  *) echo "unknown mode '$MODE' (m1|m2|m3)"; exit 1 ;;
esac

mkdir -p "$DATA_DIR"
if [ ! -f "$DATA_DIR/train.npz" ]; then
  echo "[data] $MODE -> $DATA_DIR"
  uv run gdown --folder "$FOLDER" -O "$DATA_DIR"
fi
if [ ! -f "$DATA_DIR/normalization.npz" ]; then
  uv run gdown --fuzzy "$NORM" -O "$DATA_DIR/normalization.npz"
fi
uv run python - "$DATA_DIR" << 'PY'
import sys, numpy as np
from pathlib import Path
d = Path(sys.argv[1]); tr = np.load(d / "train.npz"); nm = np.load(d / "normalization.npz")
print("train.npz:", {k: tr[k].shape for k in tr.files}, "| episodes:", len(tr["traj_lengths"]))
print("normalization.npz:", {k: np.round(nm[k], 4).tolist() for k in nm.files})
PY
