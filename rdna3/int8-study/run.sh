#!/bin/sh
# INT8 study: every program runs under an 8 GB memory cap with no swap, at most 8 CPU cores, low priority.
# Usage: run.sh <WrongGPU data folder holding model/model.wgcm and model/F/model.wgcm> [render WxH]
set -eu
cd "$(dirname "$0")"
DATA="$1"; RENDER="${2:-128x72}"
SRC=../src
CAP="systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -p AllowedCPUs=0-7 -- nice env OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8"
mkdir -p results
$CAP python3 int8_study.py "$DATA/model/model.wgcm" "$SRC/data/preset-M.v1.map.json" --label "Model M (real)" \
  --json results/weights-M.json > results/weights-M.md
if [ -f "$DATA/model/F/model.wgcm" ]; then
  $CAP python3 int8_study.py "$DATA/model/F/model.wgcm" "$SRC/data/preset-F.v1.map.json" --label "RR preset F (real)" \
    --json results/weights-F.json > results/weights-F.md
fi
$CAP python3 m_e2e.py "$DATA/model/model.wgcm" --src "$SRC" --render "$RENDER" \
  --cases pan,disocclusion,checker,exposure_bright --json results/e2e-M.json
$CAP python3 report.py results > results/e2e-M.md
