#!/usr/bin/env bash
# Shader (VALU) throughput probe on the RX 7900 XT (1002:744c, RADV). Headless: no surface, swapchain or window.
# Every process runs under env.sh's `capped` (3 GiB memory, no swap, nice) and `timeout`; GPU runs also hold the shared
# benchmark lock and wait while the other benchmark agent's process is running.
#   ./run.sh build | info | disasm | verify | time | all
set -euo pipefail
cd "$(dirname "$0")"
. ./env.sh
mkdir -p raw disasm logs

# Wait (up to 10 min) while a process of the int8 matrix probe runs. The pattern is assembled from two parts so this
# script's own command line never matches it.
wait_other() {
  local p1=probe-vk- p2=int8 n
  for n in $(seq 120); do
    if ! ps -eo pid=,args= | P="$p1$p2" awk 'index($0, ENVIRON["P"]) {f=1} END {exit !f}'; then return 0; fi
    echo "waiting for the int8 probe ($n)" >&2; sleep 5
  done
  echo "the int8 probe is still running; giving up" >&2; exit 3
}
clockstate() {
  local d=/sys/class/drm/card1/device
  echo "sclk=$(grep '\*' $d/pp_dpm_sclk | awk '{print $2}') mclk=$(grep '\*' $d/pp_dpm_mclk | awk '{print $2}') busy=$(cat $d/gpu_busy_percent)%"
}
# gpu <args to valu_bench>: one locked, capped, time-limited GPU process.
gpu() { wait_other; flock "$LOCK" bash -c '. ./env.sh; capped timeout 60 ./valu_bench "$@"' _ "$@"; }

IN=raw/in_rand.bin AUX=raw/aux_rand.bin
inputs() {
  [ -s $IN ] && [ -s $AUX ] || capped python3 -c "
import numpy as np; r = np.random.default_rng(1234)
r.integers(0, 2**32, 65536, dtype=np.uint32).tofile('$IN'); r.integers(0, 2**32, 65536, dtype=np.uint32).tofile('$AUX')"
}

do_info() { gpu info | tee logs/info.txt; }

do_disasm() {
  inputs
  for s in spv/*.spv; do
    k=$(basename "$s" .spv)
    case $k in VERIFY_*) continue;; esac
    gpu run "$s" 1 1 1 $IN 1024 $AUX - "disasm/$k.txt" > "disasm/$k.stats"
    echo "$k $(capped python3 disasm_loop.py disasm/$k.txt | head -2 | tr '\n' ' ')"
  done | tee logs/disasm_summary.txt
}

case "${1:-all}" in
  build) ./build.sh ;;
  info) do_info ;;
  disasm) do_disasm ;;
  verify) . ./verify.sh ;;
  time) . ./time.sh; capped python3 analyze.py > logs/analysis.md ;;
  all) ./build.sh; do_info; do_disasm; . ./verify.sh; . ./time.sh; capped python3 analyze.py > logs/analysis.md ;;
esac
