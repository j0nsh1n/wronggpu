# Sourced by run.sh (needs its gpu / capped / clockstate helpers). For every timed kernel: a sizing run, then iters N
# and 2N at 672 workgroups (5376 waves = 32 per SIMD) and N at 1344 workgroups, 10 timed reps each (after the harness's
# 3 untimed warm-ups). Output: raw/time_<kernel>_<wgs>_<iters>.txt; the card's sclk is sampled during each run.
TARGET_NS=${TARGET_NS:-8000000}
med() { awk '/^rep/ {print $4}' "$1" | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }
timed() {  # timed <kernel> <wgs> <iters>
  local k=$1 w=$2 n=$3 f=raw/time_$1_$2_$3.txt sp
  ( d=/sys/class/drm/card1/device; while :; do echo "clk $(cat $d/hwmon/*/freq1_input) busy $(cat $d/gpu_busy_percent)"; sleep 0.02; done ) > raw/clk_$1_$2_$3.txt &
  sp=$!
  { echo "# kernel $k wgs $w iters $n before: $(clockstate)"; gpu run spv/$k.spv $w $n 10 $IN $((w * 1024)) $AUX -; echo "# after: $(clockstate)"; } > $f
  kill $sp; wait $sp 2>/dev/null || true
}
inputs
for k in ${KERNELS:-EMPTY CALIB A_FMA32 A_F16MIX A_F16DOT16 A_F16PK A_I8DOT A_I8DOTSAT A_U8DOT A_I32MAD B_BASE B_E4H_BITS B_E4H_EXACT B_E4H_LDS B_Q8_HW B_Q8_SAT B_F2H B_F2H_SW B_RQ_F32 B_RQ_MAGIC B_GELU_LDS B_XOR16 B_BPERM B_W_GLOBAL B_W_LDS C_F16_W4 C_I8_W4 C_F16_V64 C_F16_V128 C_F16_V256 C_F16_W4V64 C_F16_W4V128 C_F16_W4V256 C_I8_W4V128}; do
  timed $k 672 256
  m=$(med raw/time_${k}_672_256.txt)
  n=$(python3 -c "import math; m=max($m,1); print(min(1<<20, max(256, 1 << round(math.log2(256*$TARGET_NS/m)))))")
  [ "$k" = EMPTY ] && n=65536
  timed $k 672 $n; timed $k 672 $((2 * n)); timed $k 1344 $n
  echo "$k N=$n $(med raw/time_${k}_672_$n.txt) $(med raw/time_${k}_672_$((2 * n)).txt) $(med raw/time_${k}_1344_$n.txt)"
done | tee logs/time_summary.txt
