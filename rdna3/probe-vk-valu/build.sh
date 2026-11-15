#!/usr/bin/env bash
# Builds the harness and every kernel's SPIR-V (each tool under the 3 GiB memory cap of env.sh), then refuses any
# SPIR-V whose [[unroll]] loop has a constant bound above 1024 (check_unroll.py). Kernel list: name and glslc defines.
set -euo pipefail
cd "$(dirname "$0")"
. ./env.sh
capped gcc -O2 -Wall -Wno-missing-field-initializers -o valu_bench valu_bench.c -lvulkan
mkdir -p spv
while read -r name defs; do
  [ -z "$name" ] && continue
  capped glslc --target-env=vulkan1.3 $defs -o "spv/$name.spv" shaders/bench.comp
done <<'K'
EMPTY -DK_EMPTY
CALIB -DK_CALIB
A_FMA32 -DK_A_FMA32
A_F16MIX -DK_A_F16MIX
A_F16DOT16 -DK_A_F16DOT16
A_F16PK -DK_A_F16PK
A_I8DOT -DK_A_I8DOT
A_I8DOTSAT -DK_A_I8DOTSAT
A_U8DOT -DK_A_U8DOT
A_I32MAD -DK_A_I32MAD
A_I24MAD -DK_A_I24MAD
B_BASE -DK_B_BASE
B_E4H_BITS -DK_B_E4H_BITS
B_E4H_EXACT -DK_B_E4H_EXACT
B_E4H_LDS -DK_B_E4H_LDS
B_Q8_HW -DK_B_Q8 -DQFN=e4m3_hw
B_Q8_SAT -DK_B_Q8 -DQFN=e4m3_sat
B_F2H -DK_B_F2H
B_F2H_SW -DK_B_F2H_SW
B_RQ_F32 -DK_B_RQ_F32
B_RQ_MAGIC -DK_B_RQ_MAGIC
B_GELU_LDS -DK_B_GELU_LDS
B_XOR16 -DK_B_XOR16
B_BPERM -DK_B_BPERM
B_W_GLOBAL -DK_B_W_GLOBAL
B_W_LDS -DK_B_W_LDS
C_F16_W4 -DK_C -DTA=float16_t -DTC=float -DNW=4 -DNV=0
C_F16_V128 -DK_C -DTA=float16_t -DTC=float -DNW=0 -DNV=128
C_F16_W4V128 -DK_C -DTA=float16_t -DTC=float -DNW=4 -DNV=128
C_I8_W4 -DK_C -DTA=int8_t -DTC=int32_t -DNW=4 -DNV=0
C_I8_W4V128 -DK_C -DTA=int8_t -DTC=int32_t -DNW=4 -DNV=128
C_F16_W4V64 -DK_C -DTA=float16_t -DTC=float -DNW=4 -DNV=64
C_F16_W4V256 -DK_C -DTA=float16_t -DTC=float -DNW=4 -DNV=256
C_F16_V64 -DK_C -DTA=float16_t -DTC=float -DNW=0 -DNV=64
C_F16_V256 -DK_C -DTA=float16_t -DTC=float -DNW=0 -DNV=256
VERIFY_Q8 -DK_VERIFY_Q8
VERIFY_E4H -DK_VERIFY_E4H
VERIFY_RQ -DK_VERIFY_RQ
K
capped python3 check_unroll.py spv/*.spv
echo built
