#!/usr/bin/env bash
# Reproduces the whole probe: build, generate inputs, run on the RX 7900 XT (1002:744c), analyse.
# Headless: no surface or swapchain is created; display variables are removed before any GPU process starts.
set -euo pipefail
cd "$(dirname "$0")"
unset DISPLAY WAYLAND_DISPLAY
export SDL_VIDEO_DRIVER=offscreen
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json   # RADV only; the harness picks 1002:744c itself
export MESA_SHADER_CACHE_DISABLE=true                                   # so the shaders are compiled (and captured) every run

gcc -O2 -Wall -Wno-missing-field-initializers -o wmma_probe wmma_probe.c -lvulkan
mkdir -p spv out disasm
for v in "f16_f32 float16_t float" "f16_f16 float16_t float16_t" "s8_s32 int8_t int32_t" "u8_u32 uint8_t uint32_t"; do
  set -- $v
  glslc --target-env=vulkan1.3 -DTA="$2" -DTC="$3" -o "spv/$1.spv" mm.comp
done
glslc --target-env=vulkan1.3 -DTA=int8_t -DTC=int32_t -DSAT -o spv/s8_s32_sat.spv mm.comp

[ -f inputs/e4_random.npz ] || nice python3 gen.py

nice ./wmma_probe props | tee out/props.txt

P=(nice timeout 60 ./wmma_probe run)
n() { python3 -c "import numpy as np; print(len(np.load('inputs/$1.npz')['kind']))"; }
for c in e4_random p1b adv specials f16_full kperm pos_shift pos_perm; do
  N=$(n "$c")
  "${P[@]}" spv/f16_f32.spv "$N" 2 4 "inputs/$c.A16.bin" "inputs/$c.B16.bin" "inputs/$c.C32.bin" "out/$c.f32" 3 disasm/f16_f32.txt
done
for c in e4_random p1b adv specials f16_full kperm; do
  N=$(n "$c")
  "${P[@]}" spv/f16_f16.spv "$N" 2 2 "inputs/$c.A16.bin" "inputs/$c.B16.bin" "inputs/$c.C16.bin" "out/$c.f16" 3 disasm/f16_f16.txt
done
"${P[@]}" spv/s8_s32.spv 4096 1 4 inputs/int_s8.A.bin inputs/int_s8.B.bin inputs/int_s8.C.bin out/int_s8.wrap 3 disasm/s8_s32.txt
"${P[@]}" spv/s8_s32_sat.spv 4096 1 4 inputs/int_s8.A.bin inputs/int_s8.B.bin inputs/int_s8.C.bin out/int_s8.sat 3 disasm/s8_s32_sat.txt
"${P[@]}" spv/u8_u32.spv 4096 1 4 inputs/int_u8.A.bin inputs/int_u8.B.bin inputs/int_u8.C.bin out/int_u8.wrap 3 disasm/u8_u32.txt

# Second, independent disassembly route: Mesa's own debug print of the compiled shader (stderr).
head -c 2048 inputs/specials.A16.bin > out/smoke.A16.bin; head -c 2048 inputs/specials.B16.bin > out/smoke.B16.bin; head -c 4096 inputs/specials.C32.bin > out/smoke.C32.bin
RADV_DEBUG=shaders "${P[@]}" spv/f16_f32.spv 4 2 4 out/smoke.A16.bin out/smoke.B16.bin out/smoke.C32.bin out/radv_debug_smoke 1 \
  > /dev/null 2> disasm/radv_debug_shaders_f16_f32.txt || true

python3 diag_terms.py | tee out/terms.txt
nice python3 pack.py
nice python3 analyse.py | tee out/analysis.txt
