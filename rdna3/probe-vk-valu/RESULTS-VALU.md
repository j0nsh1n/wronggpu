# RDNA 3 shader (VALU) throughput, reported apart from matrix (WMMA) throughput

Card: AMD Radeon RX 7900 XT (RADV NAVI31, 1002:744c), Mesa 27.0.0-devel. 84 CUs, 168 SIMDs. The integrated GPU
1002:13c0 is never opened: the harness only accepts 1002:744c.

Labels: MEASURED = observed on this card in this probe; INFERRED = reasoning from those observations.

## 0. The 16 GB allocation of the first attempt, and the fix

**Cause (MEASURED and INFERRED).** The overlap kernel `C_F16_V128` is built with `-DNW=0 -DNV=128`. Its inner loop was
`[[unroll]] for (uint v = 0u; v < NV / NW; ++v)`. glslang folds the integer `128 / 0` to 2147483647 instead of
rejecting it. The old `spv/C_F16_V128.spv` holds `OpULessThan ... %uint_2147483647` in a loop marked `Unroll`
(MEASURED, SPIR-V line 298; the old file is kept in `prev-attempt/`). RADV honours `Unroll` on any loop with a known
trip count, so the driver's shader compiler, running inside `valu_bench` during `vkCreateComputePipelines`, set about
copying the loop body 2^31 times. INFERRED: that copying is what grew to 16 GB. The disassembly pass died on exactly
this kernel: `prev-attempt/disasm/C_F16_V128.stats` is empty, so the process never got past pipeline creation
(MEASURED). It does not matter that the loop around it (`w < 0`) never runs, because the compiler still processes the
inner loop. I did not re-run the bad shader to reproduce the growth.

Checked and found sound: buffer sizes are bounded (the harness refuses more than 900 MiB in total; the output is
`workgroups x 1 KiB`), the disassembly capture frees each representation it reads, and the timed loop allocates
nothing.

**Fix.**
1. `shaders/bench.comp`: `NV / NW` is only formed under `#if NW > 0`. `#error` rejects `NV > 512`, negative values and
   an `NV` that is not a multiple of `NW`. The nested `#if` is needed because glslang's preprocessor does not
   short-circuit `&&`.
2. `check_unroll.py`, run by `build.sh` on every SPIR-V file: the build fails if any `Unroll` loop has a constant
   bound above 1024. It fails on the old `C_F16_V128.spv` (bound 2147483647) and passes all 38 new kernels (MEASURED).
3. `env.sh`'s `capped`: every process (gcc, glslc, Python, the benchmark) runs under
   `systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0`, under `nice`, and GPU runs also under
   `timeout 60`. Cap test (MEASURED): a 200 MB allocation ran; a 4 GB allocation was killed (exit 137). The fixed
   `C_F16_V128` peaked at about 56 MB of process memory, and its loop body is 128 `v_fma_f32` (MEASURED).

## 1. Method

- **Headless.** `DISPLAY` and `WAYLAND_DISPLAY` are unset and the ICD is RADV only. There is no surface, swapchain or
  WSI extension. The buffers take under 40 MiB of VRAM (the largest are the verification runs at 16 MiB in and 16 MiB
  out).
- **Timing.** The harness records GPU timestamps around one `vkCmdDispatch` per rep: 3 untimed warm-ups, then 10 timed
  reps, with a 10 ns tick. Each GPU process holds `flock ../gpu-bench.lock`. Before it starts, the script waits while
  any process of the int8 matrix probe is running; none ran during these measurements.
- **Dispatch.** 672 workgroups of 256 invocations = 5376 waves of 32 = 32 waves per SIMD. Required subgroup size 32,
  full subgroups. Occupancy from the pipeline statistics: 16 subgroups per SIMD for every kernel except `B_W_LDS`,
  which has 8 because it uses 32 KiB of LDS (MEASURED).
- **Loop body.** Confirmed from ACO's "Final Assembly" (`disasm/`, `disasm_loop.py`, `logs/disasm_summary.txt`). The
  instructions are listed per item below.
- **Linearity (MEASURED, `logs/analysis.md`).** Each kernel ran at N and 2N iterations, with N chosen so that a
  dispatch takes 3-7 ms, and at N iterations with 1344 workgroups. t(2N)/t(N) = 1.92-1.99 and t(1344 WG)/t(672 WG) =
  1.91-1.98 for every kernel except the empty loop's dispatch check (1.86). The small shortfall from 2.00 is the fixed
  launch cost.
- **Per-iteration cost.** fixed = 2 med t(N) - med t(2N). For each of the 10 reps at 2N, cost = (t(2N) - fixed) / 2N.
  Tables give the median and the min-max over the 10 reps.
- **Cycles.** Results are in SIMD-cycles per wave, calibrated with `CALIB`: 16 independent chains of VOP3
  `v_fma_f32`, 128 per wave-iteration, assumed to issue at 1 per SIMD per cycle. That gives f_eff = 2.682 GHz
  (MEASURED). The sysfs `hwmon freq1_input`, sampled every 20 ms while `gpu_busy_percent` >= 90, read a median of
  2.506 GHz with a range of 1.513-2.914 GHz over 512 samples (MEASURED). The samples include clock ramps between
  dispatches. INFERRED: the clock during the timed dispatches sat near the top of that range, consistent with the
  1-FMA-per-cycle assumption to within about 7 %. Whole-card times use the measured ns directly.
- **Baselines.** The empty loop (`EMPTY`: 1 VALU and 3 scalar/branch instructions) costs 4.07 cycles per
  wave-iteration (MEASURED). It is shown subtracted in table A. INFERRED: in VALU-bound kernels the scalar loop
  control issues alongside the vector work, so the unsubtracted number is the better estimate. `CALIB` comes out at
  128.0 cycles unsubtracted, matching its 128 FMAs. For B, the glue cost is the kernel minus `B_BASE`, which has the
  same harness: a fresh input `x + t` per value, plus an xor fold.
- **Desktop.** The card also drives the user's desktop, which may have been in use. Rep spread at 2N (max-min over
  median) was 2.4-16 % for most kernels and 24.7 % for `B_W_GLOBAL` (MEASURED). INFERRED: the spread comes from clock
  changes and desktop work, so medians are reported.

## 2. Correctness of the glue code (MEASURED, `verify.py`, `logs/verify.txt`)

The CPU references are written from the definitions (nearest representable value, ties to even, the saturation rules
of `wg_e4m3_hw` / `wg_e4m3_sat`), not from the shader's bit tricks. There are 4,194,304 inputs per conversion: every
bit pattern, values from 2^-27 to 2^13, exact E4M3 midpoints and their FP32 neighbours (one unit below and above),
and the special values.

| GPU function (GLSL port) | Mismatches |
|---|---|
| `e4m3_hw` (port of `wg_e4m3_hw`: RNE, \|x\| > 464 or inf -> 0x7f\|s, NaN -> 0xff) | 0 of 4,194,304 |
| `e4m3_sat` (port of `wg_e4m3_sat`: \|x\| >= 448 -> 0x7e\|s) | 0 of 4,194,304 |
| E4M3 -> F16, bit form (value x 2^-8) | 0 of 254 finite codes |
| E4M3 -> F16, exact (bit form, then `v_pk_mul_f16` by 256) | 0 of 254 finite codes |
| int32 -> int8 requantize, textbook form (cvt, mul, rndne, cvt, clamp) | 0 of 4,194,304 |
| int32 -> int8 requantize, magic form (`fma(a, s, 1.5*2^23)`, med3 clamp, low byte) | 0 of 4,194,304 |

The two NaN codes 0x7f and 0xff become finite F16 values in both widening forms (0x3f80 / 0xbf80 in the bit form,
480 / -480 in the exact form), not F16 NaN. The check can fail: with the reference's tie rule flipped, it reported
172,784 mismatches for `e4m3_hw`; with `floor` in place of `rint`, 414,977 for the requantize.

## 3. A: dot-product alternatives to the matrix instruction (MEASURED)

Every kernel does 128 products per lane per iteration, which is 4096 per wave: one 16x16x16 block. "Cycles" are
SIMD-cycles for one wave's 4096 products, which is also the SIMD's throughput cost. "Whole card" is the time per 4096
products with all 168 SIMDs busy.

| Path | Loop body (ACO) | Cycles / 4096 med [min-max] | Minus empty loop | Whole card ns / 4096 |
|---|---|---|---|---|
| FP32 FMA chain | 128 `v_fmac_f32` (VOP2; ACO did not pair them into VOPD) | 137.0 [131.6-140.8] | 133.0 | 0.304 |
| FP16 pairs, FP32 accumulate | 128 `v_fma_mix_f32`; **no `v_dot2_f32_f16` emitted** | 131.8 [127.1-136.4] | 127.7 | 0.292 |
| FP16 `dot(f16vec2)`, FP16 accumulate | 64 `v_fma_f16` + 64 `v_fmac_f16`; no `v_dot2_f16_f16` | 159.5 [150.6-167.6] | 155.4 | 0.354 |
| FP16 packed FMA, FP16 accumulate | 64 `v_pk_fmac_f16` | 74.9 [72.7-84.7] | 70.9 | 0.166 |
| **packed int8 dot, s8 x s8 (`dotPacked4x8EXT`)** | 32 `v_dot4_i32_iu8 ... neg_lo:[1,1,0]` | **62.6 [60.1-64.9]** | 58.6 | **0.139** |
| packed int8 dot, saturating | 32 `v_dot4_i32_iu8 ... neg_lo:[1,1,0] clamp` | 63.2 [60.2-65.5] | 59.1 | 0.140 |
| packed uint8 dot | 32 `v_dot4_u32_u8` | 62.4 [59.8-64.9] | 58.3 | 0.139 |
| plain int32 multiply-add | 128 `v_mul_lo_u32` + 64 `v_add3_u32` (+8 moves) | 584.0 [544.5-609.3] | 579.9 | 1.296 |
| *matrix: `v_wmma_f32_16x16x16_f16`* | 4 WMMA per iteration, 4 accumulators | *33.1 [31.8-34.5]* per WMMA | 32.1 | *0.074* |
| *matrix: `v_wmma_i32_16x16x16_iu8`* | 4 WMMA per iteration, 4 accumulators | *32.1 [31.1-32.9]* per WMMA | 31.1 | *0.071* |

- The packed int8 path emits the dedicated `v_dot4_i32_iu8` (signed via `neg_lo`; the saturating form adds `clamp`).
  INFERRED from 32 instructions in 62.6 cycles: it issues at half rate (2 cycles per wave32 instruction).
- The int32 multiply-add is limited by `v_mul_lo_u32` at quarter rate: 128 x 4 + 64 = 576 against 584 measured.
  A variant with byte-sized operands (`A_I24MAD`) compiled to the same instructions (no `v_mad_i32_i24`), so it was
  not timed.
- Ratios (INFERRED from the medians): the matrix instruction is about 2x the best vector path (int8 dot4), 2.3x packed
  FP16 with FP16 accumulation, 4x FP32 FMA and FP16-mix, and 18x int32 multiply-add.

## 4. B: glue around a matrix multiply (MEASURED)

One "step" is 8 values per lane, which is 256 per wave: one 16x16 tile, so the numbers drop straight into the budget.
"Slots" are VALU issue slots in the loop body (a VOPD pair counts as one) per value, including the harness. "Glue" =
kernel - `B_BASE`, only where the harness is the same.

| Item | Loop body per 32 values (ACO) | Cycles / step med [min-max] | Glue cycles / step | Slots / value (glue) | Whole card ps / value |
|---|---|---|---|---|---|
| Harness only (`B_BASE`) | `v_fma_f32`/`v_add_f32` + `v_xor3_b32` | 14.4 [14.2-14.7] | - | 1.62 | 0.125 |
| E4M3 -> F16, bit form (x 2^-8) | `v_and`/`v_lshl`/`v_and_or`/`v_lshl_or`/`v_bfe` | 35.2 [32.4-36.3] | - | 4.25 (own lighter harness) | 0.305 |
| E4M3 -> F16, exact (bit form + `v_pk_mul_f16`) | as above + 16 `v_pk_mul_f16` | 40.0 [38.7-41.2] | - | 4.75 | 0.347 |
| E4M3 -> F16, table in LDS | 32 `ds_load_u16_d16(_hi)` + address math | 44.8 [43.6-46.5] | - | 2.50 VALU + 1 LDS load | 0.389 |
| **FP32 -> E4M3, RNE + saturation, `e4m3_hw`** | ~16 VALU per value (`v_rndne`, `v_cvt_u32_f32`, `v_min`, compares, `v_cndmask`) | **147.4 [132.0-155.3]** | **133.0** | 17.47 (15.84) | 1.278 |
| FP32 -> E4M3, `e4m3_sat` | same count | 149.4 [140.0-158.6] | 135.0 | 17.47 (15.84) | 1.295 |
| FP32 -> FP16, hardware | 16 `v_fma_mixlo_f16` + 16 `v_fma_mixhi_f16` (**the harness add fuses into the conversion**) | 12.4 [12.1-13.2] | -2.0, i.e. free after an add | 1.41 (-0.22) | 0.108 |
| FP32 -> FP16, `wg_f2h` software form | ~15 VALU per value | 142.2 [130.6-150.5] | 127.8 | 16.78 (15.16) | 1.233 |
| int32 -> int8 requantize, textbook | `v_cvt_f32_i32`, `v_mul`, `v_rndne`, `v_cvt_i32_f32`, `v_maxmin_i32`, `v_perm` | 58.0 [56.3-59.7] | 43.6 | 6.41 (4.78) | 0.503 |
| **int32 -> int8 requantize, magic** | `v_cvt_f32_i32`, `v_fmaak`, `v_maxmin_f32`, `v_perm` | **41.7 [41.2-42.9]** | **27.3** | 4.44 (2.81) | 0.362 |
| Table-lookup activation (GELU, 2048-slot table in LDS) | `v_fma_mix` to F16 bits, `v_bfe`, `ds_load_b64`, `v_fma` | 55.3 [54.3-57.2] | 40.9 | 5.34 (3.72) + 1 LDS load | 0.479 |
| **Half-wave exchange, `subgroupShuffleXor(v, 16)`** | 32 `v_permlanex16_b32` (VALU, no LDS) | **14.7 [14.3-15.4]** | (no harness) | 1.00 | 0.128 |
| Half-wave exchange by run-time index (`subgroupShuffle(v, lane^16)`) | 32 `ds_bpermute_b32` | 46.5 [45.0-48.6] | (no harness) | 0.03 VALU + 1 LDS op | 0.403 |
| Weights, 16 B per lane per step, storage buffer (32 KiB, cache-resident) | `buffer_load_b128` | 8.5 [7.9-10.1] | (harness: address + xor fold) | 0.62 | 0.074 |
| Weights, 16 B per lane per step, shared memory (LDS) | 2 `ds_load_b64` | 15.7 [15.1-16.3] | (same harness) | 0.62 | 0.136 |

Notes:
- The E4M3 widening kernels feed 8 codes per lane per step from 2 dwords (`p + i` per 4 codes), a lighter harness
  than `B_BASE`, so their totals are shown without subtracting it. INFERRED: the widening costs about 35 cycles per 8
  codes in bit form and about 40 exact. The LDS table is slower than the bit form here.
- The quantizer to E4M3 is the expensive item: about 16.6 cycles per value per wave, which is 4x a whole F16 WMMA for
  one tile's 8 values per lane. The `wg_f2h` software form costs the same, whereas the hardware FP32->FP16 conversion
  is free when it follows an add or FMA.
- Weights from the storage buffer were faster than from LDS in this test. INFERRED: the 32 KiB block stays in the
  vector cache, and the LDS kernel runs at half occupancy (8 waves per SIMD, limited by LDS). With weights that do not
  fit in cache, the order could flip; not measured.
- The GELU table and the LDS E4M3 table hold random words: timing only, not values.

## 5. C: overlap of vector work with a chain of matrix multiplies (MEASURED)

Each kernel does 4 WMMA per iteration into 4 accumulators chained across iterations, interleaved with NV `v_fma_f32`
(16 independent chains). The disassembly shows the WMMA and FMA instructions interleaved in one loop body
(`disasm/C_*`).

| Kernel | Cycles / wave-iteration med [min-max] | WMMA alone + VALU alone | max(WMMA, VALU) |
|---|---|---|---|
| F16 WMMA x4 + 64 FMA | 200.4 [193.8-211.1] | 195.4 | 132.5 |
| F16 WMMA x4 + 128 FMA | 275.7 [258.8-280.9] | 258.9 | 132.5 |
| F16 WMMA x4 + 256 FMA | 392.9 [380.1-408.9] | 380.6 | 248.1 |
| I8 WMMA x4 + 128 FMA | 265.8 [253.9-276.4] | 254.7 | 128.3 |

**The times add; they do not overlap.** Every mixed kernel takes as long as the sum of its parts or 3-7 % longer.
INFERRED: on RDNA 3 the WMMA runs on the same SIMD's vector unit for about 32 cycles, so vector glue in any wave on
that SIMD queues behind it. The glue must be budgeted on top of the matrix time, not hidden under it.

## 6. Worked budget for one 16x16x16 layer step (INFERRED)

Matrix numbers: `probe-vk-int8/RESULTS-INT8.md` does not exist yet, so its numbers are **pending**. This budget uses
this probe's own WMMA throughput from section 3 (`C_F16_W4`, `C_I8_W4`, MEASURED here). Replace those two numbers
when the int8 probe reports.

The unit is SIMD-cycles for one wave doing one step: one 16x16x16 multiply, producing one 16x16 tile, which is 8
outputs per lane. All components are throughput costs that add, per section 5. ns = cycles / 2.682 GHz for one wave
on one SIMD. Divide by 168 for the whole card.

Assumptions: both operands are fresh for each multiply (A and B each 16 values per lane). The output tile is
quantized and exchanged once per step, i.e. K = 16. For a deeper K the widening scales with K/16 while the output
work stays once per tile.

| Route | Matrix | Glue | Total cycles (ns) | Matrix share |
|---|---|---|---|---|
| (1) FP16: widen A + B codes (32 codes per lane, bit form, 4 x 35.2) + F16 WMMA + quantize 8 outputs to E4M3 (133.0 glue) + half-wave exchange (14.7) | 33.1 | 140.8 + 133.0 + 14.7 = 288.5 | 321.6 (120 ns) | 10 % |
| (2) INT8: int8 operands as-is + I8 WMMA + requantize 8 outputs (magic form, 27.3 glue) | 32.1 | 27.3 | 59.4 (22 ns) | 54 % |
| (2') INT8 with the same half-wave exchange as (1) (if its output layout needs it, not measured for this route) | 32.1 | 42.0 | 74.1 (28 ns) | 43 % |
| (3) No matrix instruction: int8 `v_dot4_i32_iu8` for the 4096 products (62.6) + requantize (27.3) | - | 62.6 + 27.3 = 89.9 | 89.9 (34 ns) | - |

Reading (INFERRED):
- In the FP16 route the matrix instruction is about a tenth of the step. Software E4M3 quantization (133 cycles, 4x
  the WMMA) and widening of the 8-bit codes (141 cycles) dominate. Widening only A, with B kept as F16 weights, would
  save 70 cycles; the quantizer cannot be avoided in this route.
- The INT8 route is about 5x cheaper per step than the FP16 route, and its matrix instruction is about half the step.
- Without the matrix instruction, the best vector path (int8 dot4) plus requantize costs about 1.5x the INT8 WMMA
  route, and is still 3.6x cheaper than the FP16 route with WMMA, because that route's cost is glue, not
  multiplication.
- These numbers depend on route (1)'s E4M3 output being bit-exact with the RDNA 4 path. Route (2) changes the
  arithmetic (int8 values, int32 accumulation), so it is a different numeric format, not a faster way to get the
  same bits.

## 7. Files

- `valu_bench.c`: the harness. The first attempt's harness was sound and is unchanged.
- `shaders/bench.comp`: all kernels, with the fix in `K_C`.
- `build.sh`, `env.sh` (memory cap, headless environment), `check_unroll.py` (build-time unroll-bound check),
  `run.sh` (`./run.sh all`, or `build | info | disasm | verify | time`), `time.sh`, `verify.sh`, `verify.py`,
  `disasm_loop.py`, `analyze.py`.
- `raw/time_<kernel>_<workgroups>_<iters>.txt`: per-rep GPU ns, with the sysfs clock state before and after.
  `raw/clk_*.txt`: clock samples. `raw/verify_*`: verification inputs and outputs.
- `disasm/<kernel>.txt` and `.stats`: ACO assembly and pipeline statistics. `logs/`: summaries,
  `analysis.md` / `analysis.json`, `verify.txt`.
- `prev-attempt/`: the first attempt's `disasm/` and `raw/`, kept for the record of the failure, not used here.
