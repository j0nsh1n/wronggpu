# Lane 3: what "correct" means numerically, and whether RDNA 3 can meet it without FP8 matrix instructions

Source: upstream main 994f304 (paths relative to the repo root). Labels: MEASURED-IN-REPO, READ, INFERRED, UNKNOWN.

## Summary

1. The repo has three separate correctness contracts, and none of them is "bit-exact against another GPU":
   (a) every D3D12 port gives the same bytes as its HIP kernel on the same RDNA 4 card; (b) kernels agree with the
   NumPy/CPU reference within the spec 09 tolerances (Model M/L), the spec 49 tolerances (Model K) or bit for bit in
   some RR stages; (c) the equivalence protocol against the original (DLSS) model, MATCHES / CLOSE / DIFFERENT. READ.
2. The reference defines matrix sums as exact (float64), rounded only where the spec stores a value. The RDNA 4 FP8
   matrix instruction does not do that: it is not "exact sum rounded once to f32"; the best-fitting measured model is
   an f32 rounding after C and after every group of 8 products. The RDNA 4 F16 matrix instruction is also not
   round-once of the exact sum. MEASURED-IN-REPO.
3. So "bit-exact with RDNA 4 output" means reproducing an undocumented internal rounding of gfx12's FP8 WMMA on
   gfx11's F16 WMMA. Every E4M3 value converts exactly to a normal FP16 number (no FP16 subnormals arise), so the
   operands are not the problem. Whether gfx11's F16 WMMA rounds its partial sums the same way gfx12's FP8 WMMA
   does is UNKNOWN, and nothing in the repo measures it. That is the single test that decides option (a).
4. The repo already has two ways to compute the spec's *exact* sum without FP8 matrix hardware: an integer path on
   `v_wmma_i32_16x16x16_iu8` (RR step "I", digit planes, exact by construction) and a binary64 VALU step. RDNA 3 has
   the IU8 instruction. An exact-sum implementation matches the CPU reference bit for bit, but matches RDNA 4's default
   output only where RDNA 4's own steps happen to be exact (most values on real-like data, not all). READ + INFERRED.
5. A shader-side F32 -> E4M3 round-to-nearest-even quantizer already exists in HLSL (`wg_e4m3_hw`, `wg_e4m3_sat`,
   `wg_pack_q8*` software forms), documented as bit-identical to the hardware conversion including NaN and overflow.
   RDNA 3 can use it today; cost about 8-10 VALU per value instead of half an instruction. READ + INFERRED.

## 1. The correctness contract

### 1.1 What is compared with what

| Contract | Compared against | Rule | Tool | Where it runs | Label |
|---|---|---|---|---|---|
| D3D12 port vs HIP kernel (Model M, K, FG) | the HIP build of the same kernel, same card | every named buffer, every frame, byte for byte | `clean/dx12/compare.py:1-6`, `clean/k/dx12/k_port_check.py`, `clean/fg/runner/fg_d3d12_check.py` | RDNA 4 GPU, both backends on one device | READ (`clean/dx12/README.md:8-9`, `clean/dx12/compare.py:1-6`) |
| Kernel vs CPU reference, Model M Swin stages | `clean/reference/swin.py` (float64) | spec 09: no non-finite, <= 2 % of values more than one FP8 step off, mean abs error <= 1 % of reference RMS; stricter: < 5 % differ per case, < 1 % overall | `clean/tests/test_kernels_emu.py:97-127` | CPU lane emulator (`clean/kernels/emu`), not the GPU | READ |
| Kernel vs reference, Model M enc0 | `clean/reference/encoder.py` | spec 09 plus >= 99 % bit-identical features; motion surfaces <= 1 FP16 step, >= 99.5 % identical | `clean/tests/test_encoder_emu.py:61-64` | emulator | READ |
| GPU self-test, Model M | reference `q8`, E4M3 decoding | FP8 conversions bit-exact; WMMA register layout by one-hot operands; GELU within 1 FP8 step | `clean/runner/selftest.py:6-22` | RDNA 4 GPU | READ |
| Model K | `clean/k/reference` | spec 49: mean abs error <= 2.5e-3, max <= 3e-2 of RMS (measured 4e-5 / 3e-3 on the emulator); I/O stages bit for bit at 2560x1440 on the GPU | `clean/k/README.md:228-246` | emulator and RX 9070 XT | READ / MEASURED-IN-REPO ("all pass", 2026-10-08) |
| Model L | `clean/l/reference/oracle.py` | named FP8-op model ("exact" or "trunc13") recorded with every run | `clean/l/runner/l_c32_check.py`, `l_q7_negative_control.py` | GPU | READ (`clean/l/README.md:9-14`) |
| RR preset F stages | `clean/rr_f/tests/vectors` (manifest per output) | bit-exact where `Z` is supplied, else per-output tolerance | `clean/rr_f/gpu/run_all.py` | RX 9070 XT | READ (`clean/rr_f/README.md:30,58`; `clean/rr_f/gpu/README.md:66-81`) |
| Whole model vs the original | the original DLSS model on the spec team's machine | MATCHES: 100 % equal, or >= 99.99 % with the rest 1 step off and no bias; CLOSE: >= 99 % equal, every diff <= 2 steps, mean signed diff within +-0.01 steps; else DIFFERENT | `clean/equiv/compare.py`, `model_m.py`, `probes.py` | spec team's machine | READ (`clean/EQUIVALENCE-PROTOCOL.md:8-12,43-45,139-155`) |

Notes:
- `clean/dx12/parity_gate.py` is a timing gate (native vs HIP frame time), not a numerical one. READ
  (`clean/dx12/parity_gate.py:2-6,46-51`).
- The emulator sums each WMMA in double and rounds once to f32 (`clean/kernels/emu/wg_emu.cpp:109-113`). It is not a
  model of the RDNA 4 hardware: emulator runs check layout, addressing and rounding order, not hardware accumulation.
  The K README says so (`clean/k/README.md:230-233`). READ.
- "Equal" counts +0 and -0 as equal for FP8 codes (`clean/EQUIVALENCE-PROTOCOL.md:43-45`). READ.
- RR modes: "exact" means exact against WrongGPU's own spec and reference, "not a claim about the original"
  (`clean/rr_f/runtime/README.md:85-88`). The Exact mode's body is `stageex.*` = step E with an exact Z sum
  (`clean/rr_f/runtime/tools/build.py:141`, `clean/rr_f/gpu/stage.py:422-433`), so its matrix steps still run
  on the FP8 WMMA and inherit its rounding. Accurate and Fast relax the head and body further
  (`clean/rr_f/runtime/README.md:79-83`). READ.

### 1.2 Per model: where FP8 / FP16 matrices are used

| Model | Weight format in the converted file | Matrix instruction (RDNA 4) | Label |
|---|---|---|---|
| M | E4M3, quantized at install time from the DLSS file's FP16 (`data/preset-M.v1.map.json:9` element `f16le`; 56 tensors with op `fp8`, e.g. `:113`) | FP8 x FP8 -> FP32 (`wg_wmma_fp8`), plus F16 x F16 -> FP32 for attention P V (`clean/kernels/swin.hip:734`) | READ |
| L | E4M3 from FP16 (`data/preset-L.v1.map.json`, 68 `fp8` tensors) | as M | READ |
| K | FP16 (`f16`, `f16_scaled`; no `fp8` op in `data/preset-K.v1.map.json`) | F16 -> FP32 (`clean/k/README.md:14,255`); opt-in F16 -> F16 (`:287-293`) | READ |
| FG | FP16 (`data/fg.v1.map.json`, 84 `f16` tensors) | F16 -> FP32, tier B (`clean/fg/README.md:28`) | READ |
| RR preset F | E4M3 codes in the file (`clean/converter/README.md:95`, op `e4m3` = Q8 of spec 51.1) | FP8 -> FP32 (step E), IU8 (step I) | READ |

Model K and FG never touch FP8. On RDNA 3 their question is only whether gfx11's F16 WMMA rounds like gfx12's (section
4.4). INFERRED.

### 1.3 What a new GPU family would have to pass (INFERRED: the repo states no cross-GPU rule)

The existing gates cannot run unchanged on RDNA 3:
- The HIP kernels are built for gfx1201/gfx1200 with gfx12-only builtins (`clean/kernels/wg_platform.h:4,47-50`), so
  the "port == HIP" byte check has no HIP side on a gfx11 card. READ + INFERRED.
- The runtime's create-time probe runs one FP16 and one FP8 WMMA through the driver channel with the gfx12 wave32
  layout and refuses on a mismatch (`clean/runtime/wmma_probe.h:1-18`). On gfx11 the FP8 half cannot run and the
  F16 operand layout differs (A/B are 16 halves per lane, duplicated across the two half-waves, per
  https://gpuopen.com/learn/wmma_on_rdna3/; RDNA 4 removed that duplication, per
  https://gpuopen.com/learn/using_matrix_core_amd_rdna4/). INFERRED.

So a gfx11 path needs a contract chosen by the maintainer. Three possible levels, strictest first:
1. Byte-identical to RDNA 4's output on the same inputs (recorded on an RX 9070 XT, compared with `compare.py`-style
   byte checks). Needs section 4's conditions.
2. Byte-identical to the CPU reference wherever the spec defines exact sums (the RR "exact" sense), with RDNA 4
   then judged by the same rule. Reachable with exact-sum arithmetic (section 6, options e/f).
3. The existing tolerance rules: spec 09 / spec 49 / RR manifest against the reference, and MATCHES or CLOSE in
   `clean/equiv/compare.py` against RDNA 4's own stage dumps (`clean/equiv/model_m.py`). Any of the options
   in section 6 except (b) and (c) likely meets this. This is a product decision. INFERRED.

## 2. The arithmetic model of one matrix multiply-accumulate

### 2.1 What the reference assumes (READ)

- Model M / L Swin: "Everything is computed in float64; the only roundings are the ones the spec writes down (q8,
  f16)" (`clean/reference/swin.py:4`). A whole linear over K = 64..512 is one exact sum: `x1 + h @ W2.T + b2`
  (`clean/reference/swin.py:168-169`), quantized by `q8` only at stored points.
- Model M decoder projection: `q8(RN16(RN16(b + W[:, :32] Z[:32]) + W[:, 32:] Z[32:]))`, exact partial sums
  (`clean/reference/decoder.py:78-84`). Encoder: FP16 running sum per 20-input group, group sum exact
  (`clean/reference/encoder.py:262-272`).
- Model L FP16 form: an FP8 op = up to 32 products + FP16 start value -> FP16, chained over 32-wide chunks in ascending
  order. "What happens inside one op is not settled": model "exact" (exact sum, one FP16 rounding) or "trunc13" (two
  blocks of 16; every term truncated toward zero to 13 bits below the largest term's exponent; block 1's FP16 result
  starts block 2) (`clean/l/reference/oracle.py:79-93,96-141`). This models the *original's* hardware, not AMD's.
- RR preset F: each step `D = R(C + sum of 32 products)`, sum exact, one FP16 rounding, steps ascending in k. "With
  e4m3 inputs every partial sum is a multiple of 2^-18 below 2^23 ... so the float64 sums are exact whatever the
  summation order" (`clean/rr_f/reference/fmt.py:181-203`).
- Products: E4M3 x E4M3 products have at most 8-bit significands and magnitudes 2^-18 .. 2^17.6, exact in f32
  (`clean/rr_f/DESIGN-51.md:191-194`). I re-derived the range: smallest nonzero |product| = 2^-9 x 2^-9 = 2^-18,
  largest = 448^2 = 200704 ~ 2^17.6. INFERRED (arithmetic).
- Signed zero: an all-zero sum is +0 in the reference, "not significant" (`clean/rr_f/reference/fmt.py:199`); the fg
  reference rounds exact zero to +0 (`clean/fg/runner/fg_wmma_probe.py` `c_signed_zero` docstring). READ.
- NaN / inf: E4M3FN has no infinities and one NaN pattern per sign; `q8` saturates +-inf to +-448 and keeps NaN
  (`clean/reference/fp8.py:1-27`); RR's Q8 sends +-inf to +-448 and NaN to the NaN code with the sign of h
  (`clean/rr_f/reference/fmt.py:49-53`). READ.

### 2.2 What RDNA 4 actually does (MEASURED-IN-REPO)

| Fact | Evidence |
|---|---|
| `v_wmma_f32_16x16x16_fp8_fp8` is not "exact sum rounded once to f32" (hypothesis H-exact rejected) | `clean/rr_f/gpu/README.md:80-81` (probes P1 / P1b) |
| The best of several models tried: f32 rounding after C and after every group of 8 products (K 0-7, 8-15 of each WMMA): 2 residual differences of 51, against 51 for exact, 17 for one rounding per WMMA, 19 for groups of 4, 52 for truncation | `clean/rr_f/head/DESIGN-55.md:171-178` (the emulation script itself is "not committed") |
| `v_wmma_f32_16x16x16_f16` is not RN32 of the exact sum either: e.g. class `random` 6.55 M of 16.8 M outputs differ from RN32; on "ordinary" data 1,216 of 62.4 M outputs more than one FP16 step off after FP16 rounding | `clean/fg/records/wmma-probe-2026-11-14.json` (`hip.random.f32.rn32_mismatch`, `verdict.hip.f32_ordinary`); `clean/fg/README.md:25` |
| `v_wmma_f16_16x16x16_f16` (FP16 accumulate) is far from the exact chunk (6.05 M outputs > 1 step) | same record; `clean/fg/README.md:25` |
| HIP and D3D12 matrix results are byte-identical on the same card | `clean/fg/README.md:28`; `clean/dx12/hlsl/wg_platform.hlsli:218-221` (probe arith3) |
| `v_wmma_i32_16x16x16_iu8` returns A B + C exactly in int32 for partial sums < 2^31 (H1') | `clean/rr_f/kernels/rr_step_i.hlsli:35-36` (`clean/rr_f/gpu/probe_i8.py`) |
| On real-like data the f32 rounding rarely matters: RR step E vs the exact reference, 72 differing values over 4 stage kinds, 51 of them one code / one FP16 step; 11 stage kinds identical | `clean/rr_f/gpu/README.md:72-79` |
| D3D12 FP32 math: one IEEE op per `+ - *`, `mad` = one fused FMA, FP32 denormals flushed (HIP keeps them; accepted difference) | `clean/dx12/README.md:28-51`, `clean/dx12/hlsl/wg_math.hlsli:4-27` |

The repo's own design notes estimate why differences are rare: an f32 rounding changes an FP16 result only when the
exact sum is within about 2^-24 (relative) of an FP16 midpoint, and an FP8 code only near an FP8 midpoint
(`clean/rr_f/DESIGN-51.md:191-197`). READ.

What the repo does **not** record: the P1b fit output (`clean/rr_f/gpu/probe_p1b_fit.py` prints it; no JSON is
committed), and whether gfx12's F16 WMMA and FP8 WMMA give the same bits on the same FP8-representable products,
although `rr_p1b_wmma` computes both from the same values in one launch (`clean/rr_f/kernels/rr_probe.hlsl:51-66`).
UNKNOWN; cheapest settling test in section 7.

## 3. Where FP8 enters the math

| Place | How | Definition or execution detail? | Label |
|---|---|---|---|
| Weights, Models M / L | DLSS stores FP16 (`data/preset-M.v1.map.json:9`); the converter's `fp8` op quantizes each to E4M3 at install time (`:113`; exact q8 in `clean/converter/README.md:197-198`); packed as E4M3 codes into WMMA fragments (`clean/runner/packing.py:92`) | Part of WrongGPU's spec: the reference uses `q8`-exact matrices (`clean/reference/swin.py:244`, `decoder.py:183`) | READ |
| Weights, RR preset F | E4M3 codes taken from the original file (op `e4m3`) | Part of the original model | READ (`clean/converter/README.md:95`) |
| Weights, K / FG | FP16, no FP8 | n/a | READ |
| Activations, Model M Swin | `a = q8(rmsnorm(x))`, `u = q8(Wqk a)`, `q8(o)`, `m = q8(rmsnorm(x1))`, `h = q8(gelu(...))` | Spec-defined roundings (`clean/reference/swin.py:154-168`) | READ |
| Activation clamp | kernel clamps to +-448 (NaN kept, `wg_sat448`) or without NaN pass-through (`wg_sat448_inner`), or no clamp (`_nc`) when the weights' flags word proves |x| <= 460 | `clean/kernels/wg_platform.h:135-175`; `clean/FORMATS.md:69,78` | READ |
| Activations, RR | Q8 of 51.1 (saturating, ties to even code, NaN by sign) | spec 51.1 (`clean/rr_f/reference/fmt.py:49-53`) | READ |
| Accumulators | FP32 C/D fragments; FP16 rounding at spec points (`wg_to_f16_ags`, `wg_f2h`) | the FP32 accumulation is an execution detail; the spec says exact sums | READ |

Conclusion: quantization of weights and activations to E4M3 is part of the model's definition in Models M, L and RR
(the CPU reference applies it), so any implementation must reproduce `q8` bit for bit. How the products are summed
is not part of the definition (the spec says exact; RDNA 4 does something else). INFERRED from the above.

## 4. Feasibility: FP8 x FP8 -> FP32 emulated with F16 x F16 -> FP32

### 4.1 E4M3 inside binary16 (INFERRED, checked by arithmetic and a NumPy enumeration of the 254 finite codes)

| Property | E4M3FN | binary16 | Consequence |
|---|---|---|---|
| Significand | 1 + 3 bits | 1 + 10 bits | every E4M3 significand fits |
| Normal range | 2^-6 .. 448 (1.75 x 2^8) | 2^-14 .. 65504 | fits |
| Subnormals | m x 2^-9, m = 1..7 (smallest 2^-9) | below 2^-14 | E4M3 subnormals are **normal** FP16 numbers: no FP16 subnormal input ever arises |
| Infinity | none | +-inf | never produced by conversion |
| NaN | S.1111.111 | 0x7E00-class | converts to a quiet FP16 NaN; payload irrelevant |
| Signed zero | 0x00 / 0x80 | 0x0000 / 0x8000 | preserved |

All 254 finite codes round-trip exactly through `np.float16`. The conversion is exact for every code.

A cheaper bit form, `(b & 0x7f) << 7 | sign`, gives the value times 2^-8 and makes the E4M3 subnormals into FP16
subnormals (`clean/rr_f/DESIGN-51.md:158`). Do not feed that form to the WMMA unless gfx11's FP16 subnormal input
handling is measured. INFERRED.

### 4.2 Products

FP16 x FP16 products formed from E4M3 values need at most 8 significand bits and lie in [2^-18, 2^17.6]. That is well
inside FP32's normal range: no overflow, no FP32 denormal. 0.6 % of these products are *not* representable in FP16
(e.g. 2^-18, or 448^2 > 65504), so the instruction must not round products to FP16. An F16 -> F32 WMMA is expected to
form products wider than FP16. INFERRED; UNKNOWN for gfx11 until measured (test T2 class `products`).

### 4.3 What must hold for bit-exactness with RDNA 4

For every input reachable by the network (E4M3 A and B, FP32 C), gfx11's `v_wmma_f32_16x16x16_f16` must return the
same FP32 bits as gfx12's `v_wmma_f32_16x16x16_fp8_fp8`. That splits into:

| Condition | Status |
|---|---|
| Exact products | gfx12 FP8: MEASURED (products exact, `clean/rr_f/DESIGN-51.md:192`). gfx11 F16: UNKNOWN |
| Same grouping and rounding inside one instruction (gfx12 FP8 fits "f32 rounding after C and after each group of 8 products, K 0-7 / 8-15") | gfx12: MEASURED approximately (2 of 51 unexplained, `clean/rr_f/head/DESIGN-55.md:174-177`). gfx11: UNKNOWN |
| Which K indices form a group: gfx11 lanes hold all 16 K per half-wave, gfx12 lanes hold 8 | UNKNOWN; if gfx11 also groups by 8, the K order can be permuted at pack time to line groups up (kernels only need A and B to share a K permutation: `clean/kernels/emu/wg_emu.cpp` header lines 4-10). INFERRED |
| Rounding mode of each internal step (RNE vs truncation) and extra bits | gfx12: the fit rejects truncation (52 of 51 unexplained). gfx11: UNKNOWN |
| FP16 subnormal inputs | irrelevant with the exact conversion (4.1). INFERRED |
| FP32 denormal C / results | flushed on D3D12 on RDNA 4 (`clean/dx12/README.md:50-51`); products never denormal. gfx11 D3D12 flush: UNKNOWN, expected the same (DXC default mode). INFERRED |
| Signed zero of an all-zero sum (e.g. C = -0 and products +-0) | gfx12 FP8: UNKNOWN (fg probe measured F16: `signed_zero` class 0 mismatches against RN32 with +0 rule, `wmma-probe-2026-11-14.json`). gfx11: UNKNOWN. Matters only for byte identity; ignored by the equivalence rules |
| NaN / inf C | rare (only overflow tests); behaviour UNKNOWN on both |
| Chain order across WMMAs (D fed back as C) | set by the kernel source, the same on both. READ (`clean/dx12/README.md:38-39`) |

If one instruction is bit-identical for all inputs, every chain of them is too, because each step's inputs are then
identical. INFERRED.

### 4.4 What the repo records for the F16 format on RDNA 4

- F16 -> F32 WMMA results are byte-identical between HIP and D3D12 (`clean/dx12/hlsl/wg_platform.hlsli:220-221`,
  probe arith3; `clean/fg/README.md:28`). MEASURED-IN-REPO.
- The F16 -> F32 WMMA is not RN32 of the exact sum (`clean/fg/records/wmma-probe-2026-11-14.json`). MEASURED-IN-REPO.
- The Linux translation test checks F16 and FP8 WMMA only "within 1e-5" for multiplies (`clean/linux/README.md:21`,
  `clean/linux/ags_layout.hlsl:28-31`). It does not pin rounding. READ.
- The runtime's create probe uses small integers whose sums are exact in FP32 by design, so it is order- and
  rounding-blind (`clean/runtime/wmma_probe.h:13-18`). READ.
- No committed comparison of 0x5 (F16) against 0x7 (FP8) on identical products. The probe that would produce it
  exists (`rr_p1b_wmma`, `clean/rr_f/kernels/rr_probe.hlsl:51-66`; `clean/rr_f/gpu/probe_p1b.py:1-4`). UNKNOWN.

### 4.5 How often a mismatch would show (INFERRED)

An f32 rounding difference only matters when the FP32 window of the 17 terms (C + 16 products) exceeds 24 bits. In
Model M's long chains (MLP W2 over K = 256 or 512, D fed back as C), C is usually large compared with the products,
so many WMMAs round. A different rounding moves the FP32 result by about 1 ulp; after `q8` that flips an FP8 code only
for values within about 2^-21 (relative) of an FP8 midpoint. RR's measurements show the same pattern: 72 differing
values over ~600 k checked (`clean/rr_f/gpu/README.md:72-79`). Expected outcome if gfx11 rounds differently: almost
everything equal, a few one-step differences, so likely MATCHES or CLOSE under `clean/EQUIVALENCE-PROTOCOL.md:10-11`,
but not byte-identical.

## 5. The activation path without `v_cvt_pk_fp8_f32`

### 5.1 What a software quantizer must do (READ, `clean/dx12/hlsl/wg_math.hlsli:153-175`, `clean/dx12/hlsl/wg_platform.hlsli:151-203`)

| Case | Hardware `v_cvt_pk_fp8_f32` (no saturate flag) | Software HLSL equivalent |
|---|---|---|
| Normal range | RNE to 3 mantissa bits, rebias 127 -> 7 | `nrm = ((a + 0x7ffff + ((a >> 20) & 1)) >> 20) - (120 << 3)` |
| |x| < 2^-6 | E4M3 subnormal, RNE on the 2^-9 grid | `round(min(a, 2^-6) * 512)` (v_rndne) |
| 448 < |x| <= 464 | rounds to 448 (0x7E): 464 is a tie, even mantissa wins | covered by `nrm` |
| |x| > 464, +-inf | 0x7F / 0xFF (NaN code) | `a > 0x43e80000 ? 0x7f` (`wg_e4m3_hw`) |
| NaN | 0xFF | `a > 0x7f800000 ? 0xff` |
| Clamp (kernel's choice) | `wg_sat448` (NaN kept) or `wg_sat448_inner` (NaN -> +-448) before conversion | same helpers, then `wg_fp8` |

The repo states these software forms give the hardware's bytes "incl. NaN / overflow" (`clean/dx12/README.md:47-48`)
and that HIP's software and hardware conversions were checked bit-exact on the GPU (`clean/kernels/wg_platform.h:131-134`;
`clean/runner/selftest.py:13-15`). MEASURED-IN-REPO for gfx12. Exception: FP32 denormal inputs (flushed on D3D12),
already an accepted difference (`clean/dx12/README.md:50-51`).

### 5.2 For the F16-emulation route the quantizer can emit FP16 bits directly (INFERRED)

The WMMA needs FP16 operands, not E4M3 codes. Two equivalent ways:
- Quantize to the E4M3 value held in FP32 (the same bit trick, keeping the value rather than the code), then
  `v_cvt_pkrtz_f16_f32` or `v_cvt_f16_f32`: exact for E4M3 values, so the rounding mode does not matter.
- Or compute the code with `wg_e4m3_hw` and decode to FP16 (bit form plus a 2^8 scale with `v_pk_mul_f16`, exact,
  `clean/rr_f/DESIGN-51.md:158`).

Cost shape per value: clamp 1-2, normal-path add/shift/sub 3-4, subnormal path min/mul/rndne 3, selects 2-3, FP16
pack 0.5: about 8-10 VALU per value, against ~1.5 on gfx12 (clamp + half a `v_cvt_pk_fp8_f32`). Stored activations
(FP8 maps between stages, `clean/FORMATS.md:12-13`) still need the E4M3 code byte, so the code path stays for
stores; the FP16 bits are for the next WMMA. Exact VALU counts need an RGA build for gfx1100 (lane for performance).

Also affected:
- The MLP GELU lookup ends in `v_cvt_pk_fp8_f32` (`clean/dx12/hlsl/wg_gelu_lut.hlsli:5-14`, 4 files). Its table output
  `fma(h, a, b)` would go through the software quantizer. The table proof ("identical bytes for all 2^32 inputs") was
  done against gfx12 (`:17-18`) and must be redone. READ + INFERRED.
- FP8 -> F32 decode through the driver (`WG_AGS_T_FP8_B -> WG_AGS_T_F32_ACC`, 9 files) has a software twin,
  `wg_e4m3_to_f32` (`clean/dx12/hlsl/wg_platform.hlsli:170-178`), exact. READ.

### 5.3 Reliance on the F32 -> F16 driver conversion

Yes, heavily. `wg_to_f16_ags` (driver convert op 0x32, F32 acc -> F16 acc) is in 26 HLSL files, and the
FloatConversion op 0x1c is in 12 (`clean/dx12/hlsl/wg_platform.hlsli:238-243`; `clean/linux/ags_layout.hlsl:34-36`).
Both are documented as `v_cvt_f16_f32` RNE (`clean/dx12/hlsl/wg_ags.hlsli:15`). gfx11 has `v_cvt_f16_f32`, so the
semantics exist. Whether AMD's gfx11 D3D12 driver accepts those channel opcodes is a driver question (other lanes).
A software fallback exists, `wg_f2h` (~20 instructions per value, `clean/dx12/README.md:79-83`;
`clean/dx12/hlsl/wg_math.hlsli:88-97`). Note the driver-specific facts measured on gfx1201 that must be rechecked on a
gfx11 driver: `(half)` casts round toward zero, FP16 multiply-add fused despite `-Gis`, FP32 ops folded into the
matrix conversion (`clean/dx12/hlsl/wg_math.hlsli:25-39`). READ + INFERRED.

## 6. Alternatives and their numerical consequences

| Option | Bit-exact with RDNA 4 output? | Bit-exact with the CPU reference? | Cost shape (INFERRED) | Proof or test needed |
|---|---|---|---|---|
| (a) F16 emulation: E4M3 -> FP16 (exact), `v_wmma_f32_16x16x16_f16` | **Conditional**: only if gfx11's F16 WMMA rounds its partial sums exactly like gfx12's FP8 WMMA (4.3). UNKNOWN | No (same as RDNA 4: f32 accumulation) | A/B operands: gfx11 wave32 needs 16 halves per lane (8 VGPRs) vs 2 VGPRs of FP8 on gfx12: 4x registers and conversion work for activations; weights pre-converted to FP16 at pack time (2x memory) | T1-T3 below: per-instruction bit compare against RDNA 4's FP8 results on adversarial and real-like inputs; then stage-level byte compare |
| (b) IU8 matrix instruction with re-quantized int8 weights (per-channel scales) | **No** | No | 1 WMMA per step, cheap | A new accuracy contract (spec 09-style or equiv CLOSE) and new reference; changes the model |
| (c) F16 weights un-quantized (skip the `fp8` op) | **No** | No (reference uses q8 weights) | as (a) without activation quantization savings | New contract; outputs differ from RDNA 4 and from the spec everywhere |
| (d) No matrix instructions: FP32 FMA dot products | **Conditional and hard**: would have to reproduce gfx12's measured grouping (RN32 after C and every 8 products) exactly, and that model leaves 2 of 51 cases unexplained | No | 16-32x more VALU work per step; slow | A complete, verified model of gfx12's FP8 WMMA first (not in repo) |
| (e) IU8 digit planes, exact integer sum (RR step "I", `clean/rr_f/kernels/rr_step_i.hlsli:1-33`) | **Conditional**: equal wherever RDNA 4's step happens to be exact (most values; RR step E vs exact: 72 differences over ~600 k values) | **Yes**, by construction, within its guards; windows that fail the int32 bound fall back | 4-9 IU8 WMMAs per 32-product step instead of 2, plus digit extraction; fallback path today is step X (needs FP8 WMMA), so gfx11 needs another fallback, e.g. (f) | gfx11 IU8 exactness (H1') re-measured with `clean/rr_f/gpu/probe_i8.py`; then the step-I vs step-X stage comparison (`clean/rr_f/gpu/step_cmp.py`) on gfx11 vs RDNA 4 references |
| (f) Exact VALU step in binary64 (RR step 2, `clean/rr_f/kernels/rr_exact_math.hlsli`, `rr_step.hlsli:10-14`) | Conditional as (e) | **Yes** | Slow (binary64); suitable only as the rare fixup path for (e) | Existing RR vector tests (`run_all.py`) on gfx11 |
| (g) Exact sum via F16 WMMA twice (RR step X: `e = WMMA(a, b, -p)`) | Conditional as (e) | Only if the WMMA's error is within the certified bound; RR says step X is "exact on the vectors but not certified for every input" on RDNA 4 (`clean/rr_f/gpu/README.md:80-81`) | 2x WMMA plus error-free transforms; ~4.5x step E's time on RDNA 4 (`clean/rr_f/gpu/README.md:84-90`) | gfx11 accumulation model measured first (T1) |

Note on (b): an E4M3 value needs 4 significand bits and 4 exponent bits, so it cannot be one int8 without changing
the value. (e) avoids this by splitting each block into 7-bit digit planes on a shared scale (`rr_step_i.hlsli:5-20`).
That is the only integer route the repo has that keeps the model unchanged. READ + INFERRED.

Note on the contract: options (e)/(f) make RDNA 3 *more* exact with respect to the spec than RDNA 4's default path.
If the maintainer's contract is "same bytes as RDNA 4", only (a) or (d) can meet it, and (a) only if T1 passes. If the
contract is level 2 or 3 of section 1.3, (e) with (f) as fixup reaches level 2, and (a) likely reaches level 3 even
if T1 fails. INFERRED.

## 7. Minimal exactness test plan on an RX 7900 XT (cheapest kill test first)

T0 runs on the existing RX 9070 XT and needs no RDNA 3 card. Every later step reuses its saved inputs.

| # | Test | Reuses | Kills the F16-emulation idea if | Cost |
|---|---|---|---|---|
| T0 | On RDNA 4: run probe P1b, which computes `D8 = wmma_fp8(A, B, C)` and `D16 = wmma_f16(widen(A), widen(B), C)` on the same values; compare D8 and D16 bit for bit per data mode; save the npz (A, B, C, D8) as the golden set | `clean/rr_f/gpu/probe_p1b.py`, `clean/rr_f/kernels/rr_probe.hlsl:51-66`, `probe_p1b_fit.py` (add a D8 == D16 count) | not a kill test for gfx11 by itself, but if gfx12's own F16 path differs from its FP8 path, the two datapaths round differently and a gfx11 match is less likely; it also gives the golden D8 | minutes, existing card |
| T1 | On the 7900 XT: a single-wave probe running gfx11 `v_wmma_f32_16x16x16_f16` (HIP builtin `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` for gfx1100, or the D3D12 channel if the driver accepts it) on T0's A, B (converted exactly to FP16, gfx11 layout: 16 halves per lane, duplicated half-waves), C; compare with T0's D8 bit for bit, per mode | T0 npz; P1b's 8 data modes (sparse, dense, one band, big+tiny, wide C) | any mismatch in any finite case. Report counts, ULP distances, and which grouping model (`probe_p1b_fit.py`'s exact / floor-F / per-8) explains gfx11 | about one hour to write the probe; seconds to run |
| T2 | Adversarial classes restricted to E4M3-representable operands: cancellation, ties at FP32 rounding boundaries decided by a 2^-18 product, large C with tiny products (the Model M MLP chain case), signed zeros (C = -0, all products +-0), NaN codes, C = +-inf, FP32 values near overflow | case generators in `clean/fg/runner/fg_wmma_probe.py` (`c_cancel_acc`, `c_ties`, `c_signed_zero`, `c_overflow`), adapted to E4M3 values; golden D8 from RDNA 4 | any finite mismatch; a signed-zero-only mismatch is a byte-identity failure but not an equivalence failure | hours |
| T3 | K-order sensitivity: repeat T1 with the 16 K slots permuted (gfx12 pairs K 0-7 / 8-15; if gfx11 groups differently a pack-time permutation may restore equality) | T1 harness; the emulator already models K permutations (`clean/kernels/emu/wg_emu.cpp`, `emu_set_layout`) | no permutation gives zero mismatches | minutes after T1 |
| T4 | Software quantizer on gfx11: `wg_e4m3_hw` / `wg_pack_q8` / `wg_pack_q8_inner` over all 2^32 FP32 bit patterns, compared with gfx12's `v_cvt_pk_fp8_f32` bytes (or with `clean/reference/fp8.py` q8 for clamped values) | `clean/runner/selftest.py` item 3; `clean/dx12/gelu_lut/run.py --table` style exhaustive loop | any difference outside FP32 denormal inputs (then the quantizer, not the matrix path, is wrong) | minutes on GPU |
| T5 | Exact-sum alternative: IU8 exactness on gfx11 (H1') and the integer step vs the exact reference | `clean/rr_f/gpu/probe_i8.py`, `clean/rr_f/gpu/step_cmp.py`, RR vectors (`clean/rr_f/tests/gen_vectors`) | n/a (tests option e) | hours |
| T6 | Stage level, Model M: chained stage dumps on RDNA 4 (golden) and on the RDNA 3 port, same synthetic frames; byte compare, then `clean/equiv/compare.py` verdicts if not identical | `clean/equiv/model_m.py` (dumps every boundary), `clean/equiv/compare.py` (`--ours` RDNA 3, `--orig` RDNA 4 dumps), `clean/equiv/probes.py` inputs, synthetic models from `clean/runner/synthetic_model.py` | for contract level 1: any byte difference. For level 3: a DIFFERENT verdict | needs the port |
| T7 | Model K and FG (F16 weights): repeat T1 with FP16 operands that are not E4M3 values (full FP16 range, subnormals) against RDNA 4's F16 WMMA; the fg probe classes already cover this | `clean/fg/runner/fg_wmma_probe.py` and its record | any mismatch means K/FG cannot be byte-identical to RDNA 4 on gfx11 either | as T1 |

Order rationale: T0 costs nothing new and produces the golden data. T1 is one small kernel on the new card and can
end option (a) for contract level 1 on the first run. T2/T3 only matter if T1 is clean on random data. T4 is
independent and cheap. T6 needs the real port and comes last.

## Open questions

1. Which contract should an RDNA 3 path meet: byte-identical to RDNA 4, exact against the reference (RR "exact"
   sense), or the tolerance rules plus equivalence CLOSE against RDNA 4? This decides whether options (e)/(f) or (a)
   are worth building. Product decision for the maintainer.
2. Are the P1b results (fit output, D8 vs D16 on gfx12) recorded anywhere outside the repo
   (`QUESTIONS-51.md` and `results/2026-10-21-body-levers.md` are cited but not in the repo)?
3. The per-8-products model of gfx12's FP8 WMMA leaves 2 of 51 cases unexplained (`clean/rr_f/head/DESIGN-55.md:174-177`).
   Is there a complete model? Without one, option (d) cannot be built and T1 has no model to compare against, only
   golden bytes.
4. Does the AMD gfx11 D3D12 driver expose the intrinsic channel's matrix ops (0x28 with format 0x5 and 0xC) and the
   conversions (0x32, 0x1c)? If not, every conversion uses the software helpers (sections 5.2, 5.3). (Driver lane.)
5. Model L: the oracle's FP16-form question ("what happens inside one op is not settled", `clean/l/reference/oracle.py:82-83`)
   concerns the original's hardware; it does not change for RDNA 3, but any RDNA 3 path for Model L inherits it.
6. No published bit-level model of RDNA 3 WMMA was found. Published models cover CDNA matrix cores and NVIDIA tensor
   cores: https://arxiv.org/pdf/2609.14845, https://arxiv.org/pdf/2511.10909v1, https://arxiv.org/html/2403.00232v1.
   They show vendors differ in alignment, truncation and subnormal handling, so gfx11 must be measured, not assumed.

## What to measure on an RX 7900 XT

1. T1: gfx11 `v_wmma_f32_16x16x16_f16` on exact FP16 widenings of E4M3 operands vs RDNA 4's FP8 WMMA golden bytes
   (P1b data modes). Count mismatches, ULP distance, and fit the exact / truncation / per-8 / per-4 models.
2. T2: the same on adversarial classes (cancellation, FP32 ties, large C plus tiny products, signed zeros, NaN codes,
   infinite C).
3. T3: whether a K permutation at pack time removes the mismatches.
4. Product exactness in the F16 WMMA: products 2^-18 and 448^2 (not FP16-representable) summed with a C that exposes
   their low bits.
5. FP16 subnormal input handling of gfx11 WMMA (only needed if the 2^-8-scaled bit form is used), and FP32 denormal
   flushing of C and D under D3D12.
6. T4: the HLSL software E4M3 quantizer over all 2^32 FP32 inputs on the gfx11 D3D12 compiler, against gfx12's bytes.
7. T5: `v_wmma_i32_16x16x16_iu8` exactness (probe_i8) on gfx11, if the exact-sum route is considered.
8. Driver rules measured on gfx1201 (`clean/dx12/hlsl/wg_math.hlsli:25-39`, `clean/dx12/README.md:53-91`): `(half)`
   rounding direction, FP16 multiply-add fusion, folding into conversions, dropped stores, all re-probed on the gfx11
   driver before any byte comparison means anything.

External sources: https://gpuopen.com/learn/wmma_on_rdna3/ (gfx11 wave32 A/B fragments of 16 elements, duplicated
across half-waves); https://gpuopen.com/learn/using_matrix_core_amd_rdna4/ (RDNA 4 removed the duplication);
https://llvm.googlesource.com/llvm-project/llvm/+/6a4c0551fbd905397666676142d59af81585c2cb/test/MC/AMDGPU/gfx11_wmma.s
(gfx11 WMMA variants: f16, bf16, iu8, iu4; no FP8).
