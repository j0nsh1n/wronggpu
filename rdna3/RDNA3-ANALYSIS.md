# RDNA 3 (gfx11) for WrongGPU: analysis before any game testing, with hardware results

2026-11-14. Card measured: Radeon RX 7900 XT (Navi 31, gfx1100), Mesa RADV 27.0-devel, Linux.

Source: upstream main at 994f304. Repo paths are relative to the repo root; probe paths are relative to this analysis
folder. External facts carry their URL. Labels: "the repo records" (a measurement committed in the tree), "LLVM's
source says", "AMD's guide says" and similar (read in a named public source), "measured on an RX 7900 XT (Linux)"
(run for this document, Mesa RADV, native Vulkan cooperative-matrix and compute shaders), "inferred" (reasoning shown),
"unknown until measured" (a test is named).

How this was made: a read of the tree and of public sources (LLVM, Mesa, AMD's ISA guide, LLPC and FidelityFX
repositories, GPUOpen, papers and press), done with Claude models and checked against the cited lines; then three
probes on an RX 7900 XT under Linux: the matrix instruction's arithmetic (`probe-vk/`), the effect of its errors after
quantization (`probe-post/`, CPU only), and shader and matrix timing (`probe-vk-valu/`); and a CPU study of INT8
forms of Model M on a real converted model with the repo's float64 reference (`int8-study/`; aggregate numbers only,
no weights leave the machine). No game was run. Nothing
here was run on Windows or on an RDNA 4 card. The Linux results are facts about the chip's instructions, not about
AMD's Windows driver.

Companion files, in the same folder as this document. "Lane N" in the text means the research notes file
`findings/N-*.md`.

| Path | What |
|---|---|
| `findings/1-kernel-inventory.md` to `9-rdna3-docs-throughput.md` | research notes with full citations: kernel inventory, gates, numerics, ISA and driver, Linux stack, AMD's INT8 model, INT8 prior art, per-model formats, documented throughput |
| `probe-vk/` | matrix-instruction arithmetic probe: sources and `RESULTS.md`. Its `results/*.npz` (inputs and this card's outputs, about 0.7 GB, for the replay on RDNA 4) are downloads on the release https://github.com/j0nsh1n/wronggpu/releases/tag/rdna3-probe-data-1 : put them in `probe-vk/results/` |
| `probe-post/` | what those errors do after quantization (`POST.md`, scripts) |
| `probe-vk-valu/` | shader and matrix timing (`RESULTS-VALU.md`, sources, disassembly) |
| `int8-study/` | CPU study of INT8 forms (`STUDY.md`, scripts; `int8_study.py` runs on any model file). Its per-tensor result tables are not published: they describe a model converted from the tester's own DLSS files |

## 1. Summary

- What RDNA 3 lacks: every FP8 instruction WrongGPU uses (`v_wmma_f32_16x16x16_fp8_fp8`, `v_cvt_pk_fp8_f32`,
  `v_cvt_f32_fp8`, FP8 dot products). It has F16, BF16, IU8 and IU4 WMMA, but with a different operand layout: A and B
  replicated across half-waves, accumulator rows interleaved, so every D -> B step needs a half-wave exchange.
- Measured: the F16 matrix instruction fed exact E4M3 values returns slightly inexact FP32 sums when terms have mixed
  signs (1 + (-1) gives -2^-24). No rounding model of 13 tried fits. It is deterministic and does not depend on lane or
  dispatch position, but it does depend on the K order (26-39 % of outputs change under a K permutation).
- After quantizing to E4M3, about 450 outputs per million change by one code on P1b-style data, 86 % of them at exact
  ties; 2 per million on uniform random E4M3. So the FP16 route gives output close to the reference and to RDNA 4, not
  identical. Identity with RDNA 4 is possible only if RDNA 4's FP8 instruction shows the same artefacts. Only a replay
  of the saved inputs on an RX 9070 XT can settle that (section 10, item R0).
- Measured: F16 and IU8 matrix instructions take the same 32-33 cycles. Without hardware FP8 conversion, the FP16
  route's cost is glue: software E4M3 quantization (133 cycles per tile, 4x a WMMA) and widening of the codes (141).
  For a 16-deep layer step with fresh operands the matrix instruction is 10 % of 322 cycles; a single-plane INT8 route (one that changes the
  model, see below) needs 59 (about 5x cheaper; about 3x for deeper K, inferred). Packed INT8 dot products run at about half the matrix rate (63
  cycles per 4096 products), not at the matrix rate the compilers' cost models predict. Vector work never overlaps a
  WMMA: times add.
- INT8 is exact as an instruction (wrapping int32 matched on every tested output) but changes the model unless the
  E4M3 values are split into exact digit planes. Measured on a real Model M (CPU, synthetic probe frames, section
  7.4): one INT8 plane, under every scale choice tried, makes every stage DIFFERENT (network-output SNR 17-18 dB).
  Two planes per operand are exact, but that is about 3.9 multiplies per product, and with the E4M3 quantizer still
  needed it costs about what the FP16 route costs (about 310-330 cycles per step, inferred). So INT8 gives Model M
  either a cheap different model or an exact route at FP16-route cost, not a cheap exact one.
- AMD (FSR 4.1 on RX 7000) and Intel (XeSS DP4a) each shipped a separately made integer model; both own the training
  data. WrongGPU can calibrate from captured frames but cannot retrain.
- Per model: K and FG are FP16 models already and fit RDNA 3's F16 instruction with layout work only (use the
  FP32-accumulate form; RDNA 3's FP16-accumulate form rounds after every pair). M, L and RR are E4M3 models; they need
  the FP16 route, digit planes, or a new INT8 model.
- The decision is yours: which contract RDNA 3 must meet. "Same bytes as RDNA 4" depends on the replay. "Exact against
  the reference" points to digit planes (two planes per operand on this data). "CLOSE against RDNA 4" is likely met by
  the FP16 route. A single-plane INT8 model needs a new quality contract and calibration frames.
- For the E4M3 models the cost of either faithful route is the software E4M3 quantizer (133 of about 320 cycles per
  step), which RDNA 4 does in one instruction. A cheaper exact quantizer is the largest lever (inferred). Model K has
  no such step: about 48 cycles per step on RDNA 3's F16 instruction (inferred from the measured parts).
- Next measurements: (R0) replay `probe-vk/results` on an RX 9070 XT; (M1, M2) whether AMD's Windows driver compiles
  the WaveMatrix channel for gfx1100; (P1) digit-plane counts on captured game frames.

## 2. RDNA 3 versus RDNA 4: hardware and driver facts

### 2.1 Matrix and conversion instructions per generation

LLVM's source says (`GCNFeatures.td` lines 993-999 and the `FeatureISAVersion11_*` / `FeatureISAVersion12` sets;
`VOP3PInstructions.td` encodings 0x40-0x5a):
https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/Target/AMDGPU/GCNFeatures.td ,
https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/Target/AMDGPU/VOP3PInstructions.td .
Mesa's ACO table agrees (`src/amd/compiler/aco_opcodes.py` lines 1250-1271,
https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/amd/compiler/aco_opcodes.py ).

| Instruction (VOP3P opcode) | gfx11 (RX 7000, RDNA 3.5 APUs) | gfx12 (RX 9000) |
|---|---|---|
| `v_wmma_f32_16x16x16_f16` (0x40), `_bf16` (0x41) | yes | yes |
| `v_wmma_f16_16x16x16_f16` (0x42), `v_wmma_bf16_16x16x16_bf16` (0x43) | yes, with an OPSEL immediate choosing the half of each C/D VGPR | yes, no OPSEL, packed C/D |
| `v_wmma_i32_16x16x16_iu8` (0x44), `_iu4` (0x45) | yes | yes |
| `v_wmma_f32_16x16x16_{fp8,bf8}_{fp8,bf8}` (0x46-0x49), `v_wmma_i32_16x16x32_iu4` (0x4a) | no | yes |
| `v_swmmac_*` (0x50-0x5a) | no | yes |
| `v_cvt_pk_fp8_f32`, `v_cvt_f32_fp8`, `v_cvt_pk_f32_fp8`, bf8 forms | no (need `FP8ConversionInsts`) | yes |
| `v_dot4_f32_fp8_*` | no (need `Dot11Insts`) | yes |
| `v_cvt_f16_f32`, `v_cvt_pk_rtz_f16_f32`, `v_fma_mix*`, `v_dot2_f32_f16`, `v_dot4_i32_iu8` | yes | yes |
| `v_permlane16_b32`, `v_permlanex16_b32` | yes (since gfx10; ISA3 opcodes 603/604) | yes, plus `_var` forms |
| `v_maximum_f32` / `v_minimum_f32` family (`IEEEMinimumMaximumInsts`) | no | yes |

Measured on an RX 7900 XT (Linux): RADV advertises 14 cooperative-matrix configurations, all 16x16x16, compute only:
u8/s8 into u32/s32 (saturating and wrapping) and f16 into f16 or f32. No BF16 and no FP8 entries
(`probe-vk/RESULTS.md` section 1). ACO emitted exactly one `v_wmma_*` per product in each pipeline.

Two traps. LLVM has a `FeatureISAVersion11_7` set that is gfx11-numbered but carries the gfx12 matrix design
(`WMMA128bInsts`, the FP8 conversions, `SWMMACGfx1200Insts`); the research notes report LLVM's docs name gfx1170-1172
"RDNA 4m" APUs (not re-checked). A "gfx11" family check must not match them. RDNA 3.5 (gfx1150-1153) has the RDNA 3
set. LLVM's own feature strings state the layout difference: `wmma-256b-insts`, "Has WMMA instructions where A and B
matrices have duplicated data" (gfx11), against `wmma-128b-insts`, "... do not have duplicated data" (gfx12).

### 2.2 Operand registers per lane (VGPRs)

LLVM's register classes, AMD's matrix calculator (https://github.com/ROCm/amd_matrix_instruction_calculator ), and
GPUOpen's RDNA 3 article (https://gpuopen.com/learn/wmma_on_rdna3/ ) agree:

| Instruction | gfx11 w32 A/B, C/D | gfx11 w64 A/B, C/D | gfx12 w32 A/B, C/D | gfx12 w64 A/B, C/D |
|---|---|---|---|---|
| F32 <- F16 / BF16 | 8, 8 | 8, 4 | 4, 8 | 2, 4 |
| F16 <- F16 | 8, 8 (unpacked) | 8, 4 | 4, 4 (packed) | 2, 2 |
| I32 <- IU8 | 4, 8 | 4, 4 | 2, 8 | 1, 4 |
| F32 <- FP8 | n/a | n/a | 2, 8 | 1, 4 |

So an FP8 product that costs 2 + 2 operand VGPRs on gfx12 costs 8 + 8 as F16 on gfx11: four times the operand
registers. Wave64 brings no saving on gfx11. The measured disassembly matches: F16 A and B in 8 VGPRs each, IU8 in 4,
C/D in 8 (`probe-vk/RESULTS.md` section 1, "Disassembly evidence").

### 2.3 Lane layout (wave32)

From the calculator output and Mesa's RADV lowering comments
(`src/amd/vulkan/nir/radv_nir_lower_cooperative_matrix.c`, lines 12-37):

| Element | gfx11 | gfx12 (what WrongGPU assumes) |
|---|---|---|
| A[i][k], F16 | lanes i and i+16 (identical), VGPR k/2 | one half-wave per half of K |
| B[k][j], F16 | lanes j and j+16 (identical), VGPR k/2 | one half-wave per half of K |
| A, B, FP8 | n/a | lane 16*(k/8) + i, VGPR (k/4)%2, byte k%4 |
| C/D[i][j], F32 | lane 16*(i%2) + j, VGPR i/2 (row = 2v + h) | lane 16*(i/8) + j, VGPR i%8 (row = v + 8h) |
| C/D, F16 | as F32, one half per VGPR (OPSEL picks which) | packed, two rows per VGPR |

Measured on an RX 7900 XT (Linux): the ACO loads read A and B by `lane & 15`, so lanes 16-31 repeat lanes 0-15, and
C/D hold even rows in lanes 0-15 and odd rows in lanes 16-31 (`probe-vk/RESULTS.md` section 1).

GPUOpen's RDNA 4 article states the consequence: on RDNA 3 "in order to convert D matrix to B matrix ... data needs
to be shuffled among lanes ... This is not needed for RDNA 4", and the format "does not have backward compatibility"
(https://gpuopen.com/learn/using_matrix_core_amd_rdna4/ ). The vkd3d-proton author says the same: "On RDNA3, this
doesn't work since len(B) != len(Accum)" (https://themaister.net/blog/2025/05/ ).

The exchange. On gfx11 a lane holds 8 of a token's 16 accumulator rows and needs all 16 as B, so every D -> B step
needs the other half-wave's 8 values: one `v_permlanex16_b32` per value, 8 per lane per tile. Measured on an RX 7900
XT (Linux): GLSL `subgroupShuffleXor(v, 16)` compiles to `v_permlanex16_b32` (a vector instruction, no LDS) and costs
14.7 cycles per tile; the run-time-index form `subgroupShuffle(v, lane ^ 16)` compiles to `ds_bpermute_b32` and costs
46.5 (`probe-vk-valu/RESULTS-VALU.md` section 4). WrongGPU's `wg_xor16` is `WaveReadLaneAt(v, WaveGetLaneIndex() ^ 16)`
(`clean/dx12/hlsl/wg_platform.hlsli:103-107`), the run-time-index form. Which instruction AMD's D3D12 compiler emits
for it on gfx1100 is unknown until read in RGA. The row interleave (2v + h) costs nothing: A and B only need to share a
K order, and the packer already has a `gfx11` row table (`clean/runner/packing.py:16-19`).

One detail for gfx12 F16: the calculator puts F16 K in groups of 4 across half-waves (lanes 0-15 hold K 0-3 and
8-11), while WrongGPU puts logical K 0-7 in lanes 0-15 (`clean/runtime/wmma_probe.h:9-12`). Both multiply correctly.
It matters for rounding if the hardware groups by hardware K. On gfx11 the probe stored logical K = column k of A in
the order the instruction sees it, and the result depends on that order (section 6.3).

### 2.4 Wave size, registers, occupancy

- Wave size: both generations run wave32 and wave64 compute; WrongGPU forces it per kernel with `[WaveSize]`
  (`clean/dx12/hlsl/wg_platform.hlsli:22-23,40`) and defines the matrix helpers for wave32 only (`:206`).
- VGPR file: LLVM's source says `Feature1536VGPRs` is set for gfx1100 and gfx1101 (RX 7900, RX 7800/7700) and for
  gfx12, and is absent for gfx1102 (RX 7600) and gfx1103, which have 1024. So `clean/dx12/vgpr_limits.json` arithmetic
  carries over to the 7900 XT, not to the 7600.
- Allocation: AMD's ISA guide and LLVM say 1536-VGPR parts allocate in blocks of 24 (wave32); waves per SIMD =
  min(16, floor(1536 / roundUp(V, 24))): 128 VGPRs give 10 waves, 192 give 8, 256 give 5 (lane 9, II.5).
- LDS: 128 KB per WGP, at most 64 KB per work-group (AMD's ISA guide). Measured on an RX 7900 XT (Linux): a
  256-invocation kernel using 32 KiB of LDS ran 8 waves per SIMD, others 16, which matches 128 KB per WGP.
- Matrix and shader rates: sections 4 and 5.

### 2.5 The driver channel: what is public

- WrongGPU's codes are AMD's private WaveMatrix intrinsics. The enum is published in AMD's open-source driver
  compiler: https://raw.githubusercontent.com/GPUOpen-Drivers/llpc/dev/compilerutils/include/public/AmdExtD3DShaderIntrinsics.h
  (opcodes at lines 121-138, WaveMatrix opcode enum at 428-450, support values at 260-261). Every value in
  `clean/dx12/hlsl/wg_ags.hlsli:26-34` matches it.
- The public AGS SDK (v6.3.1) does not contain them. Its opcode list ends at `ShaderRealtimeClock 0x20`; its
  `extensionsSupported` bits stop at `shaderClock`
  (https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/master/ags_lib/hlsl/ags_shader_intrinsics_dx12.hlsl ,
  https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/master/ags_lib/inc/amd_ags.h ).
- `FloatConversion` 0x1c is the exception: it is in public AGS, with modes `FToF16Near` 1, `NegInf` 2, `PlusInf` 3,
  and an AGS support bit `floatConversion` ("Radeon Software Version 20.5.1 onwards"). Inferred: 0x1c is very likely
  available on RX 7000, and AGS can say so before any shader runs.
- How WaveMatrix support is queried: `IAmdExtD3DShaderIntrinsics::CheckSupport(AmdExtD3DShaderIntrinsicsSupport_WaveMatrix
  = 0x1F)` and `..._Float8Conversion = 0x20`, plus `GetInfo` for wave sizes. The interface comes from the driver's
  `AmdExtD3DCreateInterface`, not from AGS (LLPC header; the FidelityFX 2.0 `AmdExtD3DShaderIntrinsicsApi.h`, mirrored
  at https://github.com/Uklosk/FidelityFX-SDK-2.0.0/tree/main/Kits/FidelityFX/api/internal/dx12/AmdExtD3D ).
- The header lists the union of both generations' formats and says nothing about which GPU accepts which. AMD
  publishes no support list.
- Indirect evidence the Windows driver has a gfx11 matrix path: AMD's 2023 Agility SDK preview driver exposed SM 6.8
  WaveMMA "on Radeon RX 7000 series"
  (https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-MS-AGILITY-SDK-2023-6-711.html ). Current
  preview drivers list SM 6.10 Linear Algebra and Cooperative Vector for RX 9000 only.
- FSR 4.1 on RX 7000 uses a separate INT8 model (section 7.3). Which instructions it uses on RX 7000, and whether it
  goes through the WaveMatrix intrinsics, is not public; unknown until an RGP or RGA capture of FSR 4.1 on an RX 7900.

### 2.6 WrongGPU's channel codes against the facts

| WrongGPU code (`wg_ags.hlsli`) | AMD enum | gfx12 instruction | gfx11 hardware | Windows driver on RX 7000 |
|---|---|---|---|---|
| 0x28 fmt 0x7 (`wg_wmma_fp8`) | `WMMA_F32_16X16X16_FP8_FP8` | `v_wmma_f32_16x16x16_fp8_fp8` | none | unknown: reject at PSO, emulate, or wrong bits |
| 0x28 fmt 0x5 (`wg_wmma_f16`) | `WMMA_F32_16X16X16_F16` | `v_wmma_f32_16x16x16_f16`, A/B 4 dwords | same name, A/B 8 dwords, replicated | unknown whether accepted; if so, layout unknown |
| 0x28 fmt 0x1 (probe only) | `WMMA_F16_16X16X16_F16` | packed C/D | unpacked C/D with OPSEL | unknown |
| 0x28 fmt 0xC (RR step I) | `WMMA_I32_16X16X16_I8` (signed x signed; IU8 is 0xE) | `v_wmma_i32_16x16x16_iu8` | exists, A/B 4 dwords | unknown |
| 0x32 0x26 -> 0x1a (`wg_pack_q8*_ags`) | WaveMatrixCopy F32 Acc -> FP8 B | `v_cvt_pk_fp8_f32` | none | unknown; even if emulated, Acc and B layouts differ |
| 0x32 0x1a -> 0x26 (`sw_unpack_fp8`, `rr_d8`) | FP8 B -> F32 Acc | `v_cvt_f32_fp8` | none | unknown |
| 0x32 0x26 -> 0x24 (`wg_to_f16_ags`) | F32 Acc -> F16 Acc | `v_cvt_f16_f32`, 4 packed dwords | instruction exists; F16 Acc is 8 unpacked dwords | unknown; reading `r.v[0..3]` would be wrong if the driver uses gfx11's native Acc |
| 0x32 0x24 -> 0x26, 0x24 -> 0x1a | F16 Acc -> F32 Acc / FP8 B | element converts | F16 -> F32 exists, FP8 not | unknown |
| 0x1c mode 1 (`k_out.hlsli:121`, 12 files) | public AGS `FloatConversion`, `FToF16Near` | `v_cvt_f16_f32` | `v_cvt_f16_f32` | likely; AGS `floatConversion` bit answers it |

Unknown about the Windows driver on RX 7000, to settle on the card: (1) does `CheckSupport(WaveMatrix)` return S_OK
and does a 0x28 shader compile; (2) are the phase-0 dwords gfx11's native registers (8 dwords of F16 A/B, lanes L and
L+16 equal) or does the driver accept gfx12-shaped fragments and relayout; (3) which formats it accepts (0x5, 0x1,
0xC, 0x7); (4) whether WaveMatrixCopy Acc -> B inserts the half-wave exchange; (5) whether it emulates FP8, and with
which rounding. vkd3d-proton's layout test comment ("To detect if we're on RDNA3, which will fail this test") reflects
vkd3d-proton's own backend, not AMD's driver. A route without matrix instructions (`v_dot4_i32_iu8` through SM 6.4
`dot4add_i8packed`) needs none of these answers (inferred).

## 3. What in WrongGPU depends on RDNA 4

### 3.1 The kernels, per model

Static matrix call lines per HLSL file (grep, probes excluded; an `[unroll]` call is one line but many instructions):

| Model | Files | FP8 lines | F16 -> F32 lines | Other | Status |
|---|---|---|---|---|---|
| M (and L's shared block) | `swin_block` 11, `fused_swin` 13, `swin_c64w2` 10, `swin_gemm` 2, `bneck` 7, `fused_bneck` 9, `dec1_dec0` 14, `fused_dec` 14, `encoder` 2 (all `clean/dx12/kernels/`) | 82 | 7 (attention `A V`, e.g. `swin_block.hlsl:848`) | 0 | shipped |
| L extra | `l_ops_linear` 2, `l_ops_project` 2, `clean/dx12/hlsl/l_lin_px.hlsli` 2 | 6 | 0 | 0 | development |
| K | `clean/k/dx12/hlsl/k_block.hlsli` (`wgk_mma`, 11 calls), `k_block_wide.hlsl` (`kw_mma`, 13) | 0 | 24 | F16 -> F16 only in HIP | development, off in releases (`BUILDING.md:205-208`) |
| RR (preset F) | `rr_block` 10, `rr_step.hlsli` 12, `rr_step_y.hlsli` 29, `rr_mlp_step.hlsli` 6, `rr_block_c32` 9, `head/dx12/rr_d0r.hlsl` 20 (`clean/rr_f/`) | 86 | 0 | IU8 format 0xC, 1 line (`rr_step_i.hlsli:61`) | shipped |
| FG | `fg_mconv.hlsl:53`, `fg_mpair.hlsl:134` | 0 | 2 | 0 | groundwork |

By instructions rather than lines, M does about 98 % of its products on FP8 (inferred from a C=96 window: about 432
FP8 against 8 F16 products, `clean/FORMATS.md:85-88`; lane 8's MAC count gives 1.5 % F16 at 1080p Quality). Note:
`clean/k/kernels/k_block8.h` runs K's widest products on FP8 WMMA in one variant (`k_block8.h:5-10`), so K is not
F16-only in every build. Dynamic counts per frame are unknown until counted in RGA ISA (`clean/dx12/build_dx12.py --isa`).

Conversions. The driver's `wg_ags_convert` (0x32) appears in 32 HLSL files; FP8 packing through it
(`wg_pack_q8*_ags`, `rr_q8`, `sw_pack_q8`) in 31. Every one has a software twin already: `wg_pack_q8` / `_nc` /
`_inner` (`clean/dx12/hlsl/wg_platform.hlsli:181-193`), `wg_unpack_fp8` (`:195-203`) behind `SW_UNPACK_AGS`
(`clean/dx12/kernels/swin_common.hlsli:78-79`, default 1), `wg_f2h` (`clean/dx12/hlsl/wg_math.hlsli:88-97`). The GELU
lookup ends in `v_cvt_pk_fp8_f32` (`clean/dx12/hlsl/wg_gelu_lut.hlsli:5-14`); its "identical for all 2^32 inputs"
proof was done on gfx12 (`:17-18`).

Where the gfx12 layout is baked in:

| Place | What |
|---|---|
| `clean/dx12/hlsl/wg_ags.hlsli:60` | "wave32 fragment layouts of gfx12's v_wmma_*_w32" |
| `clean/dx12/hlsl/wg_platform.hlsli:206-243` | `wg_frag2` (FP8: dwords 0-1), `wg_frag4` (F16: 0-3), `wg_to_f16_ags` reads 4 dwords; wave32 only |
| `clean/kernels/swin.hip:6-12`, `clean/k/dx12/hlsl/k_common.hlsli:8-12`, `clean/rr_f/head/dx12/rr_d0r.hlsl:5-9` | "D already has the shape of the next B"; token t in lanes t and t+16; Layout L |
| `clean/dx12/kernels/swin_block.hlsl:776-777,848` | D -> A through the transposed product (`vt[dt] = wg_to_f16_ags(accv)` as A of `A V`); also K's `v^T` |
| `clean/runtime/model.cpp:85-89` | runtime FP8 weight packing: byte `h*128 + r*8 + j` = lane 16h + r, K 8h + j |
| `clean/runner/packing.py:16-19,92` | `D_ROW_LAYOUTS` (`gfx12`, and `gfx11` "for probes"); FP8 fragment packing |
| `clean/k/runner/k_pack.py:164-165` | refuses any accumulator layout other than gfx12 |
| `clean/runtime/runtime.cpp:617-621` | `native_direct_io` refuses a non-gfx12 `d_row` |
| `clean/rr_f/kernels/rr_step_i.hlsli:46` | `RrIA`: element i of lane l = D[8 (l >> 4) + i][l & 15] |
| `clean/rr_f/kernels/rr_block_c32.hlsl:28-45` | RR's wave64 route (`W64_DEFAULT = "D0-F,IN0"`, `clean/rr_f/gpu/stage.py:413`): gfx12 wave64 fragments, 1 dword per FP8 operand |
| `clean/kernels/wg_platform.h:23,47-50` | HIP: `WG_WAVE 32`, `_w32_gfx12` builtins |

The emulator already parameterises both layouts (`clean/kernels/emu/wg_emu.cpp:53-77`, `kvariant`, `dvariant` =
`2v + h`). It is the one place a gfx11 accumulator order is exercised in code.

### 3.2 Where FP8 lives in the data

| Model | In the model file | In the kernels | Activations |
|---|---|---|---|
| M, L | float32 tensors; the converter map applies op `fp8` (exact q8) to 56 of M's tensors and 68 of L's (`data/preset-M.v1.map.json:105-117`), so the stored values are on the E4M3 grid | encoded to E4M3 codes at load, `fp8_encode` (`clean/runtime/model.cpp:89`, `clean/runtime/fpconv.h:80`) | E4M3 maps between stages (`clean/FORMATS.md:12-13`) |
| RR | E4M3 codes (dtype `e4m3`, 104 tensors; 153 float16, 44 float32 in `data/preset-F.v1.map.json`) | codes straight into fragments; int8 digit planes for step I | E4M3 (Q8 of 51.1) between products |
| K, FG | float32 holding FP16 values | FP16 fragments | FP16 |

Weight quantization is part of each FP8 model's definition (the references apply `q8`, `clean/reference/swin.py:244`).
No model in the tree carries a per-channel or per-tensor scale (lane 8, section 0). So no RDNA 3 route that keeps the
model may skip `q8`, and any per-channel INT8 weight set is new data, not present in the player's file.

### 3.3 Everything outside the kernels

Gates. Every one says no to an RX 7900 XT, and every one fails closed:

| file:line | Check | 7900 XT today |
|---|---|---|
| `clean/runtime/rdna4.h:16-19` | vendor 0x1002, hardware, name contains "RX 9070" or "RX 9060" | false |
| `clean/runtime/interop/executor_select.h:105` | auto executor: not RDNA 4 -> HIP | HIP chosen |
| `clean/runtime/device.cpp:398,406` | HIP arch must be gfx1200 or gfx1201 | refused; context create fails; FSR front end falls back to AMD's FSR (`clean/fsr/src/config.h:35`) |
| `clean/runtime/device_d3d12.cpp:1367` (borrowed), `:1440-1444` (own) | `rdna4_adapter` | refused before the probe |
| `clean/runtime/device_d3d12.cpp:218,1385-1409` | `native_driver_probe` (FP16 at `clean/dx12/kernels/native_game_helpers.hlsl:26`, FP8 at `:27`), borrowed device only | unreachable |
| `clean/rr_f/runtime/src/rr_runtime.cpp:185-187,214-231,1344` | copy of the name rule, plus PCI ids 0x7550/0x7551/0x7590/0x7591 via D3DKMT | refused; game keeps its denoiser |
| `clean/rr_f/runtime/src/rr_runtime.cpp:1369-1375` | `DoublePrecisionFloatShaderOps` | not reached; unknown on RX 7000 D3D12 |
| `clean/rr_f/runtime/src/rr_runtime.cpp:1258` | `rr_rt_selftest`: uniform operands in every lane, so it cannot see a layout error | not reached |
| `clean/installer/src/wg_core/checks.cpp:30,74,124,536-538` | name "RX 90"/"PRO R97"/"PRO R96"; Wine PCI ids; HIP arch gfx1200/gfx1201 | install blocked unless `/FORCE` |
| `clean/installer/src/app/launcher_ui.cpp:1101` | status label | shows "· HIP", which is wrong for RDNA 3 |
| `clean/runner/build.py:149,481` | build targets gfx1200/gfx1201 only | n/a |
| `clean/tests/test_device_targets.cpp:36` | asserts gfx1100 is refused | must change with any gfx11 HIP support |

Upstream PR #225 (open) would send every non-RX 9000 card straight to the front end's own upscaler and remove the
automatic HIP route in a game (lane 8, section 9). It does not change the analysis; it changes which seam a gfx11 path
reopens.

Per-GPU tuning data:

| Item | RDNA 4 basis | For gfx11 |
|---|---|---|
| `clean/dx12/vgpr_limits.json` (20 entries, 192 and 96) | RGA gfx1201 (`build_dx12.py:293,311`), 1536 VGPRs | same file size on gfx1100; operand registers grow (2.2), so most Swin kernels will exceed 192 (inferred) |
| `clean/runtime/runtime.cpp:174-176` `kTfSlots = 128` | 4 per WGP of an RX 9070 XT | 7900 XT has 42 WGPs; correct but untuned |
| `clean/runtime/runtime.cpp:464` post kernels wave64 | measured faster on gfx1201 | unknown |
| `clean/dx12/README.md` "Driver exceptions" | gfx1201 compiler regrouping, FMA fusion, dropped stores | must be re-probed on the gfx11 compiler |
| LDS: `swin_common.hlsli:15-17` caps at 32 KB; C=64 and C=128 already stage 32 KB (`clean/FORMATS.md:122-123`) | D3D12 limit | F16 weights in LDS would need 64 KB: they cannot fit (inferred). Keep codes in LDS, or read F16 weights from memory (section 5.4) |
| Records (`clean/l/SPEED.md`, `clean/fg/records/*`, `README.md:65`) | RX 9070 XT only | no other card |

Where a second kernel set would be selected. Nothing selects kernels or models by GPU today. Seams, smallest first:
(1) the predicate `wgrt::rdna4_adapter` (`clean/runtime/rdna4.h:16`) becoming a family value, with its copies in
`rr_runtime.cpp:185` and the installer; (2) the manifest: `read_manifest(build + "/dx12/manifest.json")`
(`clean/runtime/device_d3d12.cpp:200`) into the `kernels_` multimap keyed by name, already holding several ports per
name (`:1712`), so a `dx12/gfx11/manifest.json` chosen by family fits; (3) `choose_kernels`
(`clean/runtime/runtime.cpp:936`), with `d_row[16]` already in `WgRtConfig` (`clean/runtime/wg_runtime.h:39`);
(4) packaging: `kernel_sets.json`, `build_dx12.py kernel_set()` (`:233`), an arch key in `vgpr_limits.json`;
(5) model data: `clean/runtime/model_select.h` chooses only between M, K and L.

No GPU CI exists (`.github/workflows/ci.yml:2`); the launcher job mocks HIP as gfx1201
(`clean/installer/tests/mock_hip.c:11`).

## 4. Matrix throughput: documented against measured

Documented values from lane 9 (AMD's ISA guide "ISA3", the ROCm matrix calculator, ACO's cost table, AMD's product
page). Measured values from `probe-vk-valu/RESULTS-VALU.md`: one dispatch of 5376 waves, 10 timed reps, medians,
loops of 4 independent WMMAs into 4 accumulators. Cycles are SIMD-cycles calibrated by a loop of `v_fma_f32` assumed
to issue once per cycle; that gives an effective clock of 2.682 GHz, consistent with the sysfs clock (median 2.506,
range 1.513-2.914 GHz) to within about 7 %. Whole-card rates use the measured nanoseconds directly.

| Quantity | Documented | Measured on an RX 7900 XT (Linux) | Agrees? |
|---|---|---|---|
| `v_wmma_f32_16x16x16_f16` cycles per instruction | 32 (calculator, ACO, GPUOpen "32 clocks") | 33.1 [31.8-34.5]; 32.1 minus the empty-loop cost | yes |
| `v_wmma_i32_16x16x16_iu8` cycles | 32 (ACO: "int8 and (b)f16 have the same performance") | 32.1 [31.1-32.9]; 31.1 minus loop | yes |
| `v_wmma_f16_16x16x16_f16`, BF16, IU4 (16), wave64 forms | 32, 32, 16, 32 (wave64 not stated directly) | not timed; RADV exposes no BF16 cooperative matrix | unknown until measured |
| MAC per CU per clock, F16 / IU8 | 256 / 256 | 247 / 255 (4096 MAC per instruction, 2 SIMDs per CU) | yes |
| Whole card | 51.6 T MAC/s = 103 TFLOPS at the 2.4 GHz boost (AMD product page) | 55.7 T MAC/s F16 (0.0735 ns per 4096 products), 57.5 T IU8 | yes per clock; the card ran above its rated "up to 2400 MHz" |
| Vector work during a WMMA | none: "Can co-execute with VALU: False" (calculator) | times add: 4 WMMA + 64/128/256 FMA take the sum of the parts or 3-7 % more; same for IU8 | yes; now measured |
| Arithmetic | "only round-to-nearest-even", "internally use the DOT instructions" (ISA3 7.9); no order, width or single-rounding statement | inexact for mixed-sign terms, K-order dependent, deterministic (section 6.3) | not contradicted; documents are silent |
| NOP between WMMAs when D feeds A/B | one `v_nop` (ISA3 7.9, LLVM, ACO) | not timed | unknown until measured |
| RX 9070 XT FP8 against RX 7900 XT F16 | 4.0x per CU-clock; 3.77x whole card at boost clocks | the repo records 7.7 cycles per FP8 WMMA on the RX 9070 XT (`clean/k/kernels/k_block8.h:7-8`); 33.1 here: 4.3x per SIMD-cycle (inferred from two cards) | yes |

Where the chip departs from the documents: nowhere on matrix rate. The departures are the clock (the effective
2.68 GHz is above the rated boost, so whole-card figures exceed AMD's 103 TFLOPS by about 8 %), and the arithmetic,
which the documents never specify. A press report of "52 trillion operations" for WMMA on an RX 7900 XT (gamegpu,
lane 6 section 3.1) is half of what was measured here; its counting convention is not stated.

What this means (inferred): F16 and INT8 run at the same matrix speed. INT8 saves operand registers (4 instead of 8
VGPRs) and the conversion work around the instruction (section 5), not matrix rate. The matrix ceiling is about 3.8x
below an RX 9070 XT's FP8, but section 5 shows the glue, not the matrix instruction, sets the FP16 route's cost.

## 5. Shader throughput: documented against measured

### 5.1 Dot-product and FMA paths (4096 products per wave, one 16x16x16 block)

Each kernel does 128 products per lane per iteration with 8 independent accumulators, at 16 waves per SIMD. The loop
bodies were read in ACO's final assembly (`probe-vk-valu/disasm/`).

| Path (instruction ACO emitted) | Documented MAC/SIMD/clk | Measured cycles / 4096 | Measured MAC/SIMD/clk | Whole card | Agrees? |
|---|---|---|---|---|---|
| FP32 FMA (`v_fmac_f32`, VOP2; no VOPD pairs emitted) | 32 single, 64 as VOPD pair | 137.0 [131.6-140.8] | 30 | 13.5 T MAC/s | yes for single issue (calibration basis); dual issue not reached by compiled code |
| FP16 pairs, FP32 accumulate (`v_fma_mix_f32`; no `v_dot2_f32_f16` emitted) | 32-64 | 131.8 | 31 | 14.0 T | yes |
| FP16 `dot(f16vec2)`, FP16 accumulate (`v_fma_f16` + `v_fmac_f16`; no `v_dot2_f16_f16`) | 32 | 159.5 | 26 | 11.6 T | roughly |
| FP16 packed FMA (`v_pk_fmac_f16`) | 64, never dual | 74.9 | 55 | 24.6 T | yes (85 % of 1 per clock) |
| packed int8 dot (`v_dot4_i32_iu8`, signed via `neg_lo`; `clamp` for saturating; `v_dot4_u32_u8`) | 128 (ACO: 1 cycle per instruction) | 62.6 [60.1-64.9] | 65 | 29.5 T | no: 2 cycles per instruction |
| int32 multiply-add (`v_mul_lo_u32` + `v_add3_u32`) | `v_mul_lo_u32` quarter rate | 584.0 | 7 | 3.2 T | yes (128 x 4 + 64 = 576 predicted) |
| for comparison: F16 / IU8 WMMA | 128 | 33.1 / 32.1 | 124 / 128 | 55.7 / 57.5 T | yes |

Where the chip departs from the documents:

- `v_dot4_i32_iu8` issues at half rate (32 instructions in 62.6 cycles, 8 independent chains, 16 waves per SIMD: a
  throughput limit, not latency). ACO's and LLVM's cost models give 1 cycle, and lane 9 derived "dot path = WMMA rate
  for INT8" from them. Measured, the matrix instruction is about 1.9x the best vector path. Caveats: only ACO's code
  was timed, and the cycle unit rests on the assumption that `v_fma_f32` issues once per cycle; the nanosecond figures
  do not depend on it. Whether RDNA 3's dot4 is
  slower per clock than RDNA 2's (AMD lists 512 ops per CU per clock there) is unknown. The press report's "55 trillion
  operations" for dot4 matches the 59 TOPS measured here.
- `v_permlanex16_b32` costs about 1.8 cycles per instruction (8 per tile in 14.7 cycles), against ACO's model of 1.
- AMD's "FP32 Vector 51.6 TFLOPs" assumes VOPD dual issue. ACO did not emit VOPD pairs or `v_dot2` for the GLSL
  written here, so the dual-issue and dot2 paths are untested: unknown until measured with hand-written ISA or a
  compiler that emits them.

### 5.2 Glue around a matrix multiply (per tile: 8 values per lane, 256 per wave)

Correctness first (measured, `probe-vk-valu/RESULTS-VALU.md` section 2): GLSL ports of `wg_e4m3_hw` and `wg_e4m3_sat`,
both E4M3 -> F16 widenings, and two int32 -> int8 requantizers match CPU references written from the definitions on
all 4,194,304 inputs per conversion (every bit pattern class, exact midpoints and their neighbours, specials). The
check was shown able to fail (172,784 mismatches with the tie rule flipped). The two NaN codes 0x7f / 0xff become
finite F16 values (0x3f80 / 0xbf80 in the bit form, +-480 in the exact form), not F16 NaN.

| Item (ACO loop body) | Cycles per tile med [min-max] | Glue cycles (minus harness) |
|---|---|---|
| Harness only (`B_BASE`) | 14.4 [14.2-14.7] | - |
| E4M3 -> F16, bit form (value x 2^-8) | 35.2 [32.4-36.3] | (own lighter harness) |
| E4M3 -> F16, exact (bit form, then `v_pk_mul_f16` by 256) | 40.0 [38.7-41.2] | (own harness) |
| E4M3 -> F16, table in LDS | 44.8 [43.6-46.5] | (own harness) |
| FP32 -> E4M3, RNE and saturation (`e4m3_hw`, about 16 VALU per value) | 147.4 [132.0-155.3] | 133.0 |
| FP32 -> E4M3 (`e4m3_sat`) | 149.4 [140.0-158.6] | 135.0 |
| FP32 -> FP16, hardware (`v_fma_mixlo/hi_f16`: the preceding add fuses into the conversion) | 12.4 [12.1-13.2] | -2.0 (free after an add) |
| FP32 -> FP16, software `wg_f2h` (about 15 VALU per value) | 142.2 [130.6-150.5] | 127.8 |
| int32 -> int8 requantize, textbook | 58.0 [56.3-59.7] | 43.6 |
| int32 -> int8 requantize, magic (`fma` with 1.5 x 2^23, `v_maxmin_f32`, `v_perm`) | 41.7 [41.2-42.9] | 27.3 |
| GELU lookup (2048-slot table in LDS) | 55.3 [54.3-57.2] | 40.9 |
| Half-wave exchange, `subgroupShuffleXor(v, 16)` -> `v_permlanex16_b32` | 14.7 [14.3-15.4] | (no harness) |
| Half-wave exchange by run-time index -> `ds_bpermute_b32` | 46.5 [45.0-48.6] | (no harness) |
| Weights, 16 B per lane, storage buffer (32 KiB, cache-resident) | 8.5 [7.9-10.1] | (address harness) |
| Weights, 16 B per lane, LDS (8 waves per SIMD, LDS-limited) | 15.7 [15.1-16.3] | (same) |

The quantizer to E4M3 costs about 16.6 cycles per value per wave: 4x a whole F16 WMMA for one tile. The old draft
inferred 8-10 VALU per value; measured it is about 16. The hardware FP32 -> FP16 conversion is free after an add or
FMA; the software `wg_f2h` costs as much as the E4M3 quantizer.

### 5.3 Per-layer-step budget (inferred from the measured parts)

One step = one 16x16x16 multiply producing one tile, K = 16, both operands fresh, output quantized and exchanged once.
Components add (section 4, co-execution row). From `probe-vk-valu/RESULTS-VALU.md` section 6:

| Route | Matrix | Glue | Total cycles (ns per SIMD) | Matrix share |
|---|---|---|---|---|
| (1) FP16: widen A + B codes (4 x 35.2) + F16 WMMA + quantize to E4M3 (133.0) + exchange (14.7) | 33.1 | 288.5 | 321.6 (120) | 10 % |
| (2) INT8: int8 operands + IU8 WMMA + requantize (magic, 27.3) | 32.1 | 27.3 | 59.4 (22) | 54 % |
| (2') INT8 plus the same exchange (if its layout needs it) | 32.1 | 42.0 | 74.1 (28) | 43 % |
| (3) No matrix instruction: `v_dot4_i32_iu8` for 4096 products (62.6) + requantize (27.3) | - | 89.9 | 89.9 (34) | - |

Deeper K (inferred, same measured parts). With K = 16n, the quantize and exchange happen once per tile; the widening
happens per WMMA unless widened operands stay in registers. Route (1), both operands widened per WMMA:
n x 173.9 + 147.7; route (2): n x 32.1 + 27.3. Ratio 5.4x at n = 1 and at n = 8. If the activation operand is
widened once and reused across output tiles (as "D is already the next B" allows), route (1) is n x 103.5 + 147.7:
3.5x route (2) at n = 6 (a K = 96 projection), 3.3x if route (2) also exchanges, with the matrix instruction about
26 % of the step. If weights are stored as exact F16 copies in memory and read from cache (about 17 cycles for 32 B per
lane, extrapolated from the 8.5 measured for 16 B), route (1) drops to about n x 50 + 148: about 2x route (2) at
n = 6. The E4M3 quantizer (133 cycles per tile) remains in every FP16 variant. These are throughput sums; real kernels
add address math, LDS staging and occupancy loss from the larger operand registers (2.2), none measured here.

## 6. Numerics

### 6.1 The correctness contract per model

| Model | Contract today | Where |
|---|---|---|
| All D3D12 ports | byte-identical to the HIP kernel on the same RDNA 4 card | `clean/dx12/compare.py:1-6`, `clean/k/dx12/k_port_check.py`, `clean/fg/runner/fg_d3d12_check.py` |
| M, L | spec 09 tolerances against the float64 reference (emulator); L also records its FP8-op model ("exact" or "trunc13") | `clean/tests/test_kernels_emu.py:97-127`, `clean/l/README.md:9-14` |
| K | spec 49: mean abs error <= 2.5e-3, max <= 3e-2 of RMS; I/O stages bit for bit on the GPU | `clean/k/README.md:228-246` |
| RR | bit-exact where Z is supplied, else per-output tolerance, against vectors | `clean/rr_f/gpu/run_all.py` |
| FG | exact path in integers; matrix path explicitly approximate | `clean/fg/README.md` |
| Whole model | MATCHES (100 % equal, or >= 99.99 % with the rest 1 step off and no bias) / CLOSE (>= 99 % equal, every difference <= 2 steps, mean signed difference within +-0.01 steps) / DIFFERENT | `clean/EQUIVALENCE-PROTOCOL.md:8-12` |

No rule is "byte-identical across GPU families". On gfx11 the HIP side of the port check does not exist (gfx12
builtins only), so whichever contract RDNA 3 gets needs a new reference: RDNA 4 dumps, the CPU references, or both.
Upstream issue #34 (open) already asks for an optional, clearly labelled faster Model M mode "within a published
quality limit", with "the default stays exact" (lane 8, section 9).

### 6.2 What RDNA 4's instructions measurably do

The repo records:

| Fact | Evidence |
|---|---|
| FP8 WMMA is not the exact sum rounded once to f32 (H-exact rejected) | `clean/rr_f/gpu/README.md:80-81` (P1 / P1b) |
| Best fit: f32 rounding after C and after each group of 8 products (K 0-7, 8-15). On the `stage_edge` case: exact model 51 differing values, per-8 model 2, per-WMMA 17, groups of 4 19, truncation 52 | `clean/rr_f/head/DESIGN-55.md:171-178`; the emulation script is not committed |
| F16 -> F32 WMMA is not RN32 of the exact sum either (class `random`: 6.55 M of 16.8 M differ) | `clean/fg/records/wmma-probe-2026-11-14.json` |
| F16 -> F16 WMMA rounds its FP16 accumulator to nearest even after every 16-deep step | `clean/k/README.md:294`, `clean/k/FORMATS.md:199` |
| HIP and D3D12 matrix results are byte-identical on one card | `clean/fg/README.md:28`, `wg_platform.hlsli:218-221` |
| IU8 WMMA is exact in int32 below 2^31 (H1') | `clean/rr_f/kernels/rr_step_i.hlsli:35-36`, `clean/rr_f/gpu/probe_i8.py` |
| On real-like data the f32 rounding rarely shows: RR step E vs exact, 72 differing values over 4 stage kinds, 11 kinds identical | `clean/rr_f/gpu/README.md:72-79` |
| D3D12 flushes FP32 denormals; HIP keeps them (accepted difference) | `clean/dx12/README.md:50-51` |

Not committed: the P1b fit output, and any D8 vs D16 comparison, although `rr_p1b_wmma` computes both from identical
values in one launch (`clean/rr_f/kernels/rr_probe.hlsl:51-66`). The per-8 model leaves 2 values unexplained in one
case, so golden bytes are a better target than the model.

### 6.3 What RDNA 3's matrix instructions do (measured on an RX 7900 XT, Linux)

Harness: plain Vulkan compute, `GL_KHR_cooperative_matrix`, one 16x16x16 problem per subgroup, wave32
(`probe-vk/RESULTS.md`). Inputs from `probe-vk/gen.py` with fixed seeds: 131,072 uniform-random E4M3 problems
(`e4_random`), 8,192 P1b-mimic problems in 8 data modes (`p1b`), 8,192 adversarial E4M3 problems (`adv`), specials,
full-range FP16 with subnormals (`f16_full`), K permutations, position shifts, and int8 / uint8 classes. Expected
values are exact (float64 with a TwoSum check for E4M3; Python integers for FP16).

`v_wmma_f32_16x16x16_f16` fed exact E4M3 values:
- Mixed-sign terms lose about one unit at 2^(E-24), E the exponent of the largest term. 1 + (-1) returns -2^-24;
  8 + (-1) returns 7 - 2^-21; a negative C lands above the exact value (8 with C = -1 gives 7 + 2^-20). The loss
  depends on K slot: the pair (1, -0.5) loses 2^-24 in slots 0 and 1 but 2^-25 in slots 0 and 8 (`out/terms.txt`).
  The probe's inference: negation by bit inversion without carry-in, then truncation. Unconfirmed.
- No model fits. The best of 13 (M-g2: round to FP32 after each pair of products, starting from C) still mismatches
  25 % of `p1b` outputs, 37 % of `adv`, 48 % of `e4_random`, 63 % of `f16_full`. M-exact and the repo's RDNA 4 fit M-g8
  mismatch 53 % and 52 % of all 33.5 M `e4_random` outputs. Alignment and truncation variants (23-28 bits, one's
  complement negation, C inside or outside) were tried; the best still mismatched 51,737 of 261,893 p1b C = 0 outputs.
- The errors are small: on `p1b` C = 0 outputs 87 % are within 1 unit of 2^(E-24) and 99.8 % within 4; the largest
  seen is 23 units on `e4_random`, never above 1e-4 of the largest term. Same-sign sums are mostly exact (142 of 66,800
  differ, all 1 unit low).
- Deterministic: three runs of every class bit-identical. Moving every problem one subgroup slot, or permuting A rows
  and B columns, changes 0 of 4,194,304 outputs. Permuting the 16 K slots changes 26-39 % of outputs (all 7
  permutations tried). The K order is part of the arithmetic.
- NaN and inf propagate as IEEE expects; FP16 subnormal inputs are honoured; an all-zero sum with a +0 term gives +0.
  16 of 17 exact cancellations to 0 returned a small negative value. "Every term -0" never arose (untested). Another
  mode register setting is untested (inferred: no effect; LLVM marks gfx11 WMMA `ReadsModeReg = 0`).

`v_wmma_f16_16x16x16_f16` (C = RN16(C)): close to M-g2, rounding to FP16 after every pair of products. It explains all
but 38 of 63,000 full-range FP16 outputs, but about 95 % of E4M3 outputs and 72 % of adversarial ones. This differs
from what the repo records for gfx12's form (one FP16 rounding per 16-deep step, 6.2): RDNA 3 rounds 8 times as
often (inferred from the two fits).

`v_wmma_i32_16x16x16_iu8`: wrapping results equal the exact sum mod 2^32 on all 1,048,576 outputs of both int8 and
uint8 classes. The saturating form differs from "clamp the exact sum" on 2,241 outputs, of which 2,200 overflow int32;
the remainder suggests saturation of an intermediate partial sum (inferred, not investigated). H1' holds on gfx11.

What lane 9 found in the documents (AMD's ISA guide, RDNA 3, section 7.9): "WMMA supports only round-to-nearest-even
rounding for float types" and the instructions "internally use the DOT instructions", whose pseudocode is a chain of
F32 adds. No document claims a single rounding for RDNA 3 WMMA, and none says how K = 16 is split. The measurements
contradict no document. RDNA 4's guide says instead that its WMMA adds "using fused multiply add".

### 6.4 What the errors do after quantization (measured, CPU over the saved outputs)

From `probe-post/POST.md`. D_exact = RN32(C + exact sum); q8 = RNE E4M3 with saturation at 448; RN16 = FP16 RNE. A flip
is a quantized value that differs (+0 and -0 count as equal). No bias add or activation before the quantizer.

| Data | Outputs | q8 flips vs exact (per million) | of which 1 code | RN16 flips (per million) |
|---|---|---|---|---|
| `e4_random` | 33,554,432 | 61 (2) | 61 | 7,119 (212) |
| `p1b`, all 8 modes | 2,097,152 | 945 (451) | 941 | 3,087 (1,472) |
| `p1b`, range over modes | 262,144 each | 4 to 1,030 per million | | 259 to 3,345 per million |
| `adv` (stress set) | 2,097,152 | 71,279 (33,988) | 55,583 | 344,030 (164,046) |
| 7900 XT vs M-g8 (RDNA 4 stand-in), `p1b` | 2,097,152 | 943 (450) | 939 | 3,051 (1,455) |
| exact vs M-g8 (no 7900 XT), `p1b` | 2,097,152 | 18 (9) | 18 | 139 (66) |

Why that rate: ties and cancellation depth. On `p1b`, 13,440 outputs are exact q8 ties and 6 % of them flip; 811 of
the 945 flips are at ties. Away from ties the flip probability is about u x 2^(d-21) for E4M3 (u = 1.0-1.4, d the
cancellation depth in binades), checked against the measured non-tie rates within about 40 %. Uniform random E4M3 has almost no
ties, hence 2 per million.

What follows (inferred):
- Against the RDNA 4 stand-in the RDNA 3 F16 route is as far off (450 per million) as against the exact sum. If M-g8
  describes RDNA 4, exact digit-plane arithmetic would sit about 50x closer to RDNA 4 (9 per million) than the FP16
  route does. M-g8 is a fit with 2 unexplained values, so this needs the replay (R0).
- Scale: at 1e6 values per layer and the `p1b` rate, about 450 differing codes per stage, about 9,000 in a 20-layer
  frame (range 80 to 21,000 across data modes). That fails "byte-identical". At the `p1b` rate a stage is 99.955 %
  equal with every difference one step: CLOSE (MATCHES needs 99.99 %); the quietest data modes (4 per million) would
  rate MATCHES. Whether one-code differences grow through later layers or a temporal history
  is not measured; a CPU replay of a real network with injected one-code errors would answer it.

### 6.5 Carrying E4M3 values in F16

Checked by arithmetic over the 254 finite codes; the widening forms also measured (5.2):

- Every E4M3FN value is an F16 normal number. E4M3 subnormals (m x 2^-9) are above F16's 2^-14 normal limit.
  Products of two such values have at most 8 significant bits and lie in [2^-18, 448^2], exact in FP32. About 0.6 % are
  not F16-representable, so the instruction must not round products to F16. Measured: it does not (two-term results
  and the 99.8 %-within-4-units figure would be impossible otherwise; inferred from 6.3).
- The cheap bit form `(b & 0x7f) << 7 | sign` gives value x 2^-8 and turns E4M3 subnormals into F16 subnormals
  (`clean/rr_f/DESIGN-51.md:158`). Measured: gfx11's F16 WMMA honours F16 subnormal inputs. Whether a problem scaled by
  2^-8 per operand returns exactly the scaled bits of the unscaled problem is unknown until measured (the artefact is
  relative to the largest term, so probably yes; one replay with scaled `p1b` inputs settles it). The exact form (bit
  form, then x 256) costs 5 cycles more per tile.
- NaN codes: both measured widenings map 0x7f / 0xff to finite F16 values. A route that must propagate E4M3 NaN needs
  a select or the LDS table (inferred).

### 6.6 The software quantizer

`wg_e4m3_hw` and `wg_e4m3_sat` (`clean/dx12/hlsl/wg_math.hlsli:153-175`) and the `wg_pack_q8*` software forms reproduce
`v_cvt_pk_fp8_f32` including NaN and overflow; the repo records bit-exactness of the HIP software and hardware
conversions on gfx12 (`clean/kernels/wg_platform.h:131-134`, `clean/runner/selftest.py:13-15`). Measured on an RX 7900
XT (Linux): GLSL ports compiled by ACO are bit-exact on 4,194,304 chosen inputs, at about 16.6 cycles per value.
The D3D12 path compiles the HLSL through AMD's compiler, so the 2^32 sweep (T4) still has to run there before it
counts; the gfx1201 compiler behaviours in `clean/dx12/hlsl/wg_math.hlsli:25-39` ((half) casts toward zero, F16 FMA
fusion, folding into conversions) are the risk. For the next WMMA the quantizer can emit F16 bits directly (quantize to
the E4M3 value in FP32, then `v_cvt_pk_rtz_f16_f32`, exact for E4M3 values); stored maps still need the code byte. A
cheaper exact quantizer is the largest open lever for the FP16 route (inferred from 5.3).

## 7. FP16 route against INT8 routes

### 7.1 Comparison

"Against RDNA 4" means stage outputs of the RDNA 3 route against the same stage on an RX 9070 XT. Cycles per layer step
from 5.3 (K = 16, fresh operands; deeper K in brackets where it changes the picture).

| Route | Output against RDNA 4 (evidence) | Cycles per step | Needs | Risk |
|---|---|---|---|---|
| FP16 WMMA carrying E4M3 values; codes in memory and LDS, widened in registers; software quantizer | Close: about 450 one-code differences per million on `p1b`, 2 on random E4M3 (measured, 6.4); identical only if RDNA 4's FP8 WMMA has the same mixed-sign artefacts and K dependence (unknown until R0) | 322 measured parts (about 100-175 per WMMA plus 148 per tile at deeper K, inferred) | gfx11 fragments, exchange in every chained kernel, gfx11 weight packing, VGPR and LDS re-budget; Windows driver must accept 0x28 fmt 0x5 on gfx11 | Medium: driver acceptance unknown; occupancy; per-frame code differences |
| Single-plane INT8, weights re-quantized per channel, activations calibrated | Different model. Measured on a real Model M (CPU, 7.4): DIFFERENT at every stage with a multiply, under every scale choice tried; network-output SNR 17-18 dB; display PSNR 34-35 dB worst case | 59 measured parts (74 with an exchange) | per-channel weight scales (new data), activation ranges from captured frames, new references, a new quality contract | High for fidelity; temporal drift unmeasured |
| Adaptive exact INT8 digit planes (RR step I generalised: 1 plane at exponent span <= 3, 2 at <= 10, 3 at <= 14) | Exact against the CPU reference by construction (IU8 measured exact on gfx11); against RDNA 4, differs only where RDNA 4 itself is inexact: about 9 per million on `p1b` if M-g8 holds (measured on CPU, 6.4) | about 310-330 (inferred): 3.8-3.9 WMMAs per product measured on a real Model M with probe frames (7.4), plus the E4M3 quantizer (133), digit split and shifts (estimated) | span counts on captured game frames (P1); digit packing for M and L (exists only for RR); a fixup without FP8 WMMA (today's is step X) | Medium: no faster than the FP16 route on this data; real frames may need more planes |
| No matrix instruction, INT8 dot products (`v_dot4_i32_iu8`) | Same numerics as the INT8 route it implements (single plane: different model; planes: exact) | 90 measured parts at one plane | as the INT8 route; no WaveMatrix channel, only SM 6.4 `dot4add_i8packed` | Low on the driver side; half-rate dot4 |
| No matrix instruction, FP32 (`v_fma_mix_f32` or `v_fma_f32`) | Any summation order can be written, so it could reproduce a complete RDNA 4 model; none exists (2 unexplained values). Otherwise close | about 405 (131.8 products + 140.8 widening + 133 quantize, inferred) | a complete gfx12 FP8 model, or acceptance of close | High: slowest; still needs the quantizer |

For K and FG, which are FP16 models, the comparison is simpler: F16 WMMA with FP32 accumulate, no widening, hardware
FP32 -> FP16 conversion free after an add, one exchange per tile: about 33 + 15 = 48 cycles per step (inferred from the
measured parts). Their contracts are tolerances, not byte identity (6.1).

### 7.2 Prior art for running float networks on integer instructions (lane 7)

- Exact integer slices: the Ozaki scheme splits float matrices into integer slices with a shared exponent, multiplies
  slices exactly in an integer accumulator and sums them (2012; on INT8 tensor cores since 2023,
  https://arxiv.org/abs/2306.11975 ; NVIDIA's 2025 extension picks the slice count from the exponent span,
  https://arxiv.org/abs/2511.13778 ). RR step I (`clean/rr_f/kernels/rr_step_i.hlsli:1-33`) is an Ozaki-I scheme
  specialised to E4M3, and `clean/rr_f/gpu/lever_rates.py:55-60` is its span estimator. Ozaki-II (CRT residues) needs a
  per-output reconstruction every step; weak for WrongGPU's per-step rounding (inferred).
- Block floating point (MSFP, OCP MX, https://arxiv.org/abs/2310.10537 ) is the one-plane, approximate form of the same
  idea. Arithmetic: with a per-block power-of-two scale, one IU8 plane holds E4M3 values in the top 4 binades of the
  block exactly and rounds smaller ones by at most 1/240 of the block maximum; two planes are exact down to 11 binades.
- IU4 does not help on gfx11: a signed E4M3 mantissa needs 5 bits, so E4M3 on IU4 never beats IU8 (lane 7, 4.6).
- Post-training INT8 for image-to-image networks: about 0.01-0.3 dB PSNR with a few hundred unlabelled calibration
  images (EDSR, ABPN, XLSR results; lane 7, 3.3). The zero-data methods (DFQ, ZeroQ) rely on batch-norm statistics;
  WrongGPU's models are transformer-like and have none. No PTQ result for a temporal upscaler or denoiser was found.
- FP8 to INT8: Qualcomm converted FP8-E4-trained networks to INT8 naively with no loss on ResNet18 and MobileNetV2 but
  4.5 points on BERT, a transformer with outliers (https://arxiv.org/abs/2303.17951 ). M, L and RR are of the risky
  class (inferred). Qualcomm sells INT8 hardware; treat it as one party's view.
- The only public float-format emulation for games, vkd3d-proton, widens E4M3 to F16 and uses F16 WMMA with an inexact
  fast quantizer: about 3 ms against 0.85 ms native on an RX 9070 XT at 1440p (https://themaister.net/blog/2025/05/ ).
  Its author called accurate round-to-nearest-even in soft-float "borderline unusable"; 5.2 measures an exact one at
  16.6 cycles per value.

### 7.3 How AMD and Intel did it (lane 6)

- AMD ships three FSR 4 models: Full FP8 (RDNA 4), Full INT8 (RX 7000 since FSR 4.1.1, June 2026; RX 6000 and Strix
  Halo since 4.1.2) and Lightweight INT8 (smaller APUs, "less than half the size"). AMD calls the INT8 model "slightly
  different", "fully adapted" and checked for image errors; it published no method, layers or scales. FSR 4 ships only
  as a signed DLL (`Kits/FidelityFX/docs/techniques/super-resolution-ml.md`,
  https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK ; PCGH and OC3D interviews, lane 6, 1.3).
- AMD's table: RX 9070 XT FP8 4K 1316 us, RX 7900 XTX INT8 4K 3070 us, RX 6900 XT INT8 4K 3190 us. Per CU-cycle,
  RDNA 3 INT8 costs about 2.7x RDNA 4 FP8 and 1.2x RDNA 2 INT8 (inferred from rated boost clocks). Scaled to an RX
  7900 XT that is about 2.6x the RX 9070 XT's time. This is the only end-to-end reference for an FP8-class network on
  RDNA 3. It is consistent with 5.3: on RDNA 3 the INT8 format wins on glue, not matrix rate.
- Reviewers (TechSpot, PCGH, Club386) found shipped INT8 on RX 7000 nearly identical to FP8 on RX 9070 XT, slightly
  behind in fine detail at low render resolutions; the 2025 leaked INT8 build was visibly worse. No outlet published a
  numeric metric.
- Intel XeSS: the DP4a path is a "smaller model, less parameters", separately trained from the XMX one
  (https://game.intel.com/wp-content/uploads/2024/02/GameDev2023_XeSS_v1.pdf ).
- What WrongGPU can do without training data: re-quantize weights per channel (no data), calibrate activation ranges
  from captured game frames and its own stage dumps, score an INT8 variant against the FP8 model with the equivalence
  machinery, and keep the FP8 arithmetic exactly with digit planes. What it cannot do: retrain, run QAT against ground
  truth, shrink the network, or claim an INT8 variant is the same model. AMD's result shows an owner-adapted INT8
  network can look like the FP8 one; it does not show a third-party conversion can.

### 7.4 Measured: INT8 forms of Model M (CPU study on a real converted model)

Measured on the CPU with the repo's float64 reference, re-composed so every matrix product passes through one hook
(the re-composition equals the repo's own `encode` / `run_stage` functions bit for bit at every stage). Data: a real
converted Model M (56 matrices, 1,549,568 weights) and, for weights only, a real RR preset F (104 matrices, 2,903,680
weights). Inputs: four cases of `clean/equiv/probes.py` (`pan`, `disocclusion`, `checker`, `exposure_bright`) at 128x72
render, single reset frames with no history. Verdicts from `clean/equiv/compare.py`'s `compare_stage`. Full tables,
scripts and limits: `int8-study/STUDY.md`. Only aggregate numbers were written; no weights or weight-derived arrays.

Schemes. Weights: S1 one scale per tensor; S2 one scale per output row; S3 per row with squared-error-optimal
clipping; S4 one power-of-two scale per 16x16 block, one plane; S5 two planes; S6 the exact digits of
`rr_step_i.hlsli`. Activations: one int8 or uint8 scale per tensor, per token, per 16x16 tile (one plane), two planes.

| Run | Display PSNR dB, worst / mean | Network-output SNR | Worst final verdict | Stages DIFFERENT in isolation (of 19) |
|---|---|---|---|---|
| Reference against itself; FP32 accumulation of exact products (a CPU stand-in for a WMMA with exact products) | identical | exact | MATCHES | 0 |
| One-plane weights (S1 to S4), exact activations | 34.4-35.2 / 45.3-45.9 | 17.5-18.0 dB | DIFFERENT | 18 |
| One-plane weights and one-plane activations (per tensor, per token or per tile) | 34.6-35.5 / 45.3-46.0 | 17.1-18.0 dB | DIFFERENT | 18 |
| Two-plane weights (S5), exact or two-plane activations | identical | exact | MATCHES | 0 |
| Adaptive exact planes (S6) | identical by construction | exact | MATCHES | 0 |

- One plane fails on the weights alone. After the first stage 67-74 % of enc0 features are equal; from enc2 on, 10-20 %
  of FP8 values are equal in the chained run. The 18 stages are every stage with a multiply; the 19th, post, has none.
  Adding int8 activations changes little.
- Why: exponent spans. Only 4 % (M) and 14 % (RR F) of 16x16 weight blocks span 4 binades or fewer, which is what one
  plane holds exactly. Apart from the encoder's input values, no activation tile does. The GELU output feeding W2 is
  the worst operand: 6-10 % of its nonzero values become 0 under one plane, and W2 carries the most products.
- Two planes hold everything: every weight block of both models spans at most 9 binades, every activation tile and
  tensor of M at most 10; two planes cover 11. No block needed a third plane, and no output's integer bound passed
  2^24, so int32 never came near overflow.
- Exact adaptive planes need 3.81 (per-block scales) to 3.92 (per-row scales) WMMAs per original WMMA on this data:
  about 4.0 at Wv, W1, W2, Wp and the attention scores, 3.8 at Wqk, 2.9-3.6 at the merge, expand and dec0 matrices.
  Per-row and per-token scales are free in multiplies (they fold into the one int32 to float conversion per output),
  and they do nearly as well as per-tile scales, which cost shader work per K tile.
- Squared-error-optimal clipping gained nothing over plain per-row scaling (equal to three digits): E4M3 weights have
  no long tail to trade away.

Cost (inferred; cycles per 16x16x16 step, 32.1 per IU8 WMMA plus output glue, from section 5):

| Scheme | WMMAs | Output glue | Cycles | Result on Model M |
|---|---|---|---|---|
| FP16 route | 1 | widen 141 + quantize 133 + exchange 15 | 322 (measured parts) | close to RDNA 4 (6.4); the CPU stand-in above is not a measurement of gfx11 |
| INT8, one plane, row or tensor scales | 1 | requantize 27 (+15 exchange) | 59-74 | DIFFERENT |
| INT8, one plane, block scales | 1 | requantize 27 + about 17 per K tile (+15) | 76-91 | DIFFERENT |
| INT8, two planes on both operands, or adaptive exact | 3.8-3.9 | E4M3 quantize 133 + digit split about 35 + shifts about 18 + exchange 15 (the last three estimated, not measured) | about 310-330 | exact against the reference |

The exact INT8 routes cost what the FP16 route costs because Model M rounds every product's result to E4M3: the
133-cycle software quantizer stays, and the 141 cycles of widening are traded for about three extra WMMAs and digit
handling. The only cheap INT8 route is the one that changes the model's arithmetic.

What this does not show. The probe frames are synthetic, single reset frames; game frames could have wider or narrower
spans, and error build-up through history is untested. No calibration method from 7.2 (AdaRound-style rounding,
per-layer reconstruction on a few hundred frames) was tried; with 10-20 % of FP8 values equal per chained stage, CLOSE
(99 % equal) looks out of reach for any single-plane scheme under the current rules (inferred), so a single-plane INT8
Model M would need its own quality contract, judged on images. RR has weight numbers only; K, L and FG were not on the
machine. For FP16-weight models (K, FG) even a span of 0 needs two planes (11 significant bits), so they are poor INT8
candidates unless re-quantized (inferred from the format).

For new models: `int8-study/int8_study.py <model.wgcm> <map.json>` prints the weight-side table (error per scheme,
exponent-span histograms per row and per 16x16 block, digit counts) for any model file in under a second. A model is
cheap for INT8 when nearly all of its weight rows and activation tokens span 4 binades or fewer, or when its own
definition uses an integer grid between products. Bounded activations alone are not enough: the spread inside a tile
is what counts.

## 8. Per model

### 8.1 Formats and fit (lane 8; ratings inferred)

| Model | Matrix format today | Weights | Activations | Accumulate | INT8 one plane | INT8 digit planes | FP16 on gfx11 |
|---|---|---|---|---|---|---|---|
| M | E4M3 x E4M3, about 98 % of products; FP16 for `A V` | float32 on the E4M3 grid (file holds FP16, `fp8` op) | E4M3, saturating +-448 | FP32 over chains of 4-32 WMMAs (K up to 512) | poor | feasible; cost set by spans | good fidelity, glue-bound cost |
| L | as M, plus FP16 maps `x2` / `y2` | as M | E4M3, two FP16 maps | FP32 in kernels; oracle moving to FP16 accumulate per 32 products ("q7"; `exact` vs `trunc13` not settled, `clean/l/reference/oracle.py:83-93`) | poor | as M | as M |
| K | FP16 x FP16 (one FP8 variant, `k_block8.h`) | FP16 values in float32 containers | FP16, rounded after every op | FP32 default; FP16 per 16-deep step opt-in (`fused16`) | poor | n/a | native, best fit |
| RR (F) | E4M3, 104 tensors | E4M3 codes | E4M3 (Q8); FP16 hidden state | FP32 inside a 32-product step, FP16 per step | fair | built (step I), exact, falls back to step X | good fidelity, glue-bound cost |
| FG | FP16 x FP16 | FP16 | FP16, leaky ReLU | FP16 per 16-product chunk (delivered rule) | poor | n/a | native, best fit |

The tree has no int8 model and no scale tensors. M's weights: 56 `fp8` tensors, 1,549,568 elements; L: 68, 1,586,816;
RR: 104 `e4m3` tensors, 2,903,680 elements; K: 120 `f16` tensors, 1,480,872 elements; FG: 84 `f16` tensors, 1,439,840
(map tallies, lane 8).

### 8.2 Where Model M's time goes (the repo records, `clean/l/SPEED.md`, Cyberpunk 2077, 1080p Quality, RX 9070 XT)

| Step | Model M (ms) | Share | Model L (ms) | Share |
|---|---|---|---|---|
| enc0 | 0.111 | 9 % | 0.503 | 22 % |
| Swin (ten stages) | 0.585 | 50 % | 0.751 | 33 % |
| dec0 | 0.189 | 16 % | 0.345 | 15 % |
| post | 0.289 | 25 % | 0.651 | 29 % |
| total | 1.174 | | 2.250 | |

M is about 58.5 GMAC per frame at 1080p Quality (lane 8's arithmetic from `clean/l/reference/table.py:289-333`).
enc0 and dec0 run near the DRAM floor (`clean/PERF.md`), and post has no matrix work, so a slower matrix path touches
about half of M's frame directly (inferred). On RDNA 3 the quantizer and widening also land in enc0, dec0 and the
stage boundaries, so the non-Swin half does not stay unchanged (inferred). RR is VALU-bound at levels 0-2 already
on RDNA 4 (`clean/rr_f/gpu/README.md`; WMMA share 28-41 % by the design estimate, `clean/rr_f/DESIGN-51.md:340-358`),
so RDNA 3's glue cost hits it harder than its matrix rate does (inferred).

### 8.3 Suitability per model (inferred)

- K: the natural first RDNA 3 target. FP16 operands need no widening; the output conversion is a hardware instruction;
  spec 49 is a tolerance. Use `v_wmma_f32_16x16x16_f16`. RDNA 3's FP16-accumulate form rounds after every pair, 8x as
  often as gfx12's (6.3), and K's `fused16` head max already sits at half its limit on gfx12 (3.5e-2 of 7e-2,
  `clean/k/README.md`). Needs the gfx11 layout for `v^T`, which `k_pack.py:164-165` refuses today.
- FG: FP16 like K. The delivered rule (FP16 accumulator, one rounding per 16 products) is closer to gfx12's
  FP16-accumulate form than to gfx11's; the tier-B path is already "not bit-exact" and gated on the generated frame, so
  F32 accumulate rounded per chunk is the likely gfx11 form. net2 wiring is still OPEN.
- M: the FP16 route keeps the model and gives close output; digit planes give exact-against-reference output at about
  the same cost (two planes per operand on probe frames, 7.4); a single INT8 plane is a new model and rated DIFFERENT
  at every stage in the CPU study.
- L: as M, plus its block form is not settled (PR #218 merged, #223 open), so an RDNA 3 kernel target would move.
- RR: step I already exists; it needs a fixup that does not use FP8 WMMA (today's is step X; the binary64 one,
  `rr_exact_math.hlsli`, `RR_STEP 2`, does not).

### 8.4 Models in progress upstream (lane 8, read-only queries)

Model L (issue #89; code merged, "planned for 0.6.5, off by default" per open PR #223); Model K (issue #15, "returns in
0.6.1", D3D12 ports byte-identical to HIP); frame generation (issue #96, milestone after L and K; net1 wired, net2
OPEN); RR preset F shipped. Issue #28 asks exactly this document's questions; issue #34 asks for an optional
approximate M mode. No other model or preset appears in any issue or PR.

## 9. Porting map

Assumes the FP16 route (option a) for M, L and RR, or K / FG first. Each step has a check before the next. "Now
measured" marks what the Linux probes already answer at instruction level.

| # | Change | Seam (file:line) | Size | Verified by |
|---|---|---|---|---|
| 1 | GPU family value (rdna4 / rdna3 / none) replacing the bool, unified across runtime, RR and installer; gates for players stay closed | `clean/runtime/rdna4.h:16`, `clean/rr_f/runtime/src/rr_runtime.cpp:185,214-231`, `clean/installer/src/wg_core/checks.cpp:30,74,124`, `executor_select.h:105`, `device_d3d12.cpp:1367,1440` | S | `clean/runtime/interop/tests/test_executor_select.cpp`, `clean/tests/test_device_targets.cpp:36`, installer `wg_core_tests.cpp` |
| 2 | gfx11 probe: FP16-only mode and a gfx11 layout table; split `native_driver_probe` so FP8 cannot hide F16 | `clean/runtime/wmma_probe.h:7-22`, `clean/dx12/kernels/native_game_helpers.hlsl:20-32` | S | `clean/runtime/tests/test_wmma_probe.cpp:218-222` (the `2v + h` must-fail layout becomes the must-pass one for gfx11). Layout now measured on Linux (2.3) |
| 3 | gfx11 channel helpers: 8-dword F16 A/B fragments, accumulator in gfx11 order; a build-time `WG_GFX` switch | `clean/dx12/hlsl/wg_ags.hlsli:36-71`, `clean/dx12/hlsl/wg_platform.hlsli:206-243` | M | step 2's probe; `clean/linux/ags_layout.hlsl` with a declared layout |
| 4 | Conversions: F32 -> F16 via public 0x1c or hardware conversion (free after an add, measured); FP8 pack via software `wg_pack_q8*` (bit-exact port measured); `SW_UNPACK_AGS 0`; GELU table through the software quantizer | `wg_platform.hlsli:181-203,223-243`, `swin_common.hlsli:78-79`, `wg_gelu_lut.hlsli:5-14` | S-M | T4 on Windows (2^32 sweep through AMD's compiler); `clean/dx12/gelu_lut/run.py --table`-style proof |
| 5 | FP8 product on F16: widen weight codes and activation codes to F16 in registers (bit form or exact form, measured 35-40 cycles per 8 per lane); `wg_wmma_fp8` becomes a gfx11 wrapper over format 0x5 | `wg_platform.hlsli:219` and its callers | M | instruction-level arithmetic now measured (6.3); emulator with `dvariant` |
| 6 | Accumulator -> B: one half-wave exchange per value after each conversion that feeds a product; make sure it compiles to `v_permlanex16_b32`, not `ds_bpermute_b32` (3.2x slower, measured); D -> A for `A V` and K's `v^T` | `wg_platform.hlsli:103-107`; `swin_block.hlsl:776-777,848`; `clean/k/dx12/hlsl/k_common.hlsli:8-16,56-58`; RR `rr_common.hlsli:26-45` | L | `clean/tests/test_kernels_emu.py` with the gfx11 layout in `wg_emu.cpp`; RGA ISA for the exchange; stage dumps |
| 7 | Weight packing for gfx11 A fragments (16 halves per row; lanes r and r+16 read the same bytes) and the matching K order | `clean/runtime/model.cpp:85-89`, `clean/runner/packing.py:16-19,92`, `clean/k/runner/k_pack.py:164-165`, RR `layout.pack_weights` | M | packer unit tests; probe with one-hot weights |
| 8 | Budgets: re-derive VGPR bounds with RGA `-c gfx1100` (allocation block 24); LDS stays FP8 codes, or F16 weights from memory | `clean/dx12/build_dx12.py:293,311`, `clean/dx12/vgpr_limits.json` (add an arch key) | M | `build_dx12.py` VGPR check |
| 9 | Per-family manifest and kernel choice | `device_d3d12.cpp:200,1712`, `runtime.cpp:617-621,936` | M | `clean/dx12/tests/native_probe_test.cpp`; `test_kernel_sets.py` |
| 10 | Reference for gfx11: RDNA 4 stage dumps as golden, `clean/equiv/compare.py` for verdicts (HIP-vs-D3D12 `compare.py` cannot run) | `clean/equiv/model_m.py`, `clean/equiv/compare.py` | M | T6 |
| 11 | RR: replace `rr_rt_selftest` with the strict probe for gfx11; step I fixup without FP8 | `rr_runtime.cpp:1258`, `rr_step_i.hlsli` | L | `clean/rr_f/gpu/run_all.py` vectors; IU8 exactness now measured on gfx11 (6.3) |
| 12 | Player-facing strings and gates last | `clean/installer/src/app/ui_words.cpp:87` (translation keyed on "RX 9000-series"), `launcher_ui.cpp:1101`, `README.md:23` | S | installer tests, `ui_fixture.ini` |

Models K and FG skip steps 4-5 for FP8 but need 3, 6 and 7. For an INT8 route, steps 4-5 change: requantize (27 cycles
per tile, measured) instead of the E4M3 quantizer, IU8 fragments (4 VGPRs), and new weight data with scales.

## 10. Measurement plan

Status of the old plan: M4 (bit identity of the F16 route) is half done (RDNA 3 side measured, RDNA 4 side missing);
M5 (K permutations) is measured on RDNA 3: the K order changes 26-39 % of outputs, so a pack-time permutation can only
help if RDNA 4 shows the same dependence; M6 (adversarial classes) and M7 (F16 operands outside the E4M3 grid) are
measured on RDNA 3; M9 (throughput) is measured; M11's instruction part (IU8 exactness) is measured. What remains needs
the RX 9070 XT, Windows, or real data.

| # | Test | Decides | Reuses / data | Card |
|---|---|---|---|---|
| R0 | Replay `probe-vk/results/*.npz` on the RX 9070 XT: FP8 WMMA on `A8`, `B8`, `C` in the stored K order, compare bit for bit with `D_f32`; run `probe-vk/models.py` (`analyse.py`'s `fit` takes any D array) on the RDNA 4 output; replay `f16_full` through gfx12's `v_wmma_f32_16x16x16_f16` and `v_wmma_f16_16x16x16_f16` | whether the FP16 route can be byte-identical; a full model of gfx12 FP8 (replacing M-g8); K and FG cross-family identity | layout below; pack per `clean/rr_f/gpu/probe_p1.py` / `rr_probe.hlsl` (gfx12 wave32 fragments, k = 0..7 and 8..15 split across half-waves) | RDNA 4 |
| M0 | P1b on the RX 9070 XT: save the npz (A, B, C, D8, D16) per data mode; count D8 == D16; run `probe_p1b_fit.py` | gfx12's own F16 datapath against its FP8 one; golden data in the repo's own format | `clean/rr_f/gpu/probe_p1b.py`, `rr_probe.hlsl:51-66` | RDNA 4 |
| M1 | RGA offline, `-s dx12 -c gfx1100`, on `native_driver_probe` and `ags_layout.hlsl` main32 split into F16 and FP8 kernels; also read the ISA AMD's compiler emits for `wg_xor16` | whether AMD's compiler accepts 0x28 fmt 0x5 / 0x7 and 0x32 for gfx11; `v_permlanex16_b32` or `ds_bpermute_b32` | `build_dx12.py --isa` with the target changed | either card, Windows (whether RGA's DX12 mode compiles for an absent target is unknown until tried) |
| M2 | `CheckSupport(WaveMatrix 0x1F)`, `CheckSupport(Float8Conversion 0x20)`, `GetInfo`, AGS `floatConversion` bit | the channel's advertised support on RX 7000 | new 30-line host; `AmdExtD3DCreateInterface` | RDNA 3, Windows |
| M3 | `native_driver_probe` with the name gate bypassed in a test build, F16 and FP8 in separate dispatches; also gfx11-shaped input | PSO result; which layout the phase-0 dwords mean; FP8 reject / emulate / garbage | `clean/dx12/tests/native_probe_test.cpp`, `wmma_probe.h` | RDNA 3, Windows |
| P1 | Digit-plane counts: exponent span per (16-output tile, 32-input step) for weights and per (wave, step) for activations, on real frames, M, L and RR. Partly measured (7.4): weights of a real M and RR F fit two planes everywhere; M's activations on synthetic probe frames fit two planes. Captured game frames are still open | the cost of exact digit planes (1x, about 4x or 9x matrix work) and of the bounded two-plane variant | `clean/rr_f/gpu/lever_rates.py` (its results note `results/2026-10-21-body-levers.md` is not in the tree); weights alone need no GPU | RDNA 4 dumps, or CPU |
| Q1 | INT8 forms of Model M on game frames with history (the CPU study in 7.4 used single synthetic reset frames) | whether a calibrated single-plane INT8 model could reach CLOSE; error build-up over frames | `int8-study/` scripts with captured frames (`m_e2e.py`, replace `probe_input()`) | none (CPU) |
| M8 | T4: software E4M3 quantizer and `wg_f2h` over all 2^32 FP32 inputs through AMD's D3D12 compiler for gfx1100, plus the driver-rule probes of `wg_math.hlsli:25-39` | whether the activation path is bit-exact on Windows (the ACO port is) | `clean/runner/selftest.py` item 3, `gelu_lut/run.py` loop | RDNA 3, Windows |
| M10 | RGA `-c gfx1100` VGPR, LDS and scratch for every Swin kernel rebuilt with gfx11 fragments | occupancy against `vgpr_limits.json` | `build_dx12.py` VGPR check | either |
| M12 | T6: Model M stage dumps, RDNA 3 port against RDNA 4, byte compare then equivalence verdict | the port as a whole; whether one-code differences grow through layers and frames | `clean/equiv/model_m.py`, `clean/equiv/compare.py`, `clean/runner/synthetic_model.py` | both, after the port |
| V1 | Remaining timing: wave64 WMMA, IU4 WMMA, the D -> A/B `v_nop` hazard, `v_dual_dot2acc_f32_f16` and wave64 dot2 from hand-written ISA, a full F32 D -> F16 B conversion | the dot2 path's real rate; chained-kernel cost | `probe-vk-valu/` harness | RDNA 3 |
| V2 | FSR 4.1 INT8 capture on an RX 7900 (RGP): instruction mix and per-pass time | whether AMD uses IU8 WMMA or dot4 on RDNA 3; matrix share of a shipped INT8 network | RGP | RDNA 3, Windows |

R0 data. `probe-vk/results/<class>.npz` per class (`e4_random`, `p1b`, `adv`, `specials`, `f16_full`, `kperm`,
`pos_shift`, `pos_perm`, `int_s8`, `int_u8`). Keys (from `probe-vk/RESULTS.md` section 5, checked against the files):

| Key | Shape, type | Meaning |
|---|---|---|
| `A` | [N,16,16] float16 | A of problem p: row m, column k (row-major, M x K) |
| `B` | [N,16,16] float16 | B: row k, column n (row-major, K x N) |
| `C` | [N,16,16] float32 | accumulator, row m, column n |
| `C16` | [N,16,16] float16 | RN16(C), used for the F16-accumulate runs |
| `A8`, `B8` | [N,16,16] uint8 | E4M3 codes of A and B (E4M3 classes only; `A == E4M3(A8)` exactly) |
| `kind` | [N] int8 | sub-kind (P1b mode, adversarial sub-kind, ...) |
| `D_f32` | [N,16,16] float32 | RX 7900 XT: D = A @ B + C via `v_wmma_f32_16x16x16_f16` (run 0; `D_f32_3runs_identical`) |
| `D_f16` | [N,16,16] float16 | RX 7900 XT: `v_wmma_f16_16x16x16_f16` on `C16` (`D_f16_3runs_identical`) |
| `perms` (kperm, [8,16] int64), `mperm` / `nperm` (pos_perm) | | the permutations used |
| `int_*.npz`: `A`, `B` int8/uint8, `C` int32/uint32, `D_wrap`, `D_sat` | | IU8 runs |

D[p, m, n] = C[p, m, n] + sum_k A[p, m, k] * B[p, k, n], with k in the stored order. Keep that order when packing for
gfx12. Raw little-endian `.bin` files with the same layout are in `probe-vk/inputs/` for a C harness. The whole
`probe-vk/` folder is about 2.5 GB (`inputs/`, `out/`, `results/`); `results/` alone is about 0.7 GB and is all a replay
needs. The inputs regenerate from `probe-vk/gen.py` with fixed seeds (101-109 per class).

Order: R0 and M0 cost minutes on the RX 9070 XT and decide whether byte identity is possible at all. M1 and M2 cost
minutes on Windows and can end the WaveMatrix route on RX 7000; the dot4 route does not depend on them. P1 decides
whether exact digit planes stay at two planes per operand on game frames (7.4 found two on probe frames, which puts
them at FP16-route cost). Q1 decides whether a calibrated single-plane INT8 model is acceptable.

## 11. Linux notes

Measured on an RX 7900 XT (Linux): the repo's test `clean/linux/run_wine_tests.py --tests ags-layout` (RADV, Mesa
27.0-devel, GE-Proton 11-11, vkd3d-proton 5b1089c8 + dxil-spirv #362 + the FMad fix, default build options) fails at
`CreateComputePipelineState wave32 0x80070057` with "dxil-spirv: Failed to emit instruction". Without
`-Denable_extended_emulation=true`, vkd3d-proton finds no FP8 cooperative matrix on gfx11, clears cooperative-matrix
support, and rejects every WMMA shader, F16 ones included (`libs/vkd3d/device.c:2332-2440,2660-2677` in vkd3d-proton).
Raw-fragment mode also rejects FP8 without native FP8 (`dxil_ags.cpp:1690,2021-2024` in dxil-spirv). The test's hint,
"vkd3d-proton without dxil-spirv #362?" (`clean/linux/run_wine_tests.py:256`), is misleading on this card: #362 was
present. Because `main32` mixes F16, FP8 and conversions in one PSO and the host stops at the first failure
(`clean/linux/ags_layout_host.cpp:82-84`), that run says nothing about the instructions; the native Vulkan probes do.

Transfers to Windows (hardware facts):
- The instruction set and register layout (2.1-2.3), now also seen in ACO's output on the card.
- The arithmetic of `v_wmma_f32_16x16x16_f16`, `v_wmma_f16_16x16x16_f16` and `v_wmma_i32_16x16x16_iu8` (6.3), provided
  AMD's compiler emits the same instruction with the same K order and denormal mode.
- Instruction rates (sections 4 and 5), including half-rate `v_dot4_i32_iu8` and no WMMA / VALU overlap.

Does not transfer: which instructions AMD's compiler emits for a given HLSL pattern; whether AMD's driver accepts the
channel on gfx11; raw versus tracked mode, `enable_extended_emulation`, `DXIL_SPIRV_CONFIG`, #362; vkd3d-proton's FP8
emulation (LDS transpose for Acc -> B, FP8 rounding "to nearest, ties to zero" after an F16 step, no inf/NaN handling),
which is not bit-exact and must not be used as a reference. `wmma_rdna3_workaround` forces F16 round-toward-zero in
every WMMA shader; WrongGPU needs RNE, so never set it.

## 12. Open questions for the maintainer

1. Which contract must RDNA 3 meet: same bytes as RDNA 4, exact against the reference, CLOSE against RDNA 4, or a new
   quality limit (issue #34's "published quality limit")? It decides between the FP16 route, digit planes and INT8.
2. Can you run R0 (the replay of `probe-vk/results` on the RX 9070 XT) and M0? Everything about byte identity waits on it.
3. Do you have the P1b fit output, a D8 vs D16 count, or `results/2026-10-21-body-levers.md` outside the repo?
4. Is there a complete model of gfx12's FP8 WMMA beyond per-8 (the 2 unexplained values in `DESIGN-55.md:174-177`)?
5. Have you seen `CheckSupport(WaveMatrix)` or any WaveMatrix compile on an RX 7000 with AMD's driver?
6. Does AMD's Windows driver relayout gfx12-shaped fragments on gfx11, or pass native registers through?
7. Does the RX 7000 D3D12 driver report `DoublePrecisionFloatShaderOps` (RR and K's exact input stage need it)?
8. Which cards are in scope: gfx1100/1101 only (1536 VGPRs), or also gfx1102 (RX 7600, 1024 VGPRs, 8 GB)?
9. Should RDNA 3.5 APUs (gfx1150-1153) share the path, and should gfx1170 be kept out of the "gfx11" family?
10. Is a separate model variant (single-plane INT8, AMD's route) acceptable at all, even as an opt-in mode?
11. Is K the first RDNA 3 target, given its FP16 operands and tolerance contract (8.3)?
12. Is RR step I used by any shipped route, or only opt-in?
13. Which identification should win for the family value: DXGI name, PCI id, or the driver's own report?

## 13. Side findings in today's tree

Not about RDNA 3, found while reading. Each is one line to check.

- `clean/installer/src/app/launcher_ui.cpp:1101`: an AMD card that is not RDNA 4 gets the status label "· HIP". An RX
  7900 XT would show it, though HIP refuses gfx1100.
- `clean/installer/README.md:187-188` says gfx1200 "is reported as not supported yet"; `checks.cpp:124` lists gfx1200
  as supported.
- The installer's name filter passes "PRO R97" / "PRO R96" (`checks.cpp:30`); the runtime's `rdna4_adapter`
  (`clean/runtime/rdna4.h:16-19`) accepts only "RX 9070" / "RX 9060". A Radeon PRO card installs and is then refused.
- `clean/rr_f/runtime/src/rr_runtime.cpp:1258`: `rr_rt_selftest` uses the same operands in every lane, so it cannot
  detect a wrong fragment layout.
- `clean/dx12/vgpr_limits.json` `_doc` says the VGPR granule is 8. AMD's ISA guide and LLVM give an allocation block
  of 24 for wave32 on the 1536-VGPR parts (lane 9, II.5). The bounds in use (192, 96) are multiples of both; "3
  workgroups need <= 128" would become 120.
- `clean/linux/run_wine_tests.py:256`: the `ags-layout` failure hint blames a missing dxil-spirv #362. On an RDNA 3
  card the same error appears with #362 present (section 11).
- `clean/dx12/hlsl/wg_platform.hlsli:103-107`: `wg_xor16` is written as a run-time-index read and its comment names
  HIP's `ds_bpermute`. On gfx11 the run-time-index form measured 3.2x slower than `v_permlanex16_b32` under ACO; on
  gfx12 the cost of each is not recorded in the tree.
- `clean/k/kernels/k_block8.h:5-10`: a K variant runs FP8 WMMA, so "K is F16-only" holds for the default kernels, not
  for every build.
