# Lane 4: matrix instructions and the AMD driver channel, RDNA 3 (gfx11) vs RDNA 4 (gfx12)

Scope: public facts from outside the repository (LLVM, Mesa, AMD's SDKs and docs, AMD's open-source driver
compiler LLPC, vkd3d-proton / dxil-spirv), checked against what WrongGPU relies on
(`clean/dx12/hlsl/wg_ags.hlsli`, `clean/dx12/hlsl/wg_platform.hlsli`, `clean/dx12/README.md`, upstream 994f304).
Sources were fetched on 2026-11-14 from the `main` / `master` branches unless a tag is named.

Labels: READ (seen in code or docs, URL given), MEASURED-ELSEWHERE (a third party's measurement, URL given),
INFERRED (my reasoning, shown), UNKNOWN (only a test can settle it; the test is named), RUMOR (unverified report).

## Summary

1. gfx11 has 6 WMMA instructions (F16, BF16, F16-accumulate, BF16-accumulate, IU8, IU4). It has **no FP8 / BF8 WMMA,
   no SWMMAC, no FP8 conversion instructions and no FP8 dot products**. All of those arrive with gfx12 (and with the
   new gfx1170 "RDNA 4m" APUs, which are not RDNA 3). READ, LLVM and Mesa agree.
2. The register layout differs. On gfx11 every lane of a wave32 holds **all 16 K values** of its A row / B column
   (8 VGPRs for F16), and lanes L and L+16 hold the **same** data. gfx12 splits K across the two half-waves:
   4 VGPRs for F16, 2 for FP8, no duplicates. LLVM names the two styles `wmma-256b-insts` ("A and B matrices have
   duplicated data") and `wmma-128b-insts` ("do not have duplicated data"). The accumulator also moved: gfx11 puts
   even rows in lanes 0-15 and odd rows in lanes 16-31, while gfx12 puts rows 0-7 / 8-15 there. A gfx11 F16
   accumulator is unpacked (one half per VGPR, 8 VGPRs), while a gfx12 one is packed (4 VGPRs). READ.
3. WrongGPU's opcodes and codes are AMD's private "WaveMatrix" intrinsics, and every observed value matches AMD's own
   enum: 0x28 = `WaveMatrixMulAcc`, 0x32 = `WaveMatrixCopy`; formats 0x7 / 0x5 / 0x1 = `WMMA_F32_16X16X16_FP8_FP8` /
   `WMMA_F32_16X16X16_F16` / `WMMA_F16_16X16X16_F16`; types 0x26 / 0x24 / 0x14 / 0x1a = F32 accumulator / F16
   accumulator / F16 B / FP8 B. READ (AMD LLPC header). The **public AGS SDK (latest 6.3.1) does not contain any of
   these**, and AMD publishes no list of GPUs that support them.
4. AMD shipped FSR 4.1 on RX 7000 in mid-2026 with a separate **INT8** model. AMD's own SDK table gives 3070 us
   for an RX 7900 XTX at 4K, against 1316 us for an RX 9070 XT running the FP8 model. READ (FidelityFX SDK docs).
5. Whether AMD's Windows driver accepts WaveMatrix opcode 0x28 on RX 7000, and with which register layout, is
   **UNKNOWN**. A short probe on the card settles it (see the last section).

## 1. WMMA instruction lists per generation (LLVM, cross-checked with Mesa)

### Which targets get which feature

| target | products (LLVM AMDGPUUsage) | WMMA feature | FP8 conversion | SWMMAC | FP8 dot (Dot11) |
|---|---|---|---|---|---|
| gfx1100 / 1101 / 1102 / 1103 | RX 7900 XTX/XT/GRE, W7900/W7800; RX 7800/7700; RX 7600; APU | WMMA256b | no | no | no |
| gfx1150 / 1151 / 1152 / 1153 / 1154 (RDNA 3.5) | Radeon 890M, 8060S, 860M, ... | WMMA256b | no | no | no |
| gfx1170 / 1171 / 1172 ("GFX11.7 (RDNA 4m)") | APUs, products "TBA" | WMMA128b | yes | yes | yes |
| gfx1200 / 1201 (RDNA 4) | RX 9060 (XT); RX 9070 / XT / GRE | WMMA128b | yes | yes | yes |
| gfx1030-gfx1036 (RDNA 2) | RX 6000 | none | no | no | no |

READ:
- Feature definitions, `llvm/lib/Target/AMDGPU/GCNFeatures.td` lines 993-999:
  `defm WMMA256bInsts : ... "Has WMMA instructions where A and B matrices have duplicated data"`,
  `defm WMMA128bInsts : ... "Has WMMA instructions where A and B matrices do not have duplicated data"`.
  https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/Target/AMDGPU/GCNFeatures.td
- Feature sets in the same file: `FeatureISAVersion11_0_Common` and `FeatureISAVersion11_5_Common` list
  `FeatureWMMA256bInsts` (lines 2370, 2397). `FeatureISAVersion11_Common` (line 2317) and those two sets have no
  `FeatureFP8ConversionInsts`. `FeatureISAVersion11_7_Common` (lines 2413-2427) and `FeatureISAVersion12`
  (lines 2467-2471) list `FeatureFP8ConversionInsts, FeatureCvtFP8ByteSel, FeatureOCPFP8ConversionInsts,
  FeatureWMMA128bInsts, FeatureSWMMACGfx1200Insts`, and both have `FeatureDot11Insts` ("v_dot4_f32_fp8_fp8, ...").
- Products and the "GFX11.7 (RDNA 4m)" heading: https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/docs/AMDGPUUsage.md
  (processor table, around its lines 945-1170).

INFERRED: gfx1170 is a gfx11-numbered part with the gfx12 matrix design. It is not an RX 7000 card, so a "gfx11"
port must not treat it as RDNA 3. Its WMMA encodings are shared with gfx12 (`VOP3P_Real_WMMA_gfx1170_gfx12`).

### gfx11 (WMMA256b), wave32 and wave64

`llvm/lib/Target/AMDGPU/VOP3PInstructions.td` lines 1748-1767 (pseudos) and 3017-3022 (encodings, VOP3P 0x40-0x45):

| instruction | VOP3P opcode |
|---|---|
| v_wmma_f32_16x16x16_f16 | 0x40 |
| v_wmma_f32_16x16x16_bf16 | 0x41 |
| v_wmma_f16_16x16x16_f16 | 0x42 |
| v_wmma_bf16_16x16x16_bf16 | 0x43 |
| v_wmma_i32_16x16x16_iu8 | 0x44 |
| v_wmma_i32_16x16x16_iu4 | 0x45 |

https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/Target/AMDGPU/VOP3PInstructions.td

### gfx12 (WMMA128b), wave32 and wave64

Same file, lines 2332-2381 (pseudos) and 2741-2788 (encodings, shared by gfx1170 and gfx12):

| instruction | opcode | note |
|---|---|---|
| the six above | 0x40-0x45 | same names and opcodes, new register layout |
| v_wmma_f32_16x16x16_fp8_fp8 / _fp8_bf8 / _bf8_fp8 / _bf8_bf8 | 0x46-0x49 | gfx12 only |
| v_wmma_i32_16x16x32_iu4 | 0x4a | gfx12 only |
| v_swmmac_f32_16x16x32_f16 / _bf16, v_swmmac_f16_16x16x32_f16, v_swmmac_bf16_16x16x32_bf16 | 0x50-0x53 | sparse A, gfx12 only |
| v_swmmac_i32_16x16x32_iu8 / _iu4, v_swmmac_i32_16x16x64_iu4 | 0x54-0x56 | gfx12 only |
| v_swmmac_f32_16x16x32_fp8_fp8 / _fp8_bf8 / _bf8_fp8 / _bf8_bf8 | 0x57-0x5a | gfx12 only |

There is no `v_wmma_f16_16x16x16_fp8` on gfx12; FP8 always accumulates in F32. (The F16-accumulate FP8 forms,
16x16x64/x128, are gfx1250 data-centre parts: lines 2265-2290, 2791-2815.) Lines 2384-2385 add a further detail:
"f16_f16 and bf16_bf16 Intrinsics have imm operand that controls opsel. Used by gfx11, removed in gfx12".

Mesa agrees: `src/amd/compiler/aco_opcodes.py` lines 1250-1271 list the six with `op(gfx11=0x40..0x45)` and the
FP8 / IU4x32 / SWMMAC forms with `op(gfx11_7=0x46..0x5a)`.
https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/amd/compiler/aco_opcodes.py

### Conversions and dot products gated the same way

- `v_cvt_pk_fp8_f32`, `v_cvt_sr_fp8_f32` (VOP3Instructions.td lines 829-857), `v_cvt_f32_fp8`, `v_cvt_pk_f32_fp8`,
  and the `bf8` forms (VOP1Instructions.td lines 690-806, 1277-1286) all require `FP8ConversionInsts` (gfx12,
  gfx1170 and the gfx9.4 data-centre parts), so **none exists on gfx11**. READ.
- `v_dot4_f32_fp8_fp8` and the related forms need `Dot11Insts` (VOP3PInstructions.td lines 917-920), which is gfx12 /
  gfx1170 only. READ.
- Present on both generations: `v_cvt_f16_f32`, `v_cvt_pk_rtz_f16_f32`, `v_dot2_f32_f16` (Dot10), `v_dot4_i32_iu8`
  (Dot8), and `v_dot2_f16_f16` (Dot9), from the `FeatureISAVersion11_Common` list. READ.
  `v_cvt_pk_f16_f32` (packed RNE) is gfx950 / gfx1250 only (`HasCvtPkF16F32Inst`), so it is on neither card.

## 2. Operand shapes and lane layout

### Registers per lane (VGPRs), from LLVM register classes and AMD's matrix calculator

| instruction | gfx11 w32 A/B, C/D | gfx11 w64 A/B, C/D | gfx12 w32 A/B, C/D | gfx12 w64 A/B, C/D |
|---|---|---|---|---|
| f32 <- f16 / bf16 | 8, 8 | 8, 4 | 4, 8 | 2, 4 |
| f16 <- f16 (bf16 <- bf16) | 8, 8 (unpacked, OPSEL half) | 8, 4 | 4, 4 (packed) | 2, 2 |
| i32 <- iu8 | 4, 8 | 4, 4 | 2, 8 | 1, 4 |
| i32 <- iu4 (16x16x16) | 2, 8 | 2, 4 | 1, 8 | 1 (lanes 0-31 only), 4 |
| f32 <- fp8 / bf8 | n/a | n/a | 2, 8 | 1, 4 |

READ:
- LLVM gfx11 profiles, VOP3PInstructions.td lines 1618-1641: `DstRC = !if(!eq(Suffix, "_w32"), VDst_256, VDst_128)`,
  A/B `VRegSrc_256` for F16/BF16 in **both** wave sizes (`VOP_V8F32_V16F16_V16F16_V8F32` for w32,
  `VOP_V4F32_V16F16_V16F16_V4F32` for w64); `VRegSrc_128` for IU8 and `VRegSrc_64` for IU4.
- LLVM gfx12 profiles, lines 2095-2111, for example `F32_F16_WMMA_w32 : [v8f32, v8f16, v8f16, v8f32]`,
  `F32_FP8BF8_WMMA_w32 : [v8f32, v2i32, v2i32, v8f32] // 8xf8`, `F16_F16_WMMA_w32 : [v8f16, v8f16, v8f16, v8f16]`,
  and line 2135: "IU4X16_WMMA_w64 lanes 0-31 will have 8xi4, remaining lanes are ignored".
- AMD Matrix Instruction Calculator (`matrix_calculator.py --detail-instruction`, run locally on the CPU),
  https://github.com/ROCm/amd_matrix_instruction_calculator . It gives the same counts.
- GPUOpen, "How to accelerate AI applications on RDNA 3 using WMMA" (2023): "Each thread holds A_frag and B_frag in
  8 VGPRs for fp16/bf16, 4 VGPRs for iu8, and 2 VGPRs for iu4 regardless of the wave size. C_frag and D_frag requires
  8 VGPRs in wave32 mode and 4 VGPRs in wave64 mode". https://gpuopen.com/learn/wmma_on_rdna3/

### Element-to-lane mapping (AMD Matrix Instruction Calculator output, wave32)

| | gfx11 (`--architecture rdna3`) | gfx12 (`--architecture rdna4`) |
|---|---|---|
| A[i][k], f16 | lane `i` **and** `i+16`; GPR `floor(k/2)`, half `k%2` | lane `16*(floor(k/4)%2) + i`; GPR `2*floor(k/8) + floor(k/2)%2`, half `k%2` |
| B[k][j], f16 | lane `j` and `j+16`; GPR `floor(k/2)` | lane `16*(floor(k/4)%2) + j`; GPR as A |
| A[i][k], fp8 | n/a | lane `16*floor(k/8) + i`; GPR `floor(k/4)%2`, byte `k%4` |
| B[k][j], fp8 | n/a | lane `16*floor(k/8) + j`; GPR `floor(k/4)%2` |
| C/D[i][j], f32 | lane `(16*i)%32 + j`; GPR `floor(i/2)` | lane `16*floor(i/8) + j`; GPR `i%8` |
| C/D[i][j], f16 | lane `(16*i)%32 + j`; GPR `floor(i/2)`, bits [15:0] (OPSEL[2] moves it to [31:16]) | lane `16*floor(i/8) + j`; GPR `floor(i/2)%4`, half `i%2` |
| wave64 A/B | lanes `i, i+16, i+32, i+48` (4 copies) | lane `32*(floor(k/8)%2) + 16*(floor(k/4)%2) + i` |
| wave64 C/D f32 | lane `(16*i)%64 + j`; GPR `floor(i/4)` | lane `32*(floor(i/4)%2) + 16*floor(i/8) + j`; GPR `i%4` |
| cycles (wave32) | f16: 32; iu8: 32 | f16: 16; fp8 / iu8: 8 |
| FLOPs per WGP per clock | f16: 1024; iu8: 1024 ops | f16: 2048; fp8 / iu8: 4096 |
| co-executes with VALU | no | no |

READ (calculator output). The GPUOpen RDNA 3 article says the same: "WMMA on RDNA 3 requires that the contents of
A_frag and B_frag are replicated between lanes 0-15 and lanes 16-31 of the wave in wave32 mode ... In wave64 mode,
data from lanes 0-15 must also be replicated into lanes 32-47 and 48-63". It adds: "If this flag [OPSEL] is set to
true, the elements of C and D are stored in the upper half of the VGPR". The GPUOpen RDNA 4 article says: "In the
AMD RDNA 3 architecture, we needed to duplicate some elements for A and B, but it is removed for RDNA 4 ... Each lane
only needs to load (or store) 8 elements of a matrix", "it does not have backward compatibility", and "C and D
matrices are split into even and odd for lower 16 lanes and upper 16 lanes [on RDNA 3]. Therefore, in order to
convert D matrix to B matrix ... data needs to be shuffled among lanes in RDNA 3. This is not needed for RDNA 4."
https://gpuopen.com/learn/using_matrix_core_amd_rdna4/
Per-CU throughput table in that article: FP16 256 / 512 / 1024, BF16 n/a / 512 / 1024, I8 512 / 512 / 2048
FLOPS per clock per CU for RX 6950 XT / RX 7900 XTX / RX 9070 XT. It has no FP8 row.

### What exactly changed (the redundancy claim)

The claim is confirmed with one correction. On gfx11 the duplication is in A and B only (C/D are not duplicated):

- **A/B, wave32:** gfx11 lane L holds the full 16-element K vector of row/column `L%16`, and lanes L and L^16 are
  identical: 2x storage, 8 VGPRs for F16. gfx12 gives each half-wave half of K: 4 VGPRs, no duplicates. READ.
- **A/B, wave64:** gfx11 keeps 8 VGPRs per lane with 4 copies. gfx12 drops to 2 VGPRs. READ.
- **C/D:** the same size for F32 on both (8 VGPRs in w32), but the rows are distributed differently (gfx11
  interleaves even/odd rows across the half-waves; gfx12 gives the low half-wave rows 0-7 and the high half-wave
  rows 8-15). F16 C/D shrinks from 8 unpacked to 4 packed VGPRs. READ.
- **Consequence for chaining (D of one layer -> B of the next):** on gfx12 the f32 D lane `16*floor(i/8)+j`, GPR
  `i%8` has the same shape as the fp8 B lane `16*floor(k/8)+j` (k = i), so the conversion stays in the lane. The
  vkd3d-proton author confirms this: "On RDNA4, Accum layout is basically same as B layout ... On RDNA3, this
  doesn't work since len(B) != len(Accum)" (https://themaister.net/blog/2025/05/). On gfx11 each lane needs all 16 k
  values for B but holds only 8 rows of D (the other 8 are in lane L^16), so every D -> B step needs a cross-half-wave
  exchange (for example `v_permlanex16` or DPP) and doubles the register footprint. INFERRED from the formulas,
  matching AMD's sentence above.
- **K order inside gfx12 F16 fragments:** the calculator interleaves K in groups of 4 across the half-waves
  (lanes 0-15 hold k 0-3 and 8-11), while AMD's RDNA 4 blog code loads contiguous k 0-7 into the low half-wave.
  Both give the right product because A and B share the same K permutation. INFERRED. WrongGPU already matches
  HIP's packing bit for bit on gfx12 (`clean/dx12/README.md:150-151`), so this only matters for a new gfx11 packing.

## 3. AMD's WaveMatrix shader intrinsics (the channel WrongGPU uses)

### The public AGS SDK does not contain them

- `GPUOpen-LibrariesAndSDKs/AGS_SDK` master = release v6.3.1 (tag 2026-05-27). In
  `ags_lib/hlsl/ags_shader_intrinsics_dx12.hlsl` the opcode list ends at
  `#define AmdExtD3DShaderIntrinsicsOpcode_ShaderRealtimeClock 0x20`. It has no matrix, WMMA, FP8 or WaveMatrix
  symbol. The channel itself is public: `AmdExtD3DShaderIntrinsics_MagicCode 0x5`, phase shift 24, data shift 8,
  `#define AmdExtD3DShaderIntrinsicsSpaceId space2147420894`. READ.
  https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/master/ags_lib/hlsl/ags_shader_intrinsics_dx12.hlsl
- `amd_ags.h` `AGSDX12ExtensionsSupported` has bits up to `shaderClock` ("Radeon Software Version 23.1.1 onwards")
  plus 19 padding bits, and none for matrices. README change log: 6.1 "RDNA3 GPU detection", 6.2 "shader clock
  intrinsics", 6.3 "RDNA4 detection", 6.3.1 "VS2026 support". READ.
  https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/master/ags_lib/inc/amd_ags.h ,
  https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/master/README.md
- So **no public AGS version introduced WaveMatrix**, and AGS has no `extensionsSupported` bit for it. READ.

### Where they are published

- **AMD's open-source driver compiler LLPC**, `compilerutils/include/public/AmdExtD3DShaderIntrinsics.h` (MIT,
  "Copyright (c) 2016-2025 Advanced Micro Devices"). READ.
  https://raw.githubusercontent.com/GPUOpen-Drivers/llpc/dev/compilerutils/include/public/AmdExtD3DShaderIntrinsics.h
- The FidelityFX SDK 2.0 source that AMD published by mistake in 2025 (`Kits/FidelityFX/api/internal/dx12/AmdExtD3D/`:
  `AmdExtD3DShaderIntrinsicsMatrixOps.hlsl`, "Copyright (c) 2023-2025", `...MatrixOpsHelper.hlsl`,
  `AmdExtD3DShaderIntrinsicsApi.h`). The official repo's `main` no longer has that folder (git tree listing,
  2026-11-14); copies survive in forks such as
  https://github.com/Uklosk/FidelityFX-SDK-2.0.0/tree/main/Kits/FidelityFX/api/internal/dx12/AmdExtD3D (third-party
  mirror of AMD code). READ.
- vkd3d-proton / dxil-spirv reimplement them on Vulkan for FSR 4:
  https://github.com/HansKristian-Work/dxil-spirv/blob/master/opcodes/dxil/dxil_ags.cpp and the test header
  https://github.com/HansKristian-Work/vkd3d-proton/blob/master/tests/shaders/sm_advanced/wmma_ags.h . READ.

### The enums (LLPC header) and WrongGPU's values

Opcodes (header lines 121-138): `WaveMatrixMulAcc = 0x28`, `WaveMatrixUavLoad = 0x29`, `UavStore = 0x2a`,
`GlobalLoad = 0x2b`, `GlobalStore = 0x2c`, `LdsLoad = 0x2d`, `LdsStore = 0x2e`, `ElementFill = 0x2f`,
`ElementExtract = 0x30`, `Length = 0x31`, `WaveMatrixCopy = 0x32`, `WaveMatrixFill = 0x33`,
`MatrixSparsityIndexLoad = 0x34`, `MatrixElementWiseArithmetic = 0x35`, `Float8Conversion = 0x36`.

| WrongGPU (`clean/dx12/hlsl/wg_ags.hlsli`) | AMD enum (LLPC header) | match |
|---|---|---|
| `WG_AGS_MMA 0x28u` (line 26) | `AmdExtD3DShaderIntrinsicsOpcode_WaveMatrixMulAcc = 0x28` | yes |
| `WG_AGS_CONVERT 0x32u` (line 27) | `..._WaveMatrixCopy = 0x32` (not `Float8Conversion` 0x36) | yes |
| load immediate `channel \| register << 4 \| operand << 8` (lines 8-10, 50) | `WaveMatrixInOut_ChannelShift 0` (mask 0xf), `SecondRegFlagShift 4` (mask 0xf), `MatRegTypeFlagShift 8` (mask 0xff) | yes |
| operand A = 1, B = 2, C = 3 (line 10) | `WaveMatrixRegType_A_TempReg 1`, `B_TempReg 2`, `Accumulator_TempReg 3`, `RetVal_Reg 0` | yes |
| format `0x7` FP8 x FP8 -> FP32 (line 28) | `WaveMatrixOpcode_WMMA_F32_16X16X16_FP8_FP8 = 0x7` | yes |
| format `0x5` F16 -> FP32 (line 29) | `WMMA_F32_16X16X16_F16 = 0x5` | yes |
| format `0x1` F16 -> F16 (line 30) | `WMMA_F16_16X16X16_F16 = 0x1` | yes |
| type `0x26` F32 accumulator (line 31) | DataFormat `F32 = 0x6` \| MatrixType `Accumulator = 0x2` << 4 | yes |
| type `0x24` F16 accumulator (line 32) | `F16 = 0x4` \| `Accumulator` << 4 | yes |
| type `0x14` F16 B (line 33) | `F16 = 0x4` \| `B = 0x1` << 4 | yes |
| type `0x1a` FP8 B (line 34) | `FP8 = 0xa` \| `B` << 4 | yes |

READ. The type word is `WaveMatrixModifier`: `DataFormatFlagShift 0` (mask 0xf), `MatrixTypeFlagShift 4` (mask 0x7),
`LayoutFlagShift 7` (mask 1), `ShapeShift 8` (mask 0x7), `MatrixTileShift 11` (loads and stores only). WrongGPU's
types have layout 0 and shape 0 = `WaveMatrixShape_16X16`. The MulAcc opcode word: `WaveMatrixOpcode_OpsShift 0`
(mask 0x7f) and `FlagShift 15`.

Full enums for reference:
- DataFormat: I4 0x0, U4 0x1, I8 0x2, U8 0x3, F16 0x4, BF16 0x5, F32 0x6, I32 0x7, U32 0x8, BF8 0x9, FP8 0xa.
- MatrixType: A 0x0, B 0x1, Accumulator 0x2. Shape: 16X16 0x0, 32X16 0x1, 16X32 0x2, 64X16 0x3.
- WaveMatrixOpcode: 0x0-0x13 dense WMMA (BF16, F16 accumulate, F32 <- BF16, F32 <- BF8/FP8 x4 forms, F32 <- F16,
  I32 <- I4/U4/IU4/UI4 and I8/U8/IU8/UI8 16x16x16, I32 <- 4-bit 16x16x32), 0x14-0x27 sparse SWMMA.
- Float8Conversion (0x36): `FP8_2_F32 0`, `BF8_2_F32 1`, `F32_2_FP8 2`, `F32_2_BF8 3`, saturate at bit 8.

So the "data format" enum (F16 = 4, F32 = 6, FP8 = 0xa) is **not** WrongGPU's "format" (0x7/0x5/0x1). WrongGPU's
format is the WMMA opcode enum. INFERRED from the two enums above.

### Generations, support query, and what the header does not say

- The LLPC header has no gfx, Navi or RDNA mention (grep). The opcode enum is the union of both generations' ISA
  (FP8 and SWMMA forms only exist on gfx12, the 4-bit 16x16x16 forms on both), so the header does **not** say
  which formats or fragment sizes a given GPU accepts. READ / INFERRED.
- Support query: `IAmdExtD3DShaderIntrinsics::CheckSupport(AmdExtD3DShaderIntrinsicsSupport opcodeSupport)` and
  `GetInfo(AmdExtD3DShaderIntrinsicsInfo*)` (waveSize / minWaveSize / maxWaveSize), with
  `AmdExtD3DShaderIntrinsicsSupport_WaveMatrix = 0x1F` and `_Float8Conversion = 0x20` (LLPC header lines 260-261;
  `AmdExtD3DShaderIntrinsicsApi.h` lines 14-18 in the mirror). The helper header documents each matrix function with
  "Available if CheckSupport(AmdExtD3DShaderIntrinsicsSupport_WaveMatrix) returned S_OK" and the FP8 conversion
  with "...Support_Float8Conversion". The interface comes from the driver's `AmdExtD3DCreateInterface`
  (`AmdExtD3D.h`), not from AGS. READ.
- The register meaning: vkd3d-proton's author found that "a wave matrix is represented with 8 uints" and that FSR 4
  code that relied on the register layout broke on RDNA 3: "RDNA3 and 4 actually differ. The columns are interleaved
  in very different ways." vkd3d-proton keeps a test whose comment reads "To detect if we're on RDNA3, which will fail
  this test", keyed on `WMMA_MatrixLength(TypeB)`
  (https://github.com/HansKristian-Work/vkd3d-proton/blob/master/tests/shaders/sm_advanced/cs_wmma_layout_assumptions.cs_6_6.hlsl).
  That test's expected values come from vkd3d-proton's Vulkan backend on RADV, not from AMD's driver. READ.
  INFERRED: the 8 dwords WrongGPU passes through phase 0 are the target GPU's native fragment registers. On gfx11
  they would hold gfx11's layout (16 F16 values = 8 dwords per lane for A/B), not gfx12's 4 dwords.

### Which GPUs AMD says support them

AMD publishes no support list for the private WaveMatrix opcodes. Indirect public facts:
- FSR 4 (FP8 model) is built on these opcodes ("amdxcffx64.dll creates normal D3D12 compute shaders ... with a
  metric ton of undocumented AGS magic opcodes", "FSR 4 is based on the WMMA instructions",
  https://themaister.net/blog/2025/05/). READ.
- AMD's 2023 Agility SDK preview driver 23.10.01.014 exposed Microsoft's SM 6.8 WaveMMA (a different, standard HLSL
  path) "on Radeon RX 7000 series graphics cards", with "WaveMMA supports only 16x16 matrices. FP32 is unsupported,
  except as accumulator." That feature was later dropped from SM 6.8.
  https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-MS-AGILITY-SDK-2023-6-711.html . READ.
  INFERRED: AMD's D3D12 driver has had a gfx11 WMMA lowering path at least once.
- AMD's current preview drivers (26.10.07.02, 26.20.12.01) list "Shader Model 6.10 / Linear Algebra Matrix" and
  "Cooperative Vector" under "AMD Radeon RX 9000 series graphics products will support" only, not RX 7000.
  https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-MS-AGILITY-SDK-26-20-12-01.html ,
  https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-MS-AGILITY-SDK-26-10-07-02.html . READ.

## 4. FSR 4 and FP8 networks on RDNA 3: public knowledge

| claim | label | source |
|---|---|---|
| FSR 4.1.1 in FSR SDK 2.3.0: "Added support for AMD FSR Upscaling 4.1 on AMD Radeon RX 7000 Series (AMD RDNA 3 architecture) discrete GPUs" | READ | https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/main/Kits/FidelityFX/docs/whats-new/version_2_3_0.md |
| AMD timings, performance mode, peak clocks: RX 9070 XT Full FP8 4K 1316 us / 1080p 352 us; RX 9060 XT FP8 4K 3433 us; **RX 7900 XTX Full INT8 4K 3070 us**; RX 7800 XT INT8 1440p 2050 us; RX 7600 INT8 1080p 1998 us; RX 6900 XT INT8 4K 3190 us; RX 6600 INT8 1080p 2183 us; Radeon 890M "Lightweight INT8" 1080p 2423 us | MEASURED-ELSEWHERE (AMD's numbers) | https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/main/Kits/FidelityFX/docs/techniques/super-resolution-ml.md |
| ML frame generation and FSR Ray Regeneration still require "an AMD Radeon RX 9000 Series GPU or later" | READ | same repo, `docs/techniques/frame-interpolation-ml.md`, `docs/techniques/denoising.md` |
| FSR 4.1 for RX 7000 arrived in a mid-2026 Adrenalin release (June 2026, earlier than the promised July); RX 6000 promised for 2027 | READ (press) | https://www.pcgamer.com/hardware/graphics-cards/true-to-its-word-amd-brings-fsr-4-1-to-radeon-rx-7000-series-card-owners-with-its-latest-adrenalin-drivers/ , https://www.pcgameshardware.de/FidelityFX-Super-Resolution-Software-277617/Specials/FSR-4-RDNA-3-Release-Test-Benchmarks-1546566/4/ |
| RDNA 3 lacks FP8, so AMD converted the model to INT8 | READ (press, citing AMD) | https://guru3d.com/story/amd-has-expanded-fsr-41-support-to-radeon-rx-7000-gpus/ (summarised by search; page not fetched) |
| PCGH on RX 7900 XTX / 7800 XT / 7600: FSR 4.1 INT8 costs more than FSR 3.1, most in Ultra Performance; slightly less VRAM than FSR 3.1 | MEASURED-ELSEWHERE | PCGH link above |
| 2025 leaked FSR 4 source contained INT8 model files; community DLLs ran FSR 4.0.2 INT8 on RX 7000 via OptiScaler | READ (press) | https://www.tomshardware.com/pc-components/gpus/amd-accidentally-marks-fsr-4-open-source-source-code-reveals-potential-support-for-older-radeon-gpus , https://wccftech.com/leaked-amd-fsr-4-int8-files-helps-users-turn-on-fsr-4-on-rx-7000-on-windows/ |
| 7900 XTX with leaked INT8: "6-7 FPS" lower than FSR 3.1 at 1440p RT (one user) | RUMOR / anecdotal | https://www.techspot.com/news/109495-amd-fsr-4-leak-modders-unlock-upscaling-tech.html |
| Linux: vkd3d-proton runs the FP8 FSR 4 on RDNA 3 by emulating FP8 as FP16 WMMA (`DXIL_SPIRV_CONFIG=wmma_rdna3_workaround`, also forces FP16 round-toward-zero "This hack only makes sense on RDNA3"); "massive dump on performance"; RDNA 4 FP16-emulation path 3 ms vs native 0.85 ms at 1440p on 9070 XT | READ / MEASURED-ELSEWHERE | https://themaister.net/blog/2025/05/ ; `dxil_converter.cpp` lines 7782-7789 and 10199-10205 in https://github.com/HansKristian-Work/dxil-spirv |
| Users: FP8-emulated FSR 4 on RDNA 3 about 12 % fewer FPS than the INT8 4.0.2 DLL | RUMOR (forum) | https://discuss.cachyos.org/t/fsr-4-1-0-with-vanilla-wine-on-rdna3-yes-it-actually-works/23546 |
| 7900 XT on RADV: WMMA peak ~52 TOPS vs dot4 ~55 TOPS | RUMOR (one enthusiast report, not reproduced) | https://en.gamegpu.com/news/zhelezo/modifikatsiya-shejderov-fsr-4-int8-uskorila-videokarty-amd-radeon-rx-7000-pod-linux-na-31 |

INFERRED: AMD's own numbers show the RX 6900 XT (no WMMA) at 3190 us and the RX 7900 XTX at 3070 us for the same
INT8 model at 4K. RDNA 3's IU8 WMMA rate equals its F16 rate (1024 ops/WGP/clock, table in section 2) and roughly
equals RDNA 2's dot4 rate per CU (512 I8 per CU per clock in AMD's table). So the shipped RDNA 3 path either uses
dot-product instructions or gains little from WMMA. Which instructions AMD's INT8 shaders use is UNKNOWN. Settle
it with an RGP / RGA capture of FSR 4.1 on an RX 7900.

For WrongGPU (INFERRED): AMD itself did not port its FP8 network to RDNA 3 as FP8 or as FP16 WMMA. It retrained or
quantised a new INT8 model and accepted ~2.3x the RX 9070 XT time at 4K. Emulating FP8 with FP16 WMMA, the route
vkd3d-proton took, is reported as much slower.

## 5. Other raw differences that matter for a port

- **FP8 data:** no FP8 arithmetic or conversion instructions on gfx11. WrongGPU already has bit-exact software
  FP8 packing (`wg_pack_q8*`, `clean/dx12/README.md:47-48`) and decode (`wg_unpack_fp8`,
  `clean/dx12/hlsl/wg_platform.hlsli:195-203`). Every E4M3 value (at most 448, smallest subnormal 2^-9) is exactly
  representable in F16 (largest 65504, subnormals to 2^-24). INFERRED from the formats, so FP8 -> F16 widening is
  lossless and F16 WMMA can consume FP8 data.
- **Half-precision math:** `v_cvt_f16_f32`, `v_cvt_pk_rtz_f16_f32`, `v_fma_mix*`, `v_pk_fma_f16`, `v_dot2_f32_f16`
  exist on both. READ (LLVM feature lists). No packed-RNE F32 -> F16 instruction on either.
- **Wave size:** in D3D12 the driver picks wave32 or wave64 for compute unless SM 6.6 `[WaveSize(N)]` forces it
  (AMD RDNA 3 deck: "can be controlled with the WaveSize shader attribute in SM6.6",
  https://gpuopen.com/presentations/2023/RDNA3_Beyond-the-current-gen-v4.pdf , from search summary). WrongGPU sets
  it per kernel (`clean/dx12/README.md:115`) and defines the matrix helpers only for wave32
  (`clean/dx12/hlsl/wg_platform.hlsli:206`). gfx11 wave64 WMMA keeps the full 8-VGPR A/B per lane with 4 copies, so
  wave64 brings no register saving there. READ / INFERRED.
- **VGPRs:** at most 256 VGPRs per wave on GFX10-GFX12 in both wave sizes (AMDGPUUsage `compute_pgm_rsrc1` table:
  "GFX10-GFX12 (wavefront size 32) max_vgpr 1..256"). `Feature1536VGPRs` ("1536 physical VGPRs for wave32 ...
  allocation granularity 24", GCNFeatures.td lines 1285-1290) is set for gfx1100, gfx1101, gfx1151 and gfx12, not
  for gfx1102 / 1103 / 1150 / 1152 (which have 1024; INFERRED from the feature being absent). The RX 7900 XT
  (gfx1100) therefore has the same physical VGPR budget as the RX 9070 XT. READ. A gfx11 port of the same tiling
  needs more VGPRs for A/B (8 instead of 4 per F16 fragment, 8 instead of 2 if FP8 data is widened to F16), which
  can break the `vgpr_limits.json` bounds (`clean/dx12/README.md:166-170`). INFERRED.
- **WMMA rate (per WGP per clock, calculator):** gfx11 F16 1024, IU8 1024; gfx12 F16 2048, FP8 / IU8 4096. An FP8
  network moved to F16 WMMA on gfx11 has 1/4 of gfx12's matrix peak per WGP-clock. INFERRED from READ figures.
  The RX 7900 XT has 84 CUs vs the 9070 XT's 64; clock and CU-count scaling is not covered here.
- **WMMA and VALU:** on both generations the calculator reports "Can co-execute with VALU: False". READ.
- **gfx11 opsel:** the gfx11 F16/BF16-accumulate WMMA takes an OPSEL immediate choosing the low or high 16 bits
  of each C/D VGPR. gfx12 removed it (VOP3PInstructions.td lines 2384-2385). READ.
- **RDNA 3.5 (gfx1150-1154):** same WMMA256b set as gfx11, no FP8; `FeatureISAVersion11_5_Common` adds SALU float
  instructions. READ. AMD runs a "Lightweight INT8" FSR model on the 890M (section 4).
- **RDNA 2 (gfx103x):** no WMMA at all (WMMA pseudos are gated on `HasWMMA256bInsts` / `HasWMMA128bInsts`; RDNA 2
  feature sets have neither). AMD's FSR 4.1 INT8 timings include RX 6000 anyway. READ.

## 6. Compatibility table: WrongGPU's needs vs gfx11

| WrongGPU need (where) | gfx12 (works today) | gfx11 hardware | gfx11 via AMD D3D12 WaveMatrix channel |
|---|---|---|---|
| FP8 x FP8 -> F32 MMA, format 0x7 (`wg_wmma_fp8`, `wg_platform.hlsli:219`) | `v_wmma_f32_16x16x16_fp8_fp8` (READ, LLVM) | **no instruction** (READ, LLVM, Mesa). Lossless route: widen to F16 and use F16 WMMA (INFERRED) | UNKNOWN: rejected at compile, emulated, or garbage. Test: probe 3 below |
| F16 x F16 -> F32 MMA, format 0x5 (`wg_wmma_f16`, `wg_platform.hlsli:221`) | `v_wmma_f32_16x16x16_f16`, A/B 4 VGPRs | yes, same name, A/B 8 VGPRs replicated, half rate (READ) | UNKNOWN whether accepted; if accepted the operand layout is gfx11's (INFERRED). Test: probes 1-2 |
| F16 x F16 -> F16 MMA, format 0x1 (`wg_ags.hlsli:30`) | `v_wmma_f16_16x16x16_f16`, C/D 4 VGPRs packed | yes, C/D 8 VGPRs unpacked with OPSEL (READ) | UNKNOWN. Test: probe 2 with format 0x1 |
| F32 -> FP8 conversion, WaveMatrixCopy 0x26 -> 0x1a (`wg_pack_q8_nc_ags`, `wg_platform.hlsli:223-226`) | `v_cvt_pk_fp8_f32` (README:47-48; LLVM gating READ) | **no instruction** (READ). Software path exists in repo (README:47) | UNKNOWN. Even if accepted, B-layout != accumulator layout on gfx11, so `r.v[0..1]` would not be the same elements (INFERRED). Test: probe 4 |
| F32 -> F16 conversion, WaveMatrixCopy 0x26 -> 0x24 (`wg_to_f16_ags`, `wg_platform.hlsli:239-242`) | `v_cvt_f16_f32` per element, 4 packed dwords | `v_cvt_f16_f32` exists (READ) | UNKNOWN. A gfx11 F16 accumulator is 8 unpacked dwords, so reading `r.v[0..3]` would be wrong (INFERRED). Test: probe 4 |
| wave32 gfx12 fragment layout (`wg_ags.hlsli:60`, all ports) | native | **different** (A/B duplicated across half-waves, 16 K per lane; C/D even/odd rows split; READ, calculator) | the driver presumably uses gfx11's native layout; no public source says it converts (INFERRED / UNKNOWN). Test: probe 2 |
| Accumulator -> B reuse without lane exchange (net chaining) | yes (READ, GPUOpen RDNA 4 blog) | **no**, needs a cross-half-wave shuffle (READ, same blog) | UNKNOWN whether WaveMatrixCopy Acc -> B inserts the shuffle on gfx11. Test: probe 4 |
| Support query | CheckSupport(WaveMatrix 0x1F) | n/a | UNKNOWN result on RX 7000. Test: probe 1 |

## Open questions

1. Does AMD's Windows D3D12 driver return S_OK for `CheckSupport(AmdExtD3DShaderIntrinsicsSupport_WaveMatrix)` and
   `..._Float8Conversion` on an RX 7900 XT, and does it compile opcode 0x28 shaders there?
2. If yes, are the phase-0 registers the native gfx11 layout (8 dwords for F16 A/B, duplicated), or does the driver
   accept gfx12-shaped fragments and relayout them?
3. What does the driver do with format 0x7 (FP8) and with WaveMatrixCopy to FP8 types on gfx11: reject, emulate
   (and with which rounding), or produce wrong bits?
4. Is F16 WMMA on FP8-valued inputs bit-identical to gfx12's FP8 WMMA (same products, but are the in-unit
   accumulation order and rounding the same)? WrongGPU's rule is "the same bytes as HIP", so this decides whether
   a gfx11 path can keep the gfx12 HIP reference.
5. Which instructions AMD's shipped FSR 4.1 INT8 shaders use on RDNA 3 (IU8 WMMA or `v_dot4_i32_iu8`)? That shows
   which route AMD found fastest on this hardware.
6. Is there any AMD statement on the WaveMatrix channel's gfx11 behaviour (none found)? The `32X16`, `16X32` and
   `64X16` shapes in the enum are unexplained.

## What to measure on an RX 7900 XT

1. **Support:** create `IAmdExtD3DShaderIntrinsics` via the driver's `AmdExtD3DCreateInterface`, call `GetInfo`
   (wave sizes) and `CheckSupport` with 0x1F (WaveMatrix) and 0x20 (Float8Conversion). Record HRESULTs and the
   driver version.
2. **Layout probe, F16 -> F32 (format 0x5):** fill A with `A[i][k] = i*16 + k` and B with an identity, once packed
   gfx12-style (4 dwords) and once gfx11-style (8 dwords, lanes L and L+16 equal). Read all 8 result dwords per lane
   and decode which (i, j) landed where. Confirm with the RGA ISA dump that `v_wmma_f32_16x16x16_f16` was emitted and
   whether the driver added `v_permlanex16` / DPP moves around it. Repeat with `WaveMatrixLength` (0x31) for
   F16 A, B, accumulator and FP8 B types; RDNA 3 should report a different B length.
3. **FP8 on gfx11 (format 0x7):** does PSO creation fail, does the ISA show an emulation sequence, and are results
   right for exact test values (powers of two, 448, subnormals, NaN)?
4. **Conversions (0x32):** `0x26 -> 0x24`, `0x26 -> 0x14` and `0x26 -> 0x1a` on an index-encoded accumulator: how many
   dwords carry data, which element lands in which lane, and the rounding of exact ties (RNE?) and of NaN /
   overflow, compared with WrongGPU's software `wg_pack_q8` / `wg_f2h`.
5. **Bit-exactness:** run the same FP8-valued 16x16x16 tiles through gfx11 F16 WMMA (widened inputs) and through the
   gfx12 FP8 reference already in the repo's tests; diff every output bit, including sums that cancel and long-K
   chains (accumulation order).
6. **Throughput:** a WMMA-only loop (F16 -> F32, F16 -> F16, and IU8 if useful) and a `v_dot2_f32_f16` /
   `v_dot4_i32_iu8` VALU loop, per CU per clock at a fixed clock, to see whether WMMA beats dot products on this card
   for WrongGPU's shapes.
7. **Wave64:** whether the driver honours `[WaveSize(64)]` with WaveMatrix calls, and the fragment size it uses there.
8. **VGPRs:** RGA VGPR counts of the heaviest Swin / bottleneck kernels rebuilt with gfx11-sized fragments, against
   the `vgpr_limits.json` bounds.
