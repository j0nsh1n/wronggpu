# Lane 5: the Linux translation stack and AMD's matrix intrinsics (gfx12 vs gfx11)

Scope: how vkd3d-proton -> dxil-spirv -> SPIR-V -> Mesa RADV/ACO handles the AGS "magic UAV" channel that
WrongGPU's kernels use (`clean/dx12/hlsl/wg_ags.hlsli:15-26`, space 2147420894 = 0x7FFF0ADE, opcodes 0x28, 0x32, 0x1c).

Labels: READ = read in code/docs, INFERRED = my reasoning (shown), UNKNOWN = needs a test. Nothing was built or run.

## 0. Versions read

| Component | Version | Notes |
|---|---|---|
| dxil-spirv | `58ea644b` (local checkout, branch `rr-wmma-gaps`) | = pinned `114e664e` + 4 commits (below) |
| vkd3d-proton | `5b1089c8` (2026-11-13) | its submodule gitlink pins dxil-spirv `114e664e`; the working tree under `subprojects/` is at `58ea644b` |
| Mesa | raw `main` fetched 2026-11-14 (no commit hash recorded); release notes 23.3, 25.0, 25.1, 25.2 | cites below say "Mesa main" |

**PR #362 ("raw fragments") is NOT merged upstream in this checkout.** READ: `git log origin/master..HEAD` in the
dxil-spirv checkout lists exactly four commits (`b3c1d936` "Support WMMA operands packed by the shader",
`aea0d073` tests, `87464423` constants, `58ea644b` signed int8); only a fork remote branch contains them;
`origin/master` (3b08c737, 2026-11-13) has two newer commits but not these. So the checkout used for this lane is
upstream + PR #362's three code commits + one test commit, which matches `clean/linux/README.md:42-46`
("vkd3d-proton `5b1089c8` plus that pull request's three code commits"). I could not check whether upstream merged
it after my last local fetch (UNKNOWN; `git fetch` was not allowed).

## 1. dxil-spirv: where the channel is recognised and what it becomes

### 1.1 Recognition

- Magic space: `dxil-spirv:opcodes/dxil/dxil_ags.hpp:56` (`AgsUAVMagicRegisterSpace = 2147420894`). Code word decode:
  `dxil_ags.hpp:58-67, 500-514` (magic nibble 0x5 at bit 28, phase at 24, 16-bit immediate at 8, opcode at 0). This is
  the same layout `wg_ags.hlsli:49-54` builds. READ.
- The channel is an `AtomicCompareExchange` on the magic UAV; `analyze_prepass_ags_dxil_atomic_op` / `push_ags_instruction`
  collect the phased calls (`dxil_ags.cpp:3075-3159`). READ.
- Opcode names (`dxil_ags.hpp:75-128`): 0x28 `WaveMatrixMulAcc`, 0x32 `WaveMatrixCopy` (the repo calls it "conversion"),
  0x1c `FloatConversion`, 0x36 `Float8Conversion`, 0x29-0x2e UAV/LDS load/store, 0x2f element fill, 0x30 extract,
  0x31 length, 0x33 fill, 0x34 sparsity index, 0x35 element-wise arithmetic. Anything not handled hits
  `LOGE("Unsupported AGS magic instruction ...")` and returns false (`dxil_ags.cpp:3010-3014`). READ.
- WMMA opcode enum (`dxil_ags.hpp:368-410`): 0x1 F16->F16, 0x5 F16->F32, 0x7 FP8xFP8->F32, 0xc I32 x I8 etc. The
  repo's 0x1/0x5/0x7 match. Type words (`dxil_ags.hpp:320-346, 484-491`): low nibble = format (4 F16, 6 F32, 0xa FP8),
  bits 4-6 = use (0 A, 1 B, 2 accumulator), bits 8-10 = shape. So WrongGPU's 0x26 = F32 accumulator, 0x24 = F16
  accumulator, 0x14 = F16 B, 0x1a = FP8 B decode as intended. READ.

### 1.2 Two translation modes

`analyze_ags_raw_fragments` (`dxil_ags.cpp:3026-3073`, called at `dxil_converter.cpp:9017`) decides per shader:

| Mode | When | What the SPIR-V is |
|---|---|---|
| Tracked (original) | MulAcc/Copy inputs are results of AGS loads/fills, PHIs, loads | `OpCooperativeMatrixKHR` objects moved through `OpCooperativeMatrixLoad/StoreKHR`, `MulAddKHR`; element ops through alloca-GEP rewriting (`dxil_ags.cpp:2386-2495`) |
| Raw fragments (PR #362) | any phase-0 input to MulAcc/Copy is not Constant/PHI/Load/cmpxchg-result, or a constant | operands rebuilt from the 8 dwords with `OpCompositeInsert` into a fresh coopmat; results extracted with `OpCompositeExtract` (`dxil_ags.cpp:1682-1755, 1758-1799`) |

READ. The flag is per shader: one raw call makes every sequence raw (`dxil_ags.hpp:584`, `dxil_ags.cpp:3058`).
Before PR #362 the shader failed with "Failed DXIL opcode 79" (commit message of `b3c1d936`). READ.

INFERRED risk: a kernel whose fragments come only from PHIs/`LoadInst` (e.g. a loop-carried accumulator in registers
that never passes through the buffer ops) is classified as tracked, even though it packs its own dwords. WrongGPU's
test kernel avoids it (values come from `io.Load` = DXIL buffer load); real kernels might not. Worth checking per
kernel on any Linux run.

### 1.3 Opcode 0x28 (multiply-accumulate), raw mode

`emit_wmma_muladd_raw` (`dxil_ags.cpp:1801-1837`): validates the register encoding of the 12 phase-0 calls
(`validate_raw_fragment_registers`, `1664-1679`: channel = i%2, register = i/2, no tile bit), reads the opcode from
the phase-1 immediate, then emits one `OpCooperativeMatrixMulAddKHR`.

| AGS format | Accepted in raw mode | SPIR-V element types | Source |
|---|---|---|---|
| 0x7 FP8xFP8 -> F32 | only if native FP8 (`wmma_fp8`) | A,B `Float8E4M3EXT`, C F32 | `dxil_ags.cpp:1576-1579, 1690` |
| 0x5 F16 -> F32 | yes | A,B F16, C F32 | `1580-1583` |
| 0x1 F16 -> F16 | yes | A,B,C F16 | `1584-1587` |
| 0xc I8 -> I32 (signed) | yes, raw only | int8 / int32 with all four signed flags | `1588-1591, 1827-1833` |
| BF16, BF8, FP8xBF8, I4, U8, U4, IU8, sparse (0x14-0x27) | NO: `get_matmul_operand_formats` returns false | n/a | `1572-1595` |

Per-lane element count is fixed at 8 (`RawFragmentElements`, `dxil_ags.cpp:1599`): per dword 1 F32, 2 F16, 4 FP8/I8
(`1601-1616`). Raw-mode FP8 has no emulation: `build_raw_fragment` returns 0 when the format is FP8 and `wmma_fp8` is
off (`1690-1691`), `emit_wmma_muladd_raw` then returns false, and the whole shader fails to convert. READ.

### 1.4 Opcode 0x32 (copy / convert), raw mode

`emit_wmma_convert_raw` (`dxil_ags.cpp:1997-2090`). It converts element by element inside the lane and never builds a
coopmat ("works at any wave size", comment `1984-1986`). READ. Rules:

- `saturating` must be 0, otherwise return false (`2006-2009`). WrongGPU already never sets it (`wg_ags.hlsli:26-27`).
- Formats: F32, F16, FP8 only; integer formats rejected (`2019-2020`).
- FP8 on either side needs native FP8; otherwise return false (`2021-2024`).
- Shapes of input and output must match (`2012`); only 16x16 anywhere (`184-185`).
- Implementation: `OpBitcast`/`OpCompositeExtract` to elements, `OpFConvert` per element (`1987-1995`), repack to dwords.
- Matrix-use changes (accumulator -> B) are treated as "element i stays element i". The comment says this is what
  happens on RDNA4 where accumulator and B layouts coincide (`1984-1985`). On a GPU where they do not, this is wrong
  (see section 3.3).

Tracked-mode 0x32 (`emit_wmma_convert`, `2092-2138`, `emit_wmma_complex_convert`, `1881-1982`) is the old path used by
FSR4-style shaders. It uses `OpFConvert` on coopmats, `OpCooperativeMatrixConvertNV` only when the NV coopmat2
extension option is set, and otherwise a transpose through a workgroup scratch (`527-633`, comment "Horrifying").

### 1.5 Other intrinsics WrongGPU uses

- 0x1c `FloatConversion` (`dxil_ags.cpp:2764-2790`): only mode `FToF16Near` (immediate 1); anything else returns false.
  SPIR-V: bitcast to F32, `OpFConvert` to F16, bitcast to u16, `OpUConvert` to u32. READ. The code comment says "the
  shader's float controls make the conversion round to nearest even", but the code that would add rounding execution
  modes is under `#if 0` (`dxil_converter.cpp:7760-7778`). So rounding is "whatever the driver default is"
  (RTE on RADV in practice: INFERRED; not tested). One exception, section 2.4.
- 0x36 `Float8Conversion` (not used by WrongGPU, but same data): native path `OpFConvert` to `Float8E4M3EXT`
  (+ `SaturatedToLargestFloat8NormalConversionEXT` when asked); emulated path clamps to +-448, converts F32 -> F16, then a
  helper `ScalarFP16toFP8` (`dxil_ags.cpp:1376-1530`). BF8/E5M2 is not emulated (`1428-1430`: "Not used in the wild").

### 1.6 Failure mode

Any `return false` in the emit functions fails the shader conversion; vkd3d-proton returns `E_INVALIDARG` from
`CreateComputePipelineState` (0x80070057). The repo README records that exact error for stock vkd3d-proton
(`clean/linux/README.md:46-47`, `run_wine_tests.py:256-257`). READ.

## 2. An existing RDNA 3 path (emulation) and its limits

### 2.1 What it is

Yes, upstream has one, aimed at FSR4's shaders (tracked mode). READ in `vkd3d-proton:CHANGELOG.md:157-171`:

> "There is also a quite hacky emulation path of this which relies on int8 and float16 cooperative matrix support,
> which can run on older GPUs at significant performance cost (and some cost to theoretical correctness)."
> "the default 'official' build ... only exposes this feature when the native `VK_KHR_shader_float8` is properly
> supported, i.e. RDNA4+ only. The emulation path is available when building from source with the appropriate build flags."

Switches (all READ):

| Switch | Where | Effect |
|---|---|---|
| meson `-Denable_extended_emulation=true` | `vkd3d-proton:meson_options.txt:6`, `meson.build:21,49-50` (defines `VKD3D_ENABLE_EXTENDED_EMULATION`) | lets `vkd3d_supports_minimum_coopmat_caps` accept uint8 coopmat A/B instead of FP8 (`libs/vkd3d/device.c:2332-2440`, the `#ifdef` at 2380); makes `SupportsAGSExtension(WMMA_FP8)` true without native FP8 (`libs/vkd3d/device_vkd3d_ext.c:673-679`) |
| env `DXIL_SPIRV_CONFIG=wmma_rdna3_workaround` | `dxil-spirv:dxil_converter.cpp:10200-10210`; hashed into the pipeline key (`vkd3d-proton:libs/vkd3d/device.c:10390-10394`) | (a) RDNA3 column-order rewrite of GEPs into 8-element allocas (`dxil_ags.cpp:2386-2418, 2599-2633`); (b) forces RTZ for FP16 in WMMA shaders (`dxil_converter.cpp:7782-7788`); (c) allows matrix-use-changing converts (`dxil_ags.cpp:410-419`) |
| env `DXIL_SPIRV_CONFIG=wmma_conv_hack` | same function | only (c); comment: "works on RADV for now" but breaks SPIR-V validity (`dxil_ags.cpp:425`, commit `83bf67c7`) |

Only one `DXIL_SPIRV_CONFIG` value is parsed (exact string compare, `dxil_converter.cpp:10205-10208`), so the two cannot be combined. READ.

No Proton-side meson option or env var other than these is in the two checkouts. I grepped for `rdna3`, `gfx11`, `fsr4`,
`fp8`, `VKD3D_CONFIG` hits: nothing else relevant (`rdna3` appears only in unrelated descriptor/sampler-feedback code).

### 2.2 How the FP8 emulation works (tracked mode)

When `wmma_fp8` is off (no native `VK_EXT_shader_float8` coopmat), dxil-spirv represents FP8 matrices as **F16 matrices**:
`make_fp8_type` returns F16 for compute values and uint8 for memory (`dxil_ags.cpp:124-132`). So:

1. FP8 x FP8 -> F32 multiply-add becomes an **F16 x F16 -> F32 `OpCooperativeMatrixMulAddKHR`**, i.e. RADV
   `v_wmma_f32_16x16x16_f16` on gfx11. Every E4M3 value is exactly representable in F16, so the multiply is exact; only
   accumulation order differs (as on any hardware). INFERRED from `dxil_ags.cpp:1536-1563` and `1861-1870`.
2. FP8 load: uint8 coopmat load, then a coopmat helper loops over elements doing integer bit tricks (`dxil_ags.cpp:635-660`,
   `spirv_module.cpp` helper `build_fp8_to_fp16`, ~3150-3260): sign-extend to i16, shift left 7, clear bit 14
   (0x4000), bitcast F16, multiply by 256.0 (exponent bias 15-7 = 8, and denormals). Plain ALU, no WMMA.
3. FP16 -> FP8 on store/convert: helper `build_fp16_to_fp8` (`spirv_module.cpp:2970-3060`): multiply by 1/256 (0x1c00),
   add 0x3f, shift right 7, or the sign, truncate to u8. The code and the vkd3d-proton test comment say this is
   "round to nearest, ties to zero", not exact RTNE, and it comes after an F32 -> F16 step, so it is a **double
   rounding** ("isn't 100% theoretically correct in RTE", `vkd3d-proton:tests/d3d12_sm_advanced.c`, commit `261909be`).
   Saturation to +-448 is a separate coopmat helper (`dxil_ags.cpp:385-402`); inf/NaN are not handled
   ("We don't deal with inf/nan in emulation", `1437`).
4. Matrix-use changes (accumulator -> B) go through `OpFConvert` on coopmats plus an LDS round trip
   (`LDSTransposeScratch`, `dxil_ags.cpp:544-633`), unless the NV coopmat2 option is on.
5. `WaveMatrixLength` returns the coopmat length (`689-719`), so on RDNA3 the shader sees larger element counts: tests
   record "RDNA3 pads A and B to double size", expected lengths {16,16,8,4,4,...} instead of {8,8,8,2,2,...}
   (`vkd3d-proton:tests/d3d12_sm_advanced.c:5943-5949`, commit `45f02b2d`). READ.

Stated cost: "significant performance cost (and some cost to theoretical correctness)" (CHANGELOG above). No numbers in
either repo (UNKNOWN). The vkd3d-proton tests loosen their tolerance for it (`emul_slack = 8` ULP,
`tests/d3d12_sm_advanced.c:5559`; "Slight imprecisions in the FP16 work might lead to some errors on RDNA3"). READ.

### 2.3 When it is enabled (all must hold)

1. vkd3d-proton built with `-Denable_extended_emulation=true` (otherwise RDNA3 loses WMMA entirely, see 2.5).
2. `DXIL_SPIRV_CONFIG=wmma_rdna3_workaround` set for FSR4-style shaders (needed by the layout hacks; the vkd3d-proton
   CHANGELOG does not say this, the community posts below do).
3. The application (or Proton's wrapper) uses a DLL that asks for the AGS WMMA extension (`SupportsAGSExtension(WMMA_FP8)`).
4. RADV exposes `VK_KHR_cooperative_matrix` with F16 16x16x16 and uint8 A/B on that GPU (it does on gfx11,
   section 3.1), so `vkd3d_supports_minimum_coopmat_caps` passes. The missing 8-bit C is tolerated for RADV with a warning
   ("assuming it works anyway", `device.c:2431-2439`).

### 2.4 RTZ side effect for WrongGPU

With `wmma_rdna3_workaround` set, every shader that uses WMMA gets `RoundingModeRTZ` for 16-bit floats
(`dxil_converter.cpp:7782-7788`). WrongGPU's F32 -> F16 conversions (0x32 to 0x24, and 0x1c) need round to nearest even
(`wg_ags.hlsli:21`). INFERRED: with that env var the repo's bit-exact F16 cases would differ. The raw path itself needs
no env var, so do not set it for WrongGPU.

### 2.5 Default builds and RDNA3

`vkd3d_supports_minimum_coopmat_caps` (`libs/vkd3d/device.c:2332-2440`): in a normal build `supports_8bit_a/b` are only set
when native FP8 coopmat exists. On gfx11 it does not, so the function returns false and
`device.c:2660-2677` clears `cooperativeMatrix`, `shaderFloat8*` and `KHR_cooperative_matrix`. Later
`d3d12_device_validate_shader_meta` (`device.c:10895-10911`) rejects any shader with the CooperativeMatrix capability
("Missing sufficient features to expose WMMA", `state.c:3191` returns `E_INVALIDARG`). INFERRED consequence: **F16-only
WMMA shaders also fail on gfx11 in an official build.**

### 2.6 Proton / CachyOS / GE side (community and press, not official docs)

| Claim | Source |
|---|---|
| vkd3d-proton 3.0 adds FSR4 via AGS WMMA, RDNA4 native; older GPUs only via a source build | https://gamingonlinux.com/2025/11/vkd3d-proton-3-0-brings-fsr4-support-shader-backend-rewrite-lots-of-bug-fixes-and-performance-upgrades ; https://www.guru3d.com/story/vkd3dproton-30-adds-amd-fsr4-antilag-supports-older-rdna-gpus/ |
| RDNA3 needs vkd3d-proton built with `-Denable_extended_emulation`, the env var `DXIL_SPIRV_CONFIG=wmma_rdna3_workaround`, and AMD's `amdxc64.dll` | https://discuss.cachyos.org/t/fsr-4-1-0-with-vanilla-wine-on-rdna3-yes-it-actually-works/23546 |
| Proton-CachyOS had `PROTON_FSR4_UPGRADE` and `PROTON_FSR4_RDNA3_UPGRADE` (the latter downloads an FSR4 DLL and turns on the RDNA3 WMMA workaround; value can name a version, e.g. `=4.1.0`) | https://discuss.cachyos.org/t/fsr4-on-rdna3-with-proton-cachyos-env-vars-4-0-0-vs-4-0-2-and-where-the-dlls-are-cached/19391 ; https://discuss.cachyos.org/t/proton-fsr4-rdna3-upgrade-not-working/14649 ; https://www.pcgameshardware.de/Linux-Software-26761/News/CachyOS-August-Release-1480471/ |
| A later Proton-CachyOS release (11.0-20260702, per press) removed `PROTON_FSR4_RDNA3_UPGRADE` as the RDNA3 workaround is "mostly no longer needed" and enables FSR 4.1.1 automatically on supported Radeons. Single source, not cross-checked | https://www.pcgameshardware.de/Linux-Software-26761/News/Proton-CachyOS-FSR-4-1-1-Pipewire-Release-1548330/ |
| Proton-GE: no source found for its own variable names (UNKNOWN) | n/a |

Whatever the wrapper does, the mechanism under it is the same emulation in 2.1. FSR4 on RDNA3 there needs AMD's DLL
(community notes say an INT8-style variant), which is a different shader set from WrongGPU's own kernels. Performance
claims in the press ("Qualitaetsplus vs. Leistungsminus") are not numbers I verified (UNKNOWN).

## 3. Mesa RADV / ACO

All from Mesa main as fetched (see section 0), plus release notes.

### 3.1 What RADV exposes, per generation

READ: `src/amd/vulkan/radv_physical_device.c` (main).

| Item | gfx11 (RX 7900 XT = gfx1100) | gfx12 (RX 9070 = gfx1200/1201) |
|---|---|---|
| `VK_KHR_cooperative_matrix` | yes (`radv_cooperative_matrix_enabled`: `gfx_level >= GFX11 && !use_llvm`, line ~189). First in Mesa 23.3 ("VK_KHR_cooperative_matrix on RADV/GFX11+", relnotes 23.3.0.rst:46) | yes since Mesa 25.1 (disabled in 25.0 "radv: disable ... on GFX12", re-enabled in 25.1.0.rst:3317-3319, released 2025-05-07) |
| 16x16x16 F16 A,B -> F32 and -> F16 C | yes (`fill_array_sizes_structs`, ~3740-3756) | yes |
| BF16 | gated by `radv_bfloat16_enabled` | same |
| int8 (u/s A,B; i32/u32 C, saturate) | yes | yes |
| `VK_EXT_shader_float8` and FP8/BF8 coopmat | **no**: `.EXT_shader_float8 = gfx_level >= GFX11_7` (line 936); FP8 props only for `>= GFX11_7` (~3716) | yes: first in Mesa 25.2 ("VK_KHR_shader_float8 on RADV (RDNA4+)", 25.2.0.rst:89, plus "aco: emit float8 wmma") |
| `VK_NV_cooperative_matrix2` conversions | behind a drirc option (`drirc.features.cooperative_matrix2_nv`, line ~198); relnote "advertise ... behind an env var" 25.2.0.rst:2199 | same |

Mesa main has a gfx level `GFX11_7` (= 117, `radv_physical_device.c:2583`) between gfx11 and gfx12 that gets the
RDNA4-style WMMA layout and FP8. A 7900 XT is plain GFX11, so it does not. What GFX11_7 maps to as marketing silicon
is UNKNOWN to me (do not rely on it).

### 3.2 ACO opcodes (READ: `src/amd/compiler/aco_opcodes.py:1250-1271`, `op()` at 316 fills later generations from earlier ones)

| Opcode | gfx11 | gfx11_7 | gfx12 |
|---|---|---|---|
| `v_wmma_f32_16x16x16_f16` (0x40), `_bf16` (0x41), `v_wmma_f16_16x16x16_f16` (0x42), `v_wmma_bf16_16x16x16_bf16` (0x43), `v_wmma_i32_16x16x16_iu8` (0x44), `_iu4` (0x45) | yes | yes (inherited) | yes (inherited) |
| `v_wmma_f32_16x16x16_{fp8,bf8}_{fp8,bf8}` (0x46-0x49), `v_wmma_i32_16x16x32_iu4` (0x4a) | no (-1) | yes | yes (inherited) |
| `v_swmmac_*` (0x50-0x5a) | no | yes | yes |
| `v_cvt_f32_fp8/bf8`, `v_cvt_pk_f32_fp8/bf8` (0x6c-0x6f), `v_dot4_f32_fp8_*` | no | yes | yes |

Selection: `visit_cmat_muladd` (`aco_select_nir_intrinsics.cpp` ~3707-3790) maps the NIR `cmat_muladd_amd` intrinsic
by element type: F16 -> `v_wmma_f32_16x16x16_f16` (32-bit result) or `v_wmma_f16_16x16x16_f16` (16-bit), E4M3xE4M3 ->
`v_wmma_f32_16x16x16_fp8_fp8`, int8 -> `iu8` with neg bits as sign flags. Constant C operands become inline constants.
FP8 conversions: `nir_op_f2e4m3fn`/`_sat` -> `v_cvt_pk_fp8_f32`; saturating variants clamp to +-448 first with
`v_minimummaximum_f32` ("the ISA doc is wrong and Inf isn't clamped", `aco_select_nir_alu.cpp` ~2774-2815);
`nir_op_e4m3fn2f` -> `v_cvt_f32_fp8` / `v_cvt_pk_f32_fp8` (~2819-2830). On gfx11 these opcodes do not exist, hence no FP8
extension there.

### 3.3 How SPIR-V coopmat becomes registers (the layout facts that matter)

READ: header comment and code of `src/amd/vulkan/nir/radv_nir_lower_cooperative_matrix.c` (main).

- **gfx11**: "the A&B matrices need to be replicated, lanes 0..15 are replicated to 16..31 ... A&B matrices are always
  vectors of 16 elements" (lines 12-15, 71-72). C/accumulator: `length = rows*cols/wave_size`, doubled for 16-bit
  elements because each 16-bit element occupies a full VGPR (lines 71-78, "an f16vec16 for 8 fp16 elements").
  Element index to row: `row = base_row + i * (wave_size/16)` for accumulators on gfx11 (`lower_cmat_get_coordinate`,
  ~1280-1295): rows interleave across the two half-waves.
- **gfx11_7 and gfx12**: no replication; per-lane length is `256/wave_size` for every use (lines 58-60); wave32 table at
  lines 17-37: for 32-bit elements `v_n` holds row n in lanes 0-15 and row n+8 in lanes 16-31. A and B use "the C layout"
  with rows swapped consistently, so `MulAdd` results agree.

The register layout of coopmat elements is implementation defined in Vulkan; RADV simply documents its choice.
dxil-spirv's raw mode relies on the gfx12 one.

### 3.4 Does gfx11 support the F16 formats WrongGPU's F16 paths use?

Instruction-wise yes: F16 A,B -> F32 and F16 -> F16 exist on gfx11 and are advertised (3.1, 3.2). Layout-wise, WrongGPU's
fragments (`wg_ags.hlsli:46-47`: "wave32 fragment layouts of gfx12's v_wmma_*_w32"; `run_wine_tests.py:176`) do not
match the gfx11 coopmat layout, see 5.2.

## 4. Per-use table: what happens under this stack

Assumes this lane's checkouts, Mesa as in section 3, raw mode (WrongGPU packs its own fragments). "Native" = one
instruction; "RADV" = RADV/ACO's own conversion.

| Channel use | gfx12 | gfx11 (RX 7900 XT) |
|---|---|---|
| FP8 MMA (0x28, fmt 0x7) | **Native**: `MulAddKHR` on `Float8E4M3EXT` -> `v_wmma_f32_16x16x16_fp8_fp8` (needs RADV >= 25.2). READ | **Rejected in dxil-spirv**: `build_raw_fragment` returns 0 without native FP8 (`dxil_ags.cpp:1690`) -> shader fails -> PSO `E_INVALIDARG`. In *tracked* mode it would be emulated as F16 WMMA (2.2), but raw mode has no such path. Also vkd3d-proton gate (2.5). INFERRED chain |
| F16->F32 MMA (fmt 0x5) | **Native** `v_wmma_f32_16x16x16_f16`; layout = RADV gfx12 layout. READ | Translates (F16 coopmats) *only if* vkd3d-proton accepts coopmat (needs `enable_extended_emulation`, 2.5); instruction exists, but **operand layout differs**, so results would be wrong with gfx12-packed dwords (5.2). INFERRED |
| F16->F16 MMA (fmt 0x1) | **Native** `v_wmma_f16_16x16x16_f16` | Same as above; C is one element per VGPR on gfx11 (3.3), so layout differs too. INFERRED |
| F32 -> FP8 convert (0x32, 0x26 -> 0x1a) | **Native** scalar `OpFConvert` to E4M3 -> `v_cvt_pk_fp8_f32`; no saturate flag allowed (`dxil_ags.cpp:2008`). READ | **Rejected** (raw FP8 needs native FP8, `2021-2024`). Tracked mode emulates via F16 + integer helper, not bit-exact RTNE (2.2) |
| F32 -> F16 convert (0x32, 0x26 -> 0x24) and 0x1c | **Native** scalar `v_cvt_f16_f32` (elementwise, raw) | **Translates**: elementwise `OpFConvert`, no coopmat, no layout dependence (`dxil_ags.cpp:1984-1986`). Correct *if the shader compiles* (it must not contain FP8 ops) and `wmma_rdna3_workaround` is NOT set (RTZ, 2.4). INFERRED |
| FP8 -> F32 convert (0x32, 0x1a -> 0x26) | **Native** `v_cvt_f32_fp8`/`v_cvt_pk_f32_fp8` | **Rejected** in raw mode (same rule). Tracked mode: integer bit-trick helper (exact), cheap. READ |

Compare "rejected" cells to the stock-vkd3d failure seen on gfx12: the same `E_INVALIDARG` at `CreateComputePipelineState`.

## 5. The repo's own Linux test and a prediction for RX 7900 XT

### 5.1 What `ags-layout` checks (READ)

`clean/linux/ags_layout.hlsl` has two entry points built by `dxc -T cs_6_6` and run by `ags_layout_host.cpp` on
`D3D12CreateDevice(nullptr)` (default adapter, no vendor check, `ags_layout_host.cpp:61`):

- `main32` (wave32, one group of 32 lanes): 7 cases in slots 0-5 of one kernel: F16 MMA, FP8 MMA, convert F32 acc -> F16 acc,
  F32 acc -> FP8 B, `FloatConversion` 0x1c, FP8 B -> F32 for all 256 codes.
- `main64` (wave64): `cvt3_f16`-style F32 -> F16 conversion with 3 values per lane.

`run_wine_tests.py:171-229` builds the expected values from the **gfx12 layout** (A lane l = row l%16, k = 8*(l//16)+e;
B column l%16; C/D column l%16, row 8*(l//16)+e). Multiplies must match within 1e-5 relative; conversions bit-exact
(NaN codes: any NaN). `PASS` only if all 7 slots match.

### 5.2 Prediction, RX 7900 XT, current Mesa, vkd3d-proton built from the local checkouts (INFERRED)

Two variants of "built from local checkouts":

**(a) Default meson options.** `vkd3d_supports_minimum_coopmat_caps` returns false on gfx11 (2.5) -> `CreateComputePipelineState`
for `main32` fails with `E_INVALIDARG` (0x80070057). The harness prints
`FAIL ... CreateComputePipelineState 0x80070057 (vkd3d-proton without dxil-spirv #362? use --vkd3d)`; that hint would be
**misleading** (the cause is the gfx11 gate, not missing #362). `main64` might compile but is never reached
(`ags_layout_host.cpp:83-85`, the host stops at the first FAIL).

**(b) `-Denable_extended_emulation=true`.** The device gate passes (F16 16x16x16 and uint8 A/B coopmat exist on gfx11).
`main32` contains an FP8 MMA and F32<->FP8 converts in the same kernel; raw mode rejects them without native FP8
(`dxil_ags.cpp:1690, 2021-2024`) -> the whole kernel fails -> same `E_INVALIDARG` at `main32`. So the test never
reaches a comparison. The wave64-only kernel would compile (F32 -> F16 scalars, no coopmat) and would probably be
correct; the test harness cannot show that because of the early exit.

**Even if FP8 were removed from the kernel** and the F16 MMA compiled, the F16 MMA case would almost certainly be BAD:

- A/B: the raw path fills coopmat elements 0..7 only (`dxil_ags.cpp:1699-1752`, `RawFragmentElements = 8`), but on
  gfx11 A/B have 16 elements per lane and lanes 16-31 are replicas of lanes 0-15 (3.3; test comment "RDNA3 pads A and B
  to double size"). WrongGPU's lanes 16-31 carry K = 8..15 (`run_wine_tests.py:176`), which gfx11 hardware ignores, and
  elements 8..15 would be zero. The product would only cover K = 0..7. 
- C/D: element i of the accumulator is row 2i + lane/16 on gfx11 vs row 8*(lane/16) + i on gfx12 (3.3), so C input and D
  output rows are permuted relative to the expected values.

Predicted outcome table (all INFERRED):

| Case | (a) default build | (b) extended emulation |
|---|---|---|
| all of `main32` | PSO fails (0x80070057) | PSO fails (0x80070057, FP8) |
| `main64` | not reached | would compile; probably correct |

### 5.3 What would make it a fair RDNA 3 probe

1. Report adapter and driver first: print `VkPhysicalDeviceProperties` equivalent (D3D12 adapter description + driver
   version via DXGI) and the vkd3d-proton `SupportsAGSExtension(WMMA_FP8)` / `WMMA_FP8_NATIVE` answers. Fail with a
   specific message when the gate (not #362) is the cause.
2. One compute kernel (PSO) per case, with its own `CreateComputePipelineState` result in the log, so FP8 failing does
   not hide the F16 and convert cases. Keep running the remaining cases after a PSO failure.
3. Make the expected values depend on a declared layout: `gfx12` (current) and `gfx11` (A/B replicated 16-element
   vectors, C rows `2i + lane/16`). Better: add a layout-neutral case (a kernel that multiplies via the tracked path:
   AGS matrix load/store through a UAV) so the test shows whether the *instruction* works, separately from the register
   layout.
4. Add an RDNA 3 emulation variant: run with `DXIL_SPIRV_CONFIG` unset and, separately, `=wmma_rdna3_workaround`, noting
   the RTZ side effect on the F16 conversions (2.4).
5. Use dxil-spirv's own `--full-wmma` flags only for offline SPIR-V dumps; for the run, use a vkd3d-proton build with
   `-Denable_extended_emulation=true` (do not point the existing `--vkd3d` at an official build).
6. Add `D3D12_FEATURE` / wave-size checks: RDNA3 runs wave64 compute fine, but record `WaveLaneCountMin/Max`.

## 6. For the Windows-focused maintainer: hardware facts vs translation-layer details

### Tells you something about the hardware

- RDNA 3 (gfx11) has WMMA for F16/BF16/IU8/IU4 only: `v_wmma_f32_16x16x16_f16`, `_bf16`, `v_wmma_f16_16x16x16_f16`,
  `_bf16_bf16`, `_i32_iu8`, `_iu4` (ACO opcode table, 3.2). The FP8/BF8 WMMA, SWMMAC, and the FP8 convert instructions
  exist only in `gfx11_7`+ in ACO. This matches the ISA: nothing in ACO's table can be invented, since ACO encodes
  exactly what the ISA has. So **no native FP8 matrix multiply on a 7900 XT** (READ in ACO; confirm in the ISA
  document of that lane).
- gfx11 WMMA operand layout: A and B are 16 elements per lane with lanes 16-31 duplicating 0-15; 16-bit accumulators
  use one VGPR per element (Mesa's lowering encodes this from the ISA; 3.3). gfx12 halves A/B to 8 per lane with no
  replication and gives the accumulator rows `8*(lane/16) + i`. These facts mean any kernel that packs registers by
  hand for gfx12 cannot be reused on gfx11 unchanged; this is hardware, not translation.
- Wave size: Mesa notes that cooperative matrices are not transparently run in wave32 on RDNA3-class ("Don't
  transparently use wave32 with cooperative matrices", relnote 23.3.0.rst:845). Wave64 WMMA behaviour is documented in
  the layout comment (3.3). (Hardware detail, but only partly relevant to the AGS test.)
- FP8 emulation precision loss is a consequence of hardware lacking FP8 cvt: F32->FP8 via F16 loses RTNE exactness.

### Translation-layer details only (do not transfer to Windows)

- Raw mode, `wmma_fp8` flag, `enable_extended_emulation`, `DXIL_SPIRV_CONFIG`, `E_INVALIDARG` failures, the
  uint8-as-FP8 representation, the RTZ hack, LDS transpose helper: all dxil-spirv/vkd3d-proton behaviour.
- PR #362 status, which Proton build ships what, `PROTON_FSR4_*` variables: Proton packaging.
- The "RDNA3 workaround" for FSR4's alloca column order is a quirk of FSR4's own shaders, not of the ISA.
- Whether AMD's Windows D3D12 driver accepts the AGS 0x28/0x32 intrinsics on gfx11 at all is **not answerable from
  this lane**: the Linux stack emulates them, the Windows driver is closed. UNKNOWN; settle by running the WMMA
  kernel through the real driver on the 7900 XT.

## Open questions

1. Has dxil-spirv PR #362 been merged upstream since the local fetch? (affects what stock Proton ships)
2. Would upstream accept raw-mode FP8 emulation (FP8 bytes -> F16 coopmat, exact for multiplies)? Currently only
   tracked mode has it.
3. Does the raw path's constant-classification (1.2) misclassify any WrongGPU production kernel as tracked?
4. Exact rounding mode for `OpFConvert` F32 -> F16 on RADV gfx11 with no execution mode set (assumed RTNE).
5. Real performance of the tracked emulation on gfx11 (no numbers in either repo; press is anecdotal).
6. Is `GFX11_7` shipping silicon relevant to the maintainer? (Mesa treats it like gfx12 for WMMA and FP8.)
7. Proton-GE's own FSR4 variable names (no source found).
8. First Mesa release containing `GFX11_7`/FP8 for gfx11_7 (relnotes up to 26.1 contain no hit for that string).

## What to measure on an RX 7900 XT

1. `vulkaninfo`: `VK_KHR_cooperative_matrix` property list (confirm F16, int8, no E4M3), `VK_EXT_shader_float8` absent.
2. vkd3d-proton built twice (default; `-Denable_extended_emulation=true`) with `VKD3D_DEBUG=warn`: log lines
   "Missing sufficient features to expose WMMA" and "8-bit Accumulator type not exposed".
3. The existing `ags-layout` test against both builds: confirm the predicted `0x80070057`.
4. Split cases (5.3): F16->F32 MMA alone, F16->F16 MMA alone, F32->F16 convert alone, `FloatConversion` alone,
   wave64 convert alone; record which compile.
5. `OpCooperativeMatrixLengthKHR` for A, B, C F16/F32 and FP8-as-F16 (expect 16,16,8/16,... per 3.3), using the
   vkd3d-proton test `test_wmma_matrix_length` as reference.
6. A layout dump: feed one-hot dwords through the raw path and read back which (row, column) each lane/element lands in,
   to learn the gfx11 map empirically (confirms 3.3).
7. Tracked-mode FP8 emulation: run vkd3d-proton's own WMMA tests (`d3d12_sm_advanced.c`) built with extended emulation
   and `DXIL_SPIRV_CONFIG=wmma_rdna3_workaround`, record pass/fail and the 8-ULP slack hits.
8. F32 -> F16 rounding: run the 0x1c case with the env var unset and set; count differing dwords vs RTNE.
9. Throughput of F16 `v_wmma_f32_16x16x16_f16` vs the tracked FP8-as-F16 path with the unpack helpers (GPU timestamps).
