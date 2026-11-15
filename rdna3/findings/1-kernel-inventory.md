# Lane 1: how WrongGPU's GPU kernels use the matrix instructions, per model

Source read: upstream main 994f304 (read-only). Paths are relative to the repo root. Labels: READ (seen in code or docs),
MEASURED-IN-REPO (the repo records a measurement), INFERRED (my reasoning, shown), UNKNOWN (needs a test).

Counting method: "call sites" are lines in the HLSL (or HIP) source that call a matrix helper (`wg_wmma_fp8`,
`wg_wmma_f16`, `wg_ags_mma`) or a conversion helper, found with grep. A call site inside an `[unroll]` loop is one site
but many instructions, so these are static source counts, not dynamic instruction counts. A few sites sit in comments or
in generated duplicates (`fused_*.hlsl` are generated variants of the same kernels as `swin_block.hlsl`,
`bneck.hlsl`, `dec1_dec0.hlsl`), so read the tables as "what is in the source", with errors of a few sites.

## 0. Summary

1. Every model that is shipped or in development is either FP8-matrix (M, L, RR) or FP16-matrix (K, FG). No model mixes
   them in a way that matters: M, L and RR do 97-99 % of their matrix work on FP8 and a small attention step on F16.
2. Models K and FG already use only `v_wmma_f32_16x16x16_f16` (F16 in, F32 accumulate) in the D3D12 kernels. They need no
   FP8 instruction. They do depend on the gfx12 register layout (section 2).
3. FP8 appears in three roles: (a) weights stored as E4M3 codes in packed blobs (M, L, RR) built at load from FP32
   tensors (M, L) or read as E4M3 codes straight from `model.wgcm` (RR); (b) activations stored in memory as FP8 maps
   (M, L, RR); (c) the matrix operand format. All FP8 conversions in the fast kernels go through the driver channel
   (`wg_ags_convert`); a software fallback exists in `clean/dx12/hlsl/wg_platform.hlsli:181,195` (about 8 instructions
   per value, README).
4. The bigger RDNA 3 problem is not FP8 but the fragment layout. Almost every kernel is built so that the FP32 result
   (D) of one matrix multiply is directly the B operand of the next one with no lane exchange. That holds on gfx12
   (wave32: each lane holds 8 K values) and does not hold on gfx11 (A and B replicated across the two lane halves,
   16 values per lane). Weights are repacked per layout, but the kernels' dataflow is not (section 2).
5. Already in the repo that helps: an F16 matrix path (K, FG), an exact int8 WMMA path in RR (`WG_AGS_I8_I32`, format
   0xC), software FP8 pack/unpack, a `gfx11` D-row table in the packer, an emulator that can switch layouts, and a
   Linux/Proton path where the driver channel is translated by vkd3d-proton instead of AMD's driver.
6. Hard gates that name RDNA 4: `clean/runtime/rdna4.h:16` (adapter name must contain "RX 9070" or "RX 9060"), the
   create-time probe `clean/runtime/wmma_probe.h`, and `--offload-arch` in `clean/runner/build.py:149,481`.

## 1. Inventory per model

### 1.1 Format codes (READ, `clean/dx12/hlsl/wg_ags.hlsli:25-32`)

| Name | Code | Meaning |
|---|---|---|
| mma FP8 | 0x7 | A, B E4M3 (2 dwords/lane), C/D FP32 (8 dwords) |
| mma F16->F32 | 0x5 | A, B F16 (4 dwords), C/D FP32 |
| mma F16->F16 | 0x1 | A, B F16, C/D F16 (4 dwords) |
| mma I8 (not in the header) | 0xC | `#define WG_AGS_I8_I32 0xCu`, A and B signed int8, int32 accumulate; defined only in `clean/rr_f/kernels/rr_step_i.hlsli:39` |
| convert types | 0x26 F32 acc, 0x24 F16 acc, 0x14 F16 B, 0x1a FP8 B | `wg_ags_convert(m, from, to)`, opcode 0x32 |
| other opcode | 0x1c | FloatConversion RNE F32->F16 (scalar), used at `clean/k/dx12/hlsl/k_out.hlsli:121` |

Wrapper names used by the kernels (READ, `clean/dx12/hlsl/wg_platform.hlsli:206-245`, wave32 only):
`wg_wmma_fp8` (0x7), `wg_wmma_f16` (0x5), `wg_pack_q8*_ags` (F32acc 0x26 -> FP8 B 0x1a), `wg_to_f16_ags`
(0x26 -> 0x24), and `sw_unpack_fp8`/`rr_d8`/`unpack8` (0x1a -> 0x26). Direct uses of other pairs:
F16 acc 0x24 -> F32 acc 0x26 (`clean/dx12/kernels/encoder.hlsl:274`), F16 acc 0x24 -> FP8 B 0x1a
(`clean/rr_f/kernels/rr_fast.hlsli:57,58,69`; `clean/rr_f/head/dx12/rr_d0r.hlsl:476`). The type 0x14 (F16 B) is defined
but I found no call that uses it (READ: grep over `clean/`).

Format 0x1 (F16->F16) has one D3D12 call site, in a probe: `clean/dx12/kernels/fg_probe.hlsl:46`. No production D3D12
kernel uses it. The HIP side uses it for Model K's opt-in "fused16" kernels (section 6).

### 1.2 Model M (Preset M; shipped; D3D12 backend is what players run)

Sources are listed per HLSL file. "FP8 / F16" = static matrix call sites. Conversions: Q8 = F32acc->FP8B,
H = F32acc->F16acc, U = FP8B->F32acc.

| Kernel family (file) | Launched names (`@wg-kernel`) | FP8 mma | F16->F32 mma | F16->F16 | Conversions (Q8 / H / U sites) | Notes |
|---|---|---|---|---|---|---|
| `clean/dx12/kernels/swin_block.hlsl` | `swin_block_c96_nc`, `swin_block_c96_add_ex_nc` (also instantiated for Model L at C=32 by `l_swin.hlsl`) | 11 | 1 | 0 | 14 / 4 / 2 | q/v/proj/MLP FP8; scores `S = U^T U` FP8 (`:788-789`); attention output `A V` on F16 (`:848`) |
| `fused_swin.hlsl` (generated) | `swin_block_c96_pm_nc`, `_pm_rev_nc`, `_add_xe_nc`, `_add_xe_rev_nc`, `_add_xe_xo_nc`, `_add_xe_xd_nc` | 13 | 1 | 0 | 16 / 4 / 2 | same block with merge/neighbour fusion |
| `swin_c64w2.hlsl`, `swin_c64ex.hlsl` (includes c64w2) | `swin_block_c64_nc`, `swin_block_c64_add_ex_nc` | 10 | 1 | 0 | 2 / 2 / 2 | 128-lane workgroup (`group=128`) |
| `swin_gemm.hlsl` | `patch_merge_c64`, `patch_merge_c96`, `expand_c96` | 2 | 0 | 0 | 6 / 0 / 0 | linear layers |
| `bneck.hlsl`, `fused_bneck.hlsl` | `bneck/swin_bneck6_nc`, `_mb_nc`, `_mbo_nc` | 7 / 9 | 1 / 1 | 0 | 7-8 / 2 / 3 | bottleneck block with spin waits across workgroups |
| `dec1_dec0.hlsl`, `fused_dec.hlsl` | `d1p12/swin_block_c64_add_dec0_nc`, `_xd_nc` | 14 / 14 | 1 / 1 | 0 | 9 / 4 / 0 (FP8->F32 at `:498-499,798-799`) | dec1 block fused with dec0 coefficient decoder |
| `encoder.hlsl` | `input_encoder` (+ `_direct`, `emo/`, `lev/` variants) | 2 | 0 | 0 | 2 / 1 / 0, plus F16->F32 round trip `:273-274` | enc0 projection |
| `exposure.hlsl`, `native_helpers.hlsl`, `native_game_helpers.hlsl` | `exposure_*`, `color_divide`, `scale_rgb`, `native_fill`, `timing_barrier`, `native_driver_probe` | 0 (1 in the probe) | 0 (1 in the probe) | 0 | 0 | NO matrix instruction except the create-time probe `native_driver_probe` |
| `post.hlsl`, `post_filter.hlsl` | `w64/post_*`, `pmo/post_*`, `w64/post_filter_c1_s4` | 0 | 0 | 0 | 0 (post.hlsl has 1 `wg_ags_convert` at `:209`, F32->F16) | NO matrix; WAVE64, block 256 / 512 / 64 (READ, `@wg-kernel` lines) |
| `f16lut`, `gelu_lut` tools | `f16lut_build_direct` | 0 | 0 | 0 | 1 convert (`f16lut/verify.hlsl:31`) | table builders / verification |

Model M matrix mix (INFERRED from `clean/FORMATS.md:85-88` and `swin_block.hlsl:606-857`): at C = 96, heads = 4, MLP = 384
a 16-token window does `4 x 6 KC` = 144 head fragments (q, k, v, proj) plus `2 KC MT` = 288 MLP fragments = 432 FP8
multiplies, plus the scores (2 FP8 per head) and `A V` (2 F16 per head, 8 per window). The F16 share is about 8 of about
450, under 2 %. Dynamic counts were not measured here (UNKNOWN; counting `v_wmma` in the RGA ISA output of
`build_dx12.py --isa` would settle it).

### 1.3 Model K (Preset K; development only, off in every release build)

READ, `BUILDING.md:205-208`: "Model K is off in every build that players get".

| File | Kernels | FP8 mma | F16->F32 mma | F16->F16 | Conversions |
|---|---|---|---|---|---|
| `clean/k/dx12/hlsl/k_block.hlsli` (via `k_block.hlsl`) | `k_stage_e0/e1/e2/d1/d2`, `k_stage2_e0/e1/d1/d0_hidden/d0_hp` | 0 | 11 (`wgk_mma`, `:224-537`) | 0 | `wgk_round8` = F32->F16 (`wg_to_f16_ags`), `k_common.hlsli:62` |
| `k_block_wide.hlsl` + `k_wide.hlsli` | `k_stagew2_e3/e4/d3/d4`, `k_stagew4_e4/m/d4` | 0 | 14 (`kw_mma`) | 0 | `kw_out` = `wg_to_f16_ags`, `k_wide.hlsli:153` |
| `k_in.hlsl`, `k_in_legacy.hlsl`, `k_out.hlsl`, `k_expo.hlsl`, `k_warm.hlsl` | `kx/k_in_*`, `k_out_*`, `k_expo_scalars*`, `k_warm` | 0 | 0 | 0 | scalar F32->F16 via opcode 0x1c (`k_out.hlsli:121`); NO matrix |

Total D3D12 side: 25 F16->F32 sites, 0 FP8, 0 F16->F16. The FP16-accumulating forms (F16->F16, `KAcc<true>`) are NOT
ported to D3D12 (READ, `clean/k/dx12/hlsl/k_common.hlsli:3-4`). K input/output kernels are wave64 (`k_in*`, `k_out*`,
block 128 / 256); the block kernels are wave32 (blocks 128 / 256 / 512).

### 1.4 Model L (Preset L; "in development for 0.7.0", README.md; not used by the runtime yet per `clean/l/README.md`)

Model L has no kernels of its own for the block: it re-instantiates Model M's template.

| File | Kernels | FP8 mma | F16->F32 mma | F16->F16 | Conversions |
|---|---|---|---|---|---|
| `clean/dx12/kernels/l_swin.hlsl` (includes `swin_block.hlsl`) | `l_swin_block_c32_nc`, `_enc_nc`, `_enc_lin_nc`, `_dec_nc`, `_dec_coef12_nc` | same as swin_block (11) + 2 via `l_lin_px.hlsli:24-25` | 1 (attention `A V`) | 0 | H also written as FP16 maps (`SW_OUT_F16`) |
| `l_ops_linear.hlsl` | `l_linear_32_32_relu`, `_f16`, `_fused` | 2 (`:94-95`) | 0 | 0 | 1 / 2 / 0 |
| `l_ops_project.hlsl` | `l_project_gather`, `l_dec_project` | 2 (`:73-74`) | 0 | 0 | 2 / 1 / 0 |
| `l_ops_coef.hlsl`, `l_post.hlsl`, `l_post_direct.hlsl`, `post_filter.hlsl` (l_ names) | `l_coef`, `l_coef_pack12`, `l_post_recon*`, `l_direct_color`, `l_post_filter*` | 0 | 0 | 0 | `l_post.hlsl` 5 F32->F16; NO matrix |
| `lev/input_encoder*` | encoder variants | 2 | 0 | 0 | as M |

"Model L's FP16 work" is FP16 maps in memory (`x2`, `y2` are FP16, `clean/l/FORMATS.md:28,33`) and FP16 element-wise
arithmetic. The multiplies are still FP8 (`l_linear_32_32_f16` is "FP8 -> FP16": FP8 operands, FP16 result,
`clean/l/FORMATS.md:81`). So L has no F16 matrix path to reuse. The NumPy oracle runs both width-32 blocks "in the FP16
form" (`clean/l/README.md`, `_block_tokens_q7`): that is a reference form, not a kernel (INFERRED from the README row;
UNKNOWN whether it is meant as a GPU target).

### 1.5 Ray Reconstruction (preset F; shipped; `clean/rr_f`)

Static sites (READ):

| File | Role | FP8 mma | F16->F32 | I8 | Conversions |
|---|---|---|---|---|---|
| `rr_block.hlsl` (10) + `rr_step.hlsli` (12) | window block, all stages, step variants E / X | 22 | 0 | 0 | via `rr_common.hlsli` (Q8 5, `rr_d8`) |
| `rr_step_y.hlsli` | step Y / Z (exact on split operands) | 29 | 0 | 0 | |
| `rr_mlp_step.hlsli` | input-stage MLP step | 6 | 0 | 0 | |
| `rr_block_c32.hlsl` | IN0 and D0-F (C=32), also a WAVE64 form | 12 (two per wave64 call) | 0 | 0 | 8 |
| `rr_block_fast.hlsl`, `rr_block_wide.hlsl` | other routes, share `rr_step.hlsli` | 0 (via headers) | 0 | 0 | `rr_fast.hlsli` 6 |
| `rr_step_i.hlsli` | step "I" (lever L4), exact via int8 digit planes | 0 | 0 | 1 (`:61`) | |
| `head/dx12/rr_d0r.hlsl` | D0-R block + head | 20 | 0 | 0 | 18 |
| `rr_probe.hlsl`, `rr_probe_y.hlsl`, `rr_probe_i8.hlsl` | probes | 5 | 1 | probe | |
| `runtime/kernels/rr_runtime.hlsl` | runtime glue | 1 | 0 | 0 | 1 |
| `rr_input.hlsl`, `rr_output*.hlsl`, `rr_resolve*.hlsl`, `rr_pool.hlsl`, `rr_exposure.hlsl`, `rr_input.hlsl` | input features, output stage, resolve, pool, exposure | 0 | 0 | 0 | `rr_input.hlsl` 9 convert; NO matrix |

RR has no F16-matrix path. What the shipped default runs (READ, `clean/rr_f/runtime/README.md`): body = step E (two FP8
WMMAs per 32-input step, "not exact to 51.1"); exact modes `WG_RR_BODY=Z` / `X` (verification) run the same FP8
instruction with extra WMMAs for exactness (step X: 4 WMMAs, `rr_step.hlsli:20-40`). Step "I" (`STEPS["I"]=4`,
`clean/rr_f/gpu/stage.py:27`) is an opt-in exact route on int8 WMMA; I did not find it in the default runtime path
(READ; status of its use in `rr_runtime.cpp` not traced: UNKNOWN).

RR WAVE64: `W64_DEFAULT = "D0-F,IN0"` (`clean/rr_f/gpu/stage.py:413`) compiles those two level-0 stages in wave64 "one
wave64 = two token tiles ... the v_wmma wave64 layout" (`clean/rr_f/kernels/rr_block_c32.hlsl:13-21`), only when step E and
`RR_B32=1` (`stage.py:416-419`). So a wave64 matrix path exists in shipped-style code. It issues two wave32-equivalent
products per call with 4 dwords per lane (`rr_w64f`, `rr_block_c32.hlsl:28-34`).

### 1.6 Frame generation (FG; groundwork, "nothing here is used by the runtime", `clean/fg/README.md`)

| File | Kernels | FP8 | F16->F32 | F16->F16 | Conversions |
|---|---|---|---|---|---|
| `fg_mconv.hlsl` (`:53`) | `fg_mconv1/2/4` (tier B fast path) | 0 | 1 site (x2 listed by grep; 2 counted incl. comment) | 0 | `wg_to_f16_ags` |
| `fg_mpair.hlsl` (`:134`) | `fg_mpair_b`, `_d0..d4` | 0 | 1 | 0 | `wg_to_f16_ags` (`:78,135`) |
| `fg_probe.hlsl` (`:46,72`) | `fg_probe_mma16` (F16->F16), `fg_probe_mma32` (F16->F32) | 0 | 1 | 1 | probe only |
| `fg_conv.hlsl`, `fg_pool.hlsl`, `fg_upscale*.hlsl`, `fg_add.hlsl`, `fg_leaky_fast.hlsl` | exact path (`fg_exact16.h`, integer arithmetic) | 0 | 0 | 0 | NO matrix |

FG's "exact" mode needs no matrix instruction (bit-exact FP16 chunk accumulation in integers; `clean/fg/README.md`
row `kernels/fg_exact16.h`). The matrix path is explicitly approximate: MEASURED-IN-REPO
(`clean/fg/records/wmma-probe-2026-11-14.json`, README row `records/wmma-probe`): neither F16->F16 nor F32-accumulate
equals the spec's per-chunk rounded sum; on 62.4 M ordinary outputs F32 accumulate rounded to F16 per chunk is more
than one step off in 1,216 outputs. Whether gfx11 behaves the same is UNKNOWN.

### 1.7 Totals

Static call sites, counting each source family once (M: swin_block, fused_swin, swin_c64w2, swin_gemm, bneck, fused_bneck,
dec1_dec0, fused_dec, encoder, l_ops_*, l_lin_px; probes excluded):

| Model | FP8 mma | F16->F32 | F16->F16 | I8 | FP8 share of sites |
|---|---|---|---|---|---|
| M (incl. L's shared kernels) | 11+13+10+2+7+9+14+14+2 = 82 | 1+1+1+0+1+1+1+1+0 = 7 | 0 | 0 | 82 / 89 = 92 % of sites (about 98 % of instructions, INFERRED above) |
| L extra (ops + lin_px) | 2+2+2 = 6 | 0 | 0 | 0 | 100 % |
| K | 0 | 25 | 0 (HIP only) | 0 | 0 % |
| RR | 22 + 29 + 6 + 12 + 20 = 89 (production kernels; +5 probes, +1 runtime) | 0 | 0 | 1 | about 99 % |
| FG | 0 | 2 | 0 (probe) | 0 | 0 % |
| All | about 177 | about 34 | 0 | 1 | about 84 % of sites are FP8 |

Because the share by site is dominated by how many generated duplicates and variants a model has, the better statement is:
three of the five models (M, L, RR) are FP8-matrix, two (K, FG) are F16-matrix. Dynamic fractions: UNKNOWN.

## 2. Fragment layout the shaders assume

### 2.1 Packing in the channel (READ, `clean/dx12/hlsl/wg_ags.hlsli`)

- A, B and C are loaded "two dwords at a time" into the channel: `wg_ags_load` (`:47-51`) calls
  `wg_ags_call(op, 0, (i&1) | ((i>>1)<<4) | (operand<<8), f.v[2i], f.v[2i+1])` for i = 0..3, so a fragment is always
  8 dwords per lane (`struct wg_frag { uint v[8]; }`, `:36`), trailing dwords 0.
- Result: `wg_ags_result` (`:53-58`) reads 8 dwords, one per call.
- Dwords per lane per format (READ, `:25-32`, `wg_platform.hlsli:210-221`):

| Format | A | B | C / D | Used dwords |
|---|---|---|---|---|
| FP8 x FP8 -> F32 | 2 (8 bytes = 8 K values) | 2 | 8 floats | `wg_frag2`: v[0..1]; rest 0 |
| F16 -> F32 | 4 (8 halfs) | 4 | 8 floats | `wg_frag4`: v[0..3] |
| F16 -> F16 | 4 | 4 | 4 dwords (8 halfs) | `wg_frag4` |
| I8 -> I32 | 2 | 2 | 8 ints | `rr_step_i.hlsli:57-62` |
| convert F32acc -> F16acc / FP8B | 8 floats in | | 4 dwords / 2 dwords out | `wg_to_f16_ags`, `wg_pack_q8_nc_ags` |

### 2.2 Lane mapping (wave32, gfx12)

Stated identically in several places. Quoted, `clean/dx12/kernels`-independent sources:

- `clean/rr_f/head/dx12/rr_d0r.hlsl:5-9`: "Fragment layouts of v_wmma_f32_16x16x16_fp8_fp8 (wave32): lane l, r = l & 15,
  hf = l >> 4; A = row r, K 8hf .. 8hf+7; B = column r, K 8hf .. 8hf+7; C / D = column r, rows 8hf .. 8hf+7."
- `clean/runtime/wmma_probe.h:7-12`: A element j of lane l = `A[r][8h + j]`; B element j = `B[8h + j][r]`; C, D element v of
  lane l = row `v + 8h`, column r. A and B are chosen so that any other row/column/lane/half order changes a result.
- `clean/k/dx12/hlsl/k_common.hlsli:8-12` (same for FP16: 8 halfs per lane = 4 dwords).
- `clean/FORMATS.md:33-46`: "The hardware's actual K index for a slot does not matter, as long as the A and B operands
  use the same one"; accumulator `D_ROW[h][v] = v + 8h` is "the default assumed for gfx12".
- The channel header itself: "wave32 fragment layouts of gfx12's v_wmma_*_w32" (`wg_ags.hlsli:60`).

What the kernels rely on beyond the table (READ):

1. D feeds B with no lane movement. `clean/kernels/swin.hip:6-9`: "the FP32 result of one WMMA (lane holds column
   n = token, 8 rows) already has the shape of the next WMMA's FP8 B operand (lane holds column n = token, 8 K values), so
   no data moves between lanes except for the two RMSNorm sums and the softmax max/sum (one lane-exchange each)". The
   same sentence appears for Model K (`k_common.h:10-16`) and RR (`rr_d0r.hlsl:8-9`).
2. D also feeds A through the transposed product: Model K's `v^T` product with swapped operands "relies on this gfx12
   layout as in HIP" (`k_common.hlsli:11-12`); in M, `vt[dt] = wg_to_f16_ags(accv)` is used as the A operand of
   `wg_wmma_f16(vt[dt], p, ...)` (`swin_block.hlsl:776-777,848`).
3. Lane half exchange `lane ^ 16` is the only cross-lane op in the matrix code (`wg_xor16`, `wg_platform.hlsli:103-107`,
   `WaveReadLaneAt(v, WaveGetLaneIndex() ^ d)`); it moves the "other 8 channels of the same token" (`k_common.hlsli:56-58`).
4. Token t of a window lives in lanes t and t + 16 (`swin.hip:6-7`); "Layout L": lane half h holds channels
   `[h*C/2, (h+1)*C/2)` (`swin.hip:12`, `FORMATS.md:48`). Feature maps in memory are `[H][W][C]`, so each lane reads C/2
   contiguous bytes. That loads 8 K values per lane per fragment, the gfx12 shape.
5. Weight fragments: 256 bytes per FP8 fragment, lane l reads bytes `[8l, 8l+8)` (`FORMATS.md:33-34`); FP16 tile: 32 lanes x
   8 halfs = 512 bytes (`clean/k/FORMATS.md:36-41`); loaded by one 8/16-byte load per lane (`wg_ld64`/`wg_ld128`).

### 2.3 Helpers that pack and unpack fragments (quoted)

- Dword-pair load / result: `clean/dx12/hlsl/wg_ags.hlsli:47-58`.
- `wg_frag_v8f`, `v8f_frag`, `wg_frag2`, `wg_frag4`: `clean/dx12/hlsl/wg_platform.hlsli:208-217`
  (`f.v[0] = a.x; f.v[1] = a.y; [unroll] for (uint i = 2; i < 8; ++i) f.v[i] = 0u;`).
- FP8 pack: `wg_pack_q8_nc_ags` returns `uint2(r.v[0], r.v[1])` of the converted fragment (`:223-227`): element i -> byte i.
- FP16 fragment as 4 dwords, element i in dword i/2, low half first: `wgk_el` (`k_common.hlsli:28`).
- RR wave64 forms: `rr_w64f` and the two-call `wg_wmma_fp8` (`rr_block_c32.hlsl:28-45`).
- RR int accumulator `RrIA`: "element i of lane l: D[8 (l >> 4) + i][l & 15]" (`rr_step_i.hlsli:46`).
- Emulator model: `clean/kernels/emu/wg_emu.cpp:2-10,69-77,92-118`: `kslot(h, j) = 8h + j`, `drow(h, v) = v + 8h`;
  alternate `kvariant` (j<4 ? 4h+j : 8+4h+(j-4)) and `dvariant` (`2v + h`, the gfx11-style accumulator row).
- Python packer: `clean/runner/packing.py:16-19`: `D_ROW_LAYOUTS` has `"gfx12": v + 8h` and `"gfx11": 2v + h ("alternative, for probes")`.
- Model K packer refuses the gfx11 form for blocks: `clean/k/runner/k_pack.py:164-165`:
  "the model K block kernels need the gfx12 WMMA accumulator layout (d_row rows v + 8 h)".

### 2.4 Wave size assumptions

- `WG_WAVE` default 32; each kernel's `@wg-kernel ... wave=32|64`; `WG_KERNEL` adds `[numthreads(WG_BLOCK,1,1)]
  [WaveSize(WG_WAVE)]` (`wg_platform.hlsli:22-23,40`). `wg_wave_lane()` is `WaveGetLaneIndex()` (`:52`), checked equal to
  `wg_lane_id() % WG_WAVE` by an arithmetic probe.
- Matrix helpers exist for wave32 only: `#if WG_WAVE == 32 #include "wg_ags.hlsli" ...` (`:206`); wave64 kernels that
  need a conversion include the header by hand (`k_out.hlsli:117`, `rr_block_c32.hlsl`).
- Wave64 kernels (all NO matrix except RR's level-0 route): `w64/post_*` (block 256/512), `post_filter` (64), `kx/k_in*` (128/256),
  `k_out*` (128). READ from the `@wg-kernel` lines.
- HIP: `#define WG_WAVE 32  // gfx12 kernels run in wave32` (`clean/kernels/wg_platform.h:23`).
- Lane constants hard-coded as 32 / 16: `lane & 31u`, `lane & 15u`, `lane >> 4` appear in `swin.hip`, `fg_mconv.hlsl:61-65`,
  `fg_mpair.hlsl:161-162`, `fg_probe.hlsl:36-39`. INFERRED: every matrix kernel contains the lane arithmetic of section 2.2.

Comments and constants that name the architecture (READ): "gfx12" / "RDNA4" in the header of every `.hip` (`swin.hip:2`,
`encoder.hip:2`, `decoder.hip:2`, `post.hip:3`, `exposure.hip:2`, `k_common.h:10,18`), "wave32 fragment layouts of gfx12's
v_wmma_*_w32" (`wg_ags.hlsli:60`), "RDNA 4 / D3D12" (`rr_block.hlsl:1`, `rr_d0r.hlsl:1`), "tuned for RDNA 4" design
docs (`clean/rr_f/DESIGN-51.md`, `clean/rr_f/head/DESIGN-55.md`).

## 3. Where FP8 comes from

| Model | Weights in the model file | Weights in kernels | Activations |
|---|---|---|---|
| M, L | `model.wgcm` holds FP32 tensors (`data/preset-M.v1.map.json`: all dtype float32 except one uint8; same for L) | quantized to E4M3 at load and packed into 256-byte fragments: `clean/runtime/model.cpp:89` `fp8_encode(M.at(...))`; Python `packing.py:92` | FP8 maps in memory (`FORMATS.md:12-13`); quantized by `wg_pack_q8*` (driver conversion `v_cvt_pk_fp8_f32`, or software `wg_pack_q8`) after each product; read back by `wg_unpack_fp8` / driver conversion to FP32 |
| K | FP32 tensors, FP16 values (`clean/k/FORMATS.md`, `preset-K.v1.map.json` all float32) | FP16 tiles | FP16 maps; no FP8 |
| RR | `model.wgcm` container v2 with dtype 6 = e4m3: preset F map has 104 e4m3 tensors, 153 float16, 44 float32 (`clean/converter/OUTPUT-FORMAT.md:36,103`, `data/preset-F.v1.map.json`) | read as E4M3 codes straight into fragments; also int8 digit planes for step I (`layout.pack_weights_int`) | e4m3 (Q8) between products (`rr_common.hlsli:26-44`) |
| FG | float32 tensors holding FP16 values (`fg.v1.map.json`) | FP16 weight fragments | FP16 |

Evidence on the first row: the clean-room equivalence tool requires M's values to be exactly on the FP8 grid
(`clean/equiv/model_m.py:118-129`: "float32 values (spec 90: exactly on the format's grid)... raises when a value is
off-grid"), and the packer rounds with `q8` first so exact values map exactly (`clean/reference/fp8.py:60-62`). Whether the
FP32 values the converter writes for M are always FP8-exact is not recorded in the files I read (UNKNOWN; a test:
run `is_fp8_exact` on a real converted model's weight tensors).

Activation quantization, by who does it (READ):
- Driver conversion (`wg_ags_convert` F32acc 0x26 -> FP8 B 0x1a): the production fast path. `wg_pack_q8_ags`,
  `wg_pack_q8_nc_ags`, `wg_pack_q8_inner_ags` (`wg_platform.hlsli:223-237`), `sw_pack_q8`; RR's `rr_q8`
  (`rr_common.hlsli:37-41`). Wave-uniform control flow only; saturate flag never used (clamp in the shader first).
- Software shader arithmetic: `wg_pack_q8` / `_nc` / `_inner` (`wg_platform.hlsli:180-200`, any control flow, bit-identical).
- FP8 -> F32 unpack: driver conversion 0x1a -> 0x26 (`sw_unpack_fp8`, `swin_common.hlsli:82-90`, switch `SW_UNPACK_AGS`,
  default 1; software fallback `wg_unpack_fp8`, `wg_platform.hlsli:195-203`); also `unpack8` (`bneck.hlsl:123`), `rr_d8`
  (`rr_common.hlsli:45`), and the FP8-code lookup tables of dec0 (`FORMATS.md:237`).
- HIP: `__builtin_amdgcn_cvt_pk_fp8_f32` and `__builtin_amdgcn_cvt_f32_fp8` (`clean/kernels/wg_platform.h:149-196`).

Places that need hardware FP8 conversion: every `wg_pack_q8*_ags` and `*_unpack` site above. A pure-software substitute
already exists for each (INFERRED from the `SW_UNPACK_AGS=0` switch and the software packs; speed cost not recorded: UNKNOWN).

## 4. Other RDNA 4 / driver-specific dependencies

### 4.1 `clean/dx12/vgpr_limits.json`
READ (its `_doc` string): the RGA (gfx1201) `USED_VGPRs` bound of each Swin / bottleneck kernel, chosen to keep
workgroups per CU: "VGPR file 1536 per SIMD lane, granule 8, a 256-lane wave32 workgroup = 4 waves on each of a CU's
2 SIMDs, so 2 workgroups per CU need <= 192, 3 need <= 128". Entries: 192 for the C=64/96 and bottleneck kernels,
96 for the five `l_swin_block_c32*` kernels. A violation fails the build (`build_dx12.py:321-345`; `WG_DX12_VGPR_CHECK=0`
skips). The numbers are gfx12 register-file and occupancy arithmetic (RGA call is `-c gfx1201`, `build_dx12.py:293,311`).
MEASURED-IN-REPO: "swin_block_c96_add_xe_* at 215 VGPRs ran 1 per CU, +38 us per frame at 1440p Q" and the GELU T5
table kernels at 201 / 215 VGPRs lost 103-130 us (`clean/dx12/README.md`, "Build and compare"). On gfx11 the VGPR file is
different (UNKNOWN: needs the gfx1100 RGA numbers), so every bound would be re-derived.

### 4.2 `clean/dx12/dxil_asm.cpp`
READ: assembles a DXIL text listing (LLVM IR from `dxc -Fc`) into an unsigned DXIL container using the DXC library's
`IDxcAssembler`; `build_dx12.py` then signs it with `dxv`. Purpose is the "exact-arithmetic rewrite" of the listing:
remove `precise` marks so the driver lowers intrinsics as in DXC default mode (`clean/dx12/README.md`, "Arithmetic mode").
It does not touch the matrix code. It is Windows-only (`windows.h`, `LoadLibraryExA`). Nothing in it is RDNA 4 specific.

### 4.3 "Rules on this driver" (READ, `wg_ags.hlsli:19-22`, `clean/dx12/README.md` "Driver exceptions")
- Every channel call in wave-uniform control flow (a divergent call is not defined); helpers branchless.
- Issue all loads a group of multiplies needs before its first call (the compiler never moves loads across the UAV atomics).
- Never pass the saturate flag (40 % slower); clamp in the shader.
- The gfx1201 compiler regroups multiplication chains, fuses FP16 multiply+add into `v_fma_mix`, folds an FP32 op into the
  matrix conversion, drops stores to two pointers at one offset, unrolls `[loop]`; workarounds are opaque-zero tricks
  (`wg_grid_ >> 31`). Lanes past the grid end corrupted other lanes' matrix results (frame generation `fg_mpair`).
- Kernels that rely on those compiler behaviours (INFERRED): every file with `wgk_opq`, `rr_fence`, `sw_zero`.
  They were tuned against AMD's gfx1201 compiler, and gfx1100 may behave differently (UNKNOWN: re-run `compare.py` and
  ISA store counts).
- "Match HIP ISA bits" is a design rule for M, K, RR, FG: the D3D12 port reproduces the HIP kernel's rounding instruction by
  instruction (`v_cvt_pk_fp8_f32`, `v_fma_mix`, `v_cvt_pkrtz`). Some of these instructions may not exist on gfx11
  (UNKNOWN).

### 4.4 LDS / groupshared
- D3D12 limit 32 KB; `swin_common.hlsli:15-17` errors above that; K's NG=4 layout was re-done to fit 32 KB (HIP: 56 KB)
  (`k_wide.hlsli:34`). Weights stream to LDS in 6-16 KB units, double-buffered (`FORMATS.md:122-125`).
- LDS size on gfx11: UNKNOWN (RDNA 3 has 64 KB per workgroup in HIP; D3D12 still caps groupshared at 32 KB).

### 4.5 Gates keyed to RDNA 4 (READ)
- `clean/runtime/rdna4.h:16-18`: vendor 0x1002 and description contains "RX 9070" or "RX 9060".
  Used at `clean/runtime/device_d3d12.cpp:1367`, `clean/runtime/interop/executor_select.h:105`,
  `clean/rr_f/runtime/src/rr_runtime.cpp:185,220,1152,1344,3784,4285`.
- Create-time probe `native_driver_probe` / `wmma_probe.h` runs one wave of FP16 and FP8 products and fails if any lane
  layout differs (`device_d3d12.cpp:1393-1395`). On RDNA 3 the FP8 half of this probe cannot pass.
- HIP builds: `ARCH = WG_GPU_ARCH or "gfx1201"`; `supported release targets are gfx1200 and gfx1201 only`
  (`clean/runner/build.py:149,481-482`); `clean/runtime/device.cpp:398,406` accepts only gfx1200 / gfx1201 devices.
- Launcher `clean/installer/src/wg_core/checks.cpp` also names gfx12 (not read in detail).

## 5. Pieces that do not need FP8

| What | Where | Matrix? | What it is for | Status |
|---|---|---|---|---|
| Model K network | `clean/k/dx12`, `clean/k/kernels` | F16->F32, 25 sites | second network, exact FP16 | development; off in every release (BUILDING.md:205-208) |
| Model K FP16-accumulating forms | `k_common.h:286-316`, `k_block.hip`, `k_block_wide.hip` | F16->F16 (HIP only) | spec 49 speed approximation | opt-in HIP; not ported to D3D12 |
| Model K FP8 forms | `k_block8.h` | FP8 (HIP only) | "fused8" speed approximation, "not bit-exact" | opt-in HIP |
| Frame generation fast path | `fg_mconv.hlsl`, `fg_mpair.hlsl` | F16->F32 | tier B net1 | groundwork; approximate by measurement |
| FG exact path | `fg_conv/pool/upscale/add/leaky_fast.hlsl`, `fg_exact16.h` | none | bit-exact reference-grade path; validation and fallback | groundwork |
| F16 attention `A V` | `swin_block.hlsl:848` and kin | F16->F32, 1-2 per head per window | attention output (softmax weights in FP16) | production |
| Non-matrix passes | M: exposure, native helpers, post*, post_filter; L: `l_post`, `l_ops_coef`; RR: input features, output, resolve, pool, exposure; K: `k_in/out/expo/warm` | none | pre/post processing | production / development as above |
| RR step I (int8 WMMA) | `rr_step_i.hlsli` | I8->I32 | exact FP8 products via int8 digit planes (`:1-40`) | opt-in lever; default path is step E |
| RR exact scalar step | `rr_exact_math.hlsli`, `RR_STEP 2` | none | exact binary64 reference step | diagnostic |
| CPU oracles | `clean/reference`, `clean/l/reference/oracle.py`, `clean/k/reference`, `clean/rr_f/reference`, `clean/fg/reference` | none | NumPy exact references | tests |
| Emulator | `clean/kernels/emu/` | emulates both formats | host test-only lane emulator, "never shipped" (`wg_emu.cpp:2`) | tests |
| Model L "FP16 form" oracle | `clean/l/reference/oracle.py` `_block_tokens_q7` | none | oracle of the FP16 variant of the width-32 blocks | reference |

INFERRED but important: F16 products of FP8-grid values are exact in FP32 accumulate (E4M3 has a 4-bit significand;
F16 has 11; the product has at most 8 significant bits; exponent range of E4M3 products fits F16 inputs even when the
product exceeds 65504 because the accumulator is FP32). So an F16-WMMA version of an FP8 kernel can give the same sums as
FP8 WMMA as long as the order of accumulation does not matter. The repo's own note says FP8 WMMA is not "exact sum rounded
once to f32" (`clean/rr_f/gpu/README.md:80`, probe P1), so bit identity with the FP8 instruction is UNKNOWN; a test:
run the RR probes P1 / P1b on gfx11 F16 WMMA.

## 6. HIP side

- Builtins (READ, `clean/kernels/wg_platform.h:47-50`): `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` and
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`; plus `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12`
  (`clean/k/kernels/k_common.h:293`); conversions `__builtin_amdgcn_cvt_pk_fp8_f32`, `__builtin_amdgcn_cvt_f32_fp8`
  (`wg_platform.h:149-196`); `__builtin_amdgcn_ds_bpermute`, `readfirstlane`, `fdot2`, `sched_group_barrier` etc.
  All `_w32_gfx12` suffixed, so they do not compile for gfx11 targets (INFERRED from the names; LLVM has
  `_w32` forms with different operand types for gfx11, see https://llvm.googlesource.com/llvm-project/llvm/+/6a4c0551fbd905397666676142d59af81585c2cb/test/MC/AMDGPU/gfx11_wmma.s).
- Counts: swin.hip 14 FP8 + 1 F16; swin_v2.hip 9 + 2; decoder.hip 4; encoder.hip 2; mlpbench.hip 2; `k_block8.h` 6 FP8;
  `k_common.h` 3 F16->F32 and 5 F16->F16 mentions; `k_linear.hip` 1; `fg_net1_mma.hip` 3; `fg_probe.hip` 4.
- Target flags: `clang -x hip --offload-arch=gfx1201 --cuda-device-only -nogpuinc -O3` (`clean/runner/build.py:15`),
  `--extra-archs gfx1200` (`BUILDING.md:65`). Kernels build without HIP headers (`-nogpuinc`).
- Is HIP shipped? READ: players need "No AMD HIP runtime ... (only the optional sparkle guard uses it)"
  (`README.md:23-24`); the runtime has four backends and the native D3D12 backend 3 has "no HIP runtime or fallback"
  (`clean/runtime/device.h:4-13`). `BUILDING.md:63-65` still builds HIP code objects for gfx1201 and gfx1200 into the
  runtime library; `clean/dx12/README.md:5-8` calls the hybrid backend 2 "the hybrid oracle". Conclusion (INFERRED from
  these lines): HIP is a development and verification tool and the byte-for-byte oracle for the D3D12 ports; the shipping
  game path is D3D12 backend 3. Whether the release zip contains the HIP code objects: UNKNOWN (check `out/runtime/*.co`
  in a release).
- Emulator: `clean/kernels/emu` runs the HIP kernel source on a CPU with configurable layout (section 2.3). It is the only
  place where a gfx11 accumulator or K layout is already exercised in code (`dvariant`), but the repo says kernels depend
  on the row/column-per-lane part only.

## 7. Linux / Proton note (relevant because ROADMAP lists RDNA 3, Vulkan and Linux separately)
`clean/linux/README.md:3-6`: under Proton the driver channel is "translated by vkd3d-proton's dxil-spirv instead of AMD's
driver", tests include `ags-layout` ("does not keep the gfx12 register semantics the kernels rely on" is the failure it
would catch), RDNA 4 + RADV required. So a second implementation of the channel already exists (in vkd3d-proton). Whether it
could expose gfx11's WMMA: UNKNOWN.

## 8. What RDNA 3 changes, as far as this lane can say

External facts (https://gpuopen.com/learn/wmma_on_rdna3/ and https://gpuopen.com/learn/using_matrix_core_amd_rdna4/ via search
summaries; not read in full by me):
- RDNA 3 WMMA requires A and B contents replicated between lanes 0-15 and 16-31 in wave32; each thread holds 8 VGPRs per C/D
  fragment. RDNA 4 changed the VGPR layout and removed the duplicated A / B elements so each lane loads 8 elements.
- RDNA 3 WMMA has F16 / BF16 / iu8 / iu4 forms; I did not confirm that it has any FP8 form (my knowledge says it has not).

Consequences for the kernels (INFERRED from sections 1-2):
1. M, L, RR, FG, K all build `B = previous D` with no lane exchange. On gfx11 the accumulator rows are interleaved
   (`2v + h` in the repo's own `gfx11` table) and B needs 16 K values per lane in both halves; a half exchange per fragment
   would be needed (`wg_xor16`-like, `ds_swizzle` / permlane). Cost: UNKNOWN until measured.
2. FP8 operands do not exist; M / L / RR would run on the F16 instruction after widening FP8 to F16 (exact) with twice the
   operand bytes in registers and LDS. K and FG use F16 already.
3. Weights are repacked per layout by `packing.py` / `k_pack.py` / `model.cpp` (the packers are the one layout-aware
   part, and `D_ROW_LAYOUTS` already has "gfx11"), but K's packer refuses non-gfx12 for blocks and the kernel dataflow is not
   parameterised.
4. The channel opcode / format codes (0x28 / 0x32) are AMD driver internals. Whether RDNA 3 drivers accept them and with
   which format codes is UNKNOWN.

## Open questions

1. Do AMD's RDNA 3 D3D12 drivers honour the 0x7FFF0ADE channel at all, and which mma format codes does it take (F16 = 0x5 on gfx11?)?
2. Does wave64 matter on gfx11? (RDNA 3 supports both; RR already compiles two stages in wave64 with a "wave64 layout".)
3. Are Model M's converted FP32 weights always on the FP8 grid (`is_fp8_exact`)? If not, "FP8 weights" are a rounding step the
   runtime does at load, and an F16 path must reproduce it.
4. Dynamic share of F16 vs FP8 instructions per frame (the static count here is not time-weighted).
5. Is RR step I (int8 WMMA) used in any shipped route? It would be a direct bridge to gfx11's iu8 WMMA.
6. What does the release zip contain from HIP (`*.co`), and is the HIP backend reachable by a player?
7. Does gfx11 round F32->F16 conversions and F16 accumulation as gfx12 does (the FG and K "match HIP bit for bit" rules)?
8. Which kernels use instructions that do not exist on gfx11 (`v_cvt_pk_fp8_f32`, `v_maximumminimum_f32`, `v_fma_mix`,
   `v_pk_*_f16` forms)? Needs an ISA diff with RGA for gfx1100.

## What to measure on an RX 7900 XT

1. Run `clean/runtime/wmma_probe.h`'s probe with the F16 half only (new FP8-free variant) through the AGS channel; record lane layout of D and A/B. Confirms or refutes section 8.
2. Run `clean/dx12/tests/native_probe_test.cpp` and `clean/linux/ags_layout_host.cpp` unchanged: expect FP8 format 0x7 and conversions 0x1a to fail; list which others pass.
3. RGA `-c gfx1100 --isa` over `clean/dx12/kernels` (build_dx12.py has `--isa`): VGPR, LDS, scratch per matrix kernel; compare to `vgpr_limits.json`.
4. F16 WMMA bit behaviour (accumulate rounding, F16->F16 rounding) using `clean/fg/runner/fg_wmma_probe.py` against gfx11.
5. Throughput of F16 WMMA per CU on gfx11 versus FP8 on gfx12 (the repo measured 7.7 vs 15.5 cycles per instruction on the RX 9070 XT, `clean/k/kernels/k_block8.h:7-9`, for the same instruction pair).
6. Cost of the lane-half exchange that replaces "D is B": microbenchmark a chain of two dependent 16x16x16 F16 WMMAs with and without `lane ^ 16` data movement.
7. Dynamic instruction counts (`v_wmma` per frame) for each model on gfx12 to weight section 1.7.
