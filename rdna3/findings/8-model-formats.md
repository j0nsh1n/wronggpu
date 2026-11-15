# Lane 8: number-format profile of every model in the tree

Source read: upstream main at 994f304 (the local snapshot). Upstream state (section 9) comes from read-only `gh`
queries made today, so it can be newer than the snapshot. Labels: MEASURED-IN-REPO, READ, INFERRED, UNKNOWN. All paths
are relative to the repo root. Nothing was built or run; the MAC counts in 4.1 are my arithmetic on formulas from the repo.

## 0. One-page answer

| Model | Matrix format today | Weights | Activations between products | Accumulate | INT8 fit (INFERRED) | FP16 fit (INFERRED) |
|---|---|---|---|---|---|---|
| M (Swin, `clean/dx12`, `clean/kernels`) | FP8 E4M3 x E4M3, about 98 % of products; FP16 only for attention `A V` | float32 values on the E4M3 grid (file holds FP16, `fp8` op rounds) | E4M3 codes, saturating at +-448, RMS-normalised inputs | FP32, chains of up to 32 WMMA (K = 512) | Poor as one plane; exact only with 2-3 digit planes | Good in fidelity (E4M3 is exact in F16); costs speed |
| L (M's ten stages plus width-32 blocks) | same as M, plus FP16 map `x2` / `y2` between ops | same as M | E4M3, except two FP16 maps | FP32 in kernels; oracle is moving to an FP16-accumulate "q7" form per 32 products | as M | as M |
| K (`clean/k`) | FP16 x FP16, FP32 or FP16 accumulate | FP16 values (float32 containers) | FP16 everywhere, rounded after every op | FP32 default; FP16 per 16-deep step opt-in, measured within spec | Poor (unquantised FP16 activations) | Native; best fit |
| RR preset F (`clean/rr_f`) | FP8 E4M3, 104 tensors | E4M3 codes (file holds FP16, `e4m3` op) | E4M3 (Q8, saturating +-448); FP16 hidden state | FP32 inside a 32-product step, rounded to FP16 per step | IU8 digit planes already exist (step "I"), exact, 4-9 WMMAs per step | Good in fidelity; same cost story as M |
| FG (`clean/fg`) | FP16 x FP16 | FP16 | FP16, leaky ReLU, FP16 chunk accumulator | FP16 accumulator, one rounding per 16 products (delivered rule) | Poor | Native; matches the 16-deep chunk |

The tree contains no int8 model, no per-tensor or per-channel scale tensor, and no FP8 or INT8 statement about the original
network's hardware beyond what section 5 quotes.

## 1. Model M (clean/dx12, clean/kernels, clean/reference)

### 1.1 Weights (READ)

- Converter source block `W` has element type `f16le` (`data/preset-M.v1.map.json`, `sources.W`), so the player's file
  holds FP16. The map then applies the op named in the table below. All tensors are 129 over 14 stages (`data/preset-M.v1.map.json`,
  tallied from `stages[].tensors[]`):

| dtype in map | op | tensors | elements |
|---|---|---|---|
| float32 | `fp8` (exact round to E4M3, ties to even, saturating) | 56 | 1,549,568 |
| float32 | `f16` | 36 | 7,524 |
| float32 | `f16_scaled` (mul = div = sqrt(64): the RMSNorm gain, bias B) | 30 | 11,392 |
| float32 | `exp`, `u32_as_f32` (scalars) | 3 + 3 | 3 + 7 |
| uint8 | `u16_shr` (dither table T, 8 x 16 x 16) | 1 | 2,048 |

- Only the weight matrices (`W`, `Wqk`, `Wv`, `Wp`, `W1`, `W2`, `Wm`, `We`) take `fp8`. Biases, norm gains and the
  attention bias are FP16 values. Evidence: tensor lists of `enc0`, `enc1`, `enc2` (map stages).
- The runtime packs the E4M3 values into 1-byte codes at load (`clean/runtime/model.cpp:85-89`, `clean/runtime/fpconv.h:80`).
- "Weight quantization is part of each FP8 model's definition (the references apply `q8`, `clean/reference/swin.py:244`)",
  and `random_params` draws FP8-exact matrices (`clean/reference/swin.py:316-327`).
- Scales: none for the FP8 weights. `f16_scaled` is a one-off constant fold (`mul` / `div` hex constants named
  "sqrt(64)"), not a quantisation scale. READ in the map `op` entries.
- Weight size per block: 49,168 B (C = 64), 117,776 B (C = 96), 188,432 B (C = 128) (`clean/FORMATS.md:20-26`). One byte per FP8 weight, so an FP16 weight set would roughly double those (my arithmetic).

### 1.2 Activations (READ)

- Feature maps between stages are E4M3, one byte per channel (`clean/FORMATS.md:12-13`). The decoder input `x = e + S`
  is not re-quantised: the kernel adds two FP8 values in FP32 (`clean/FORMATS.md:17-19`).
- Inside a block, the reference quantises at these points (`clean/reference/swin.py:176-190`): normalised input `a`
  (step 1-2), `u` = Wqk a (shared q and k, step 3), `o` = attention output (step 7 input), `m` = second normalised
  input, `h` = GELU output. `v` is FP16 (`f16(Wv a)`, `:178`) and the attention weights `A` are FP16 (`:187`).
  The residual stream `x1`, `x2` stays unrounded in the reference; the kernels carry it as an FP32 accumulator start
  (`clean/kernels/swin.hip:564,764`).
- Bounds that already exist in the design (READ):
  - `q8` saturates at +-448 (`clean/reference/fp8.py:7-8,33-36`); the kernels have a clamped and a `_nc` ("no clamp")
    variant, chosen per stage from the weights (`clean/FORMATS.md:94-101`).
  - The `_nc` proof gives static bounds per block from the weights alone: RMSNorm output has L2 norm <= max|gamma| sqrt(C);
    |q/k| <= ||Wqk row||2 ||a||2; |m| <= 1.07 max|gamma2| sqrt(C); GELU never increases magnitude. See `clean/FORMATS.md:101-102`.
  - Attention scores go through a cubic soft cap, `Sc - Sc^3/(3 CAP^2)`, with CAP = 32.369 and then tau = 0.35337
    (`clean/reference/swin.py:11-12,131-134`), then softmax: a bounded nonlinearity.
  - GELU: the default is a sigmoid form `x / (1 + exp(-z))`, with an FP16 polynomial form (`gelu_poly16`) clamping `u` to
    [-1, 1] and x to 448 (`clean/reference/swin.py:120-140`). A lookup-table GELU ending in `v_cvt_pk_fp8_f32` also exists
    (`clean/dx12/hlsl/wg_gelu_lut.hlsli:5-18`, "identical for all 2^32 inputs", proven on gfx12).
- Stated numeric ranges of the real data: none. The repo has no real weights or activations (`data/README.md`: "None of them holds model weights").
  Static bounds are per-block formulas, not measured values. UNKNOWN: actual activation histograms; measure by dumping `a`, `u`, `o`, `m`,
  `h` of each stage on a real frame (RDNA 4 machine, execute-only).

### 1.3 Accumulators and exactness (READ)

- Matrix products are `v_wmma_f32_16x16x16_fp8_fp8` (FP32 accumulate). K per product: C (64-128) for q, k, v and W1, `32 * heads` (64-128) for Wp,
  MLP (256-512) for W2, 4C (256-384) for merge, C_low (96-128) for expand (`clean/reference/swin.py:20-26`, `clean/FORMATS.md:32-35`).
  So one rounding after a chain of 4 to 32 WMMAs of 16 products each (INFERRED from K / 16).
- Biases are the accumulator start value (`clean/FORMATS.md:44`, `clean/kernels/swin.hip:564`).
- Exactness the design already depends on:
  - The sum of two FP8 values is exact in FP32 (`clean/FORMATS.md:17-19`).
  - The inner pack uses an optional integer-style bound argument (above).
  - The GELU lookup ends in `v_cvt_pk_fp8_f32` and claims identity "for all 2^32 inputs" on gfx12 (`clean/dx12/hlsl/wg_gelu_lut.hlsli:5-18`).
  - Dither quantisation in enc0 is three integer operations (`clean/PERF.md`, "Dither in 3 integer operations", checked over 6.7e9 inputs).
- Tolerances the specs accept (READ): reference runs in float64 with only the prescribed roundings (`clean/reference/swin.py:3-4`). Kernel
  tests accept spec 09: no non-finite value, <= 2 % of values more than one FP8 step off, mean abs error <= 1 % of reference RMS
  (`clean/tests/test_kernels_emu.py:97-127`). The default GELU is an approximation accepted on the whole-frame rule, not the per-stage one
  (`clean/reference/swin.py:97-101`). Whole-model verdict scale: MATCHES = 100 % equal (or >= 99.99 % with the rest 1 step off), CLOSE = >= 99 %
  equal and every difference <= 2 steps (`clean/EQUIVALENCE-PROTOCOL.md:8-12`).
- Measured on gfx12: the FP8 WMMA is not "exact sum, one rounding" but fits "FP32 rounding after C and after each group of 8 products"
  (2 of 51 values unexplained) (`clean/rr_f/head/DESIGN-55.md:171-178`, MEASURED-IN-REPO, from RR). Whether it applies to M is the same
  instruction, so likely (INFERRED).

### 1.4 Layer inventory and time share (READ plus my arithmetic)

Per frame, counting multiply-adds from the formulas in `clean/l/reference/table.py:289-333` (`block_macs_per_token` = QKV `2 * 32 * heads * C`, scores plus
weighted sum `2 * 16 * 32 * heads`, Wp `32 * heads * C`, MLP `2 * mlp * C`; merge `4 C C_out` per merged token, expand `C_low 4 C` per source token; enc0 80x64
and dec0 64x68 per token). Window attention is 4 x 4 tokens, head dim 32.

| Render size (padded grid F) | Level-0 tokens | Swin stages (10) | enc0 | dec0 | Total |
|---|---|---|---|---|---|
| 1280x720 (F 1280x736), 1080p Quality | 235,520 | 56.3 GMAC | 1.21 | 1.02 | 58.5 GMAC |
| 1706x960 (F 1728x960), 1440p Quality | 414,720 | 99.2 GMAC | 2.12 | 1.80 | 103.1 GMAC |

- Of the 58.5 GMAC, the FP16 `A V` products are about 0.90 GMAC (1.5 %); the rest is FP8 (INFERRED; consistent with "about 98 % FP8", `clean/FORMATS.md:85-88` as quoted by the analysis document).
- Per stage at 1080p Quality (GMAC): enc1 12.5, enc2 7.5, b0..b5 2.7 each, dec2 7.5, dec1 12.5.
- Blocks per frame: 10 Swin stages (2 + 6 + 2), each with 2-4 attention heads, one MLP, one projection; plus merge x2, expand x2.
- Time share in game (Cyberpunk 2077, 1080p Quality, RX 9070 XT, MEASURED-IN-REPO, `clean/l/SPEED.md`):

| Step | Model M (ms) | share |
|---|---|---|
| enc0 | 0.111 | 9 % |
| Swin (all ten stages) | 0.585 | 50 % |
| dec0 | 0.189 | 16 % |
| post | 0.289 | 25 % |
| total | 1.174 | |

- Standalone budgets: enc0 0.2531 ms, dec0 0.1337 ms at 1706x960 / 3840x2160 (the older perf milestone; `clean/PERF.md`): these stages are close to the
  DRAM floor, not the matrix rate (`clean/PERF.md` "Reading"). So the matrix path matters for the Swin stages (about half the frame), and little for enc0, dec0 or post.

## 2. Model L (clean/l)

- Map: 154 tensors; 68 `fp8` (1,586,816 elements), 44 `f16` (8,032), 36 `f16_scaled` (12,544, mul = div = sqrt(32)), `exp` 3, `u32_as_f32` 2, `u16_shr` 1
  (2,048) (`data/preset-L.v1.map.json`, tally). Source `W` is `f16le` (`sources.W`).
- Ten stages shared with M (`clean/l/reference/table.py:19`), plus width-32 blocks at pixel level for the encoder and the decoder, input linears 20 -> 32 -> 32,
  a 2 x 2 gather projection (128 -> 64) and a decoder projection (64 -> 128) (`clean/l/reference/table.py:55-95`).
- Format changes against M (READ, `clean/l/FORMATS.md:20-45`):
  - `x2` (output of the second input linear) and `y2` (decoder block output) are FP16 maps, 64 bytes per pixel.
  - Linears run "acc: f16": FP16 accumulate, then `relu_q8` or none (`clean/l/reference/table.py:55-66`).
  - The projection keeps a running FP16 sum per group of 32 input channels, clamps to +-6.28125, then FP8 (`clean/l/FORMATS.md`, `l_project_gather`).
  - Encoder 20 input values are dithered to FP8 (`q`, "values 0..19, bytes 20..31 zero").
- The oracle is being changed to an FP16 block form, "q7" (`clean/l/reference/oracle.py:84-129`, `clean/l/reference/table.py:99-106`): FP8 matrix op = up to 32 products plus an
  FP16 start value, FP16 result, chained over 32-wide chunks in ascending order; RMS norm in FP16 with a fixed tree; attention weights by FP16 polynomial and bit shift (no exp);
  FP16 GELU polynomial; residuals as accumulator starts. `P V` is one FP16 op.
- Two models of the arithmetic inside one FP8 op are carried and named in every record (`clean/l/reference/table.py:103-106`): `exact` (sum exact, one FP16 rounding; default) and
  `trunc13` (two blocks of 16 products, terms truncated to 13 bits below the largest, FP16 round per block). "What happens inside one op is not settled" (`clean/l/reference/oracle.py:81-82`).
  This matches the repo's own measurement of gfx12 (per-group rounding) only loosely; see section 5.
- The ten shared stages still run M's FP32-accumulate oracle (`PROVISIONAL["S8"]`, `clean/l/reference/table.py:115-118`), "their FP16 form is pending a decision".
- Matrix work (my arithmetic from `table.py` formulas at 1280x736 F): extra pixel-level work is `942,080 * (1,664 + 16,384 + 16,384)` = 32.4 GMAC, plus projections
  `235,520 * (8,192 + 8,192)` = 3.9 GMAC, so L = 56.3 + 36.3 = 92.6 GMAC against M's 58.5 (1.58 x). Measured network time 2.250 vs 1.174 ms (1.92 x) in game.
- Time share in game (MEASURED-IN-REPO, `clean/l/SPEED.md`): enc0 0.503 (22 %), Swin 0.751 (33 %), dec0 0.345 (15 %), post 0.651 (29 %), total 2.250 ms.
  The pixel-level width-32 blocks sit in enc0 and dec0, so about 38 % of L's frame is outside M's Swin stages.
- Statements about status (section 9): PR 223 says "the C = 32 block kernels and the oracle follow Model M's block form and are being reworked to the specification's block form".

## 3. Model K (clean/k)

### 3.1 Weights (READ)

- Source `T` is `base64` segments of 4,325,996 bytes plus `O` int32 (`data/preset-K.v1.map.json`). Map: 151 tensors; `f16` 120 tensors (1,480,872 elements), `f16_scaled` 27
  (234,400; mul = RN32(sqrt C) on the norm gain `g1`), `capped_softmax` 3 (24,576: the per-head 64 x 64 mixing blocks `M`, with cap 0x41700000 = 15.0, div 0x4428c000 = 675.0, sub 0x40600000 = 3.5), `i32` 1 (window offsets).
- So weights are FP16 values stored in float32 containers. The runtime keeps FP16 (`clean/k/FORMATS.md:3-5`, "Everything in model K is FP16" `clean/k/kernels/k_common.h:3-4`).
- No scale tensors. The `f16_scaled` factors are fixed constants.
- Weight blob layout: 16 x 16 FP16 tiles, 512 bytes each (`clean/k/FORMATS.md:38-44`).

### 3.2 Activations (READ)

- Every map is FP16 `[H][W][C]` (`clean/k/FORMATS.md:9-12`). Elementwise steps round to FP16 after every single operation (`clean/k/kernels/k_common.h:3-8`).
- Bounded or normalising elements:
  - Activation: `t = clamp(x, -2, 2)`, `p = a - b|t|`, `q = 0.5 + t p`, `act = x q`, with a = 0.412109375, b = 0.0810546875 (`clean/k/reference/act.py:4-8`).
    Evaluating the formula at the clamp: q(+2) = 0.5 + 2 (0.412109375 - 0.162109375) = 1.0 and q(-2) = 0, so beyond |x| = 2 the function is ReLU-like and
    unbounded for positive x (INFERRED; arithmetic mine).
  - Softmax weights: scores clamped to +-21.640625 (`clean/k/reference/network.py:29`, `:53-59`), then FP16 weights.
  - RMS norm: FP32 sum of FP16 squares, `r = f16(1/sqrt s)`; `s = +inf` or 0 give r = 0 (`clean/k/README.md`, "Design notes").
  - FP16 overflow at 65504 is the only other range limit; a NaN or inf input stays NaN (`clean/k/kernels/k_common.h:93-97`).
- Stated ranges of real data: none (no real parameters available to the clean team, `clean/k/README.md`: "Real parameters are not available to the clean team").
  UNKNOWN; measure by dumping every `S0..S4` skip map and the head output on a real frame.

### 3.3 Accumulators (READ, MEASURED-IN-REPO)

- Default (`fused`): `v_wmma_f32_16x16x16_f16`, FP32 accumulate, one FP16 rounding per product ("Biases are the starting value of the FP32 accumulators, so `f16(W x + b)` rounds once"
  `clean/k/README.md`, Design notes). Spec 41 as written.
- Opt-in `fused16`: `v_wmma_f16_16x16x16_f16`; "rounds its FP16 accumulator to nearest even after every 16-deep step (measured on the RX 9070 XT; the same throughput as the FP32-accumulating form)" (`clean/k/README.md`, "Speed approximations").
  Merge keeps FP32 (K = 4C up to 512).
- Tolerance (spec 49 d4, MEASURED-IN-REPO, `clean/k/README.md` table): per stage mean 2.5e-3 and max 3e-2 of reference RMS; whole network d0 head mean 8e-3 and max 7e-2.
  Measured, FP16 accumulate: 1.06e-3 mean / 1.32e-2 max per stage, 3.4e-3 / 3.5e-2 head; FP32: 4.4e-5 / 5.4e-3, 1.6e-3 / 2.8e-2. All within limits, the head at about half.
  Adding the residual into the Wo / W2 start failed the d0 head max (7.4e-2 vs 7e-2): "every rounding of the running sum then happens at the magnitude of x".
- Longest sum: merge, K = 4C up to 512; blocks K up to 4C = 512 for W2 at C = 128 (hidden = 4C).

### 3.4 Layer inventory and time (READ, MEASURED-IN-REPO)

- 11 stages (e0..e4, m, d4..d0), channel width 32 to 160, 8 x 8 token windows (`clean/k/FORMATS.md`, `clean/k/reference/layout.py`).
- At 2560x1440 output, s = 0.5 (base 640x384), per frame 43.1 GFLOP (about 21.5 GMAC; GFLOP / 2, INFERRED), by stage (GFLOP): e0 9.58, e1 6.41, e2 2.19, e3 1.47,
  e4 0.69, m 0.31, d4 0.69, d3 1.47, d2 2.19, d1 8.03, d0 10.09 (`clean/k/README.md`, speed table).
- Time: network 1.026 ms (FP32 default) or 0.878 ms (`fused16`); in game GPU time about 1.8 ms per frame with `fused`, 1.588 ms with `fused16` plus the compact input pass
  (`clean/k/README.md`). Per-stage achieved rate 13-64 TFLOP/s (levels 0 to 2 about 40-59).
- The "k_block" kernels use 11 matrix call lines, `k_block_wide` 13, all F16 -> F32 (`k_block.hlsli`, `k_block_wide.hlsl`; analysis section 3.1).
- Layout dependence: the block kernel needs the gfx12 accumulator layout for the swapped `v^T = n^T Wv^T` product, so the packer refuses any other `d_row` (`clean/k/FORMATS.md:50-51`,
  `clean/k/runner/k_pack.py:164-165`).

## 4. Ray Reconstruction preset F (clean/rr_f)

### 4.1 Weights (READ)

- Source `W` is `f16le` (`data/preset-F.v1.map.json`). Map groups: 18 groups, 303 tensors:

| dtype | op | tensors | elements |
|---|---|---|---|
| e4m3 | `e4m3` | 104 | 2,903,680 |
| float16 | `f16` | 121 | 12,308 |
| float16 | `score_bias` (16 attention-bias tables) | 16 | 208,896 |
| float16 | constants (`const_f16x2`, `const_f16lo`, `const_f32_to_f16`, `attn_wmax`) | 16 | 16 |
| float32 | `u32_as_f32`, `const_u32_as_f32`, `exp` | 5 + 36 + 3 | 2,056 + 36 + 3 |
| uint16 / uint32 | `u16` (sigmoid-class table), `u32` (shifts) | 1 + 1 | 2,048 + 44 |

- So weights are stored directly as E4M3 codes (`dtype e4m3`, container version 2 adds code 6 `e4m3`, `clean/converter/OUTPUT-FORMAT.md:36`); the runtime
  arranges them into 256-byte fragments at load ("by our loader, not the converter", `clean/rr_f/DESIGN-51.md:62`). Bias values are FP16, widened exactly to FP32 as accumulator
  starts (`clean/rr_f/DESIGN-51.md:65`). No scale tensors for the weights; constants only (attention scale and clamp constants A_SCALE, A_CLAMP, A_P1, A_P0 in FP16).

### 4.2 Activations (READ)

- E4M3, one byte per channel, "Q8" of an FP16 (`clean/rr_f/DESIGN-51.md:57`), saturating, with the clamp `v_maximumminimum_f32(h, -448, 448)` keeping NaN (`:157`).
  Attention weight P <= 1.0005, so Q8 of P has no clamp (`:167`). The activation function (step 8) clamps `k` by `pk_max / pk_min` (`:170`). Sigmoid via a 64 K-entry FP16 table (`:30`).
- Hidden state: 4 FP16 values per token (`:68`); hidden update uses `sig g` multiplications in FP16.
- Attention weight: `u = s * A_SCALE`, clamp, two packed FMAs, then an integer expression (`bits(w) = (bits(p) << 5) & 0x7FE0`), "no transcendental, no rounding" (`:28-29`).
  The converter checks that a clamped score stays in range (`:288`).
- Ranges: spec ranges for the constants exist only as converter checks, not as numbers in the tree. UNKNOWN; see "What to measure".

### 4.3 Accumulators (READ, MEASURED-IN-REPO)

- Spec 51.1's step is "R(C + sum of 32 products)", one FP16 rounding per 32 inputs, start value the FP16 bias or previous partial (`clean/rr_f/DESIGN-51.md:21-24`, `clean/rr_f/README.md`, `reference/fmt.py`).
  One step = two FP8 WMMAs on an FP32 accumulator, then `v_cvt_f16_f32` and back. Chain length per rounding: 2 WMMAs. Longer products (K = C up to 160) are `C/32` chained steps with an FP16 round each. The opt-in variant L
  keeps FP32 over the whole chain ("an occasional one-step difference", `:209-211`).
- MEASURED-IN-REPO: the FP8 WMMA does not return the exact sum. "f32 rounding after C and after every group of 8 products (K 0-7, 8-15 of each WMMA) 2" of 51 values differ, exact model 51
  (`clean/rr_f/head/DESIGN-55.md:171-178`). On real-like data only 72 differing values over 4 stage kinds, 11 kinds identical (`clean/rr_f/gpu/README.md:72-79`).
- Design already depends on exactness in places:
  - Step X certifies a step with an error-free transform `e = WMMA(a, b, -p)` (`clean/rr_f/gpu/README.md`); its 15-call total is 29.70 ms against 5.31 ms for step E (same table), about 5.6 x.
  - Step I is exact INTEGER: int8 digit planes, int32 accumulate, one rounding to FP16 (`clean/rr_f/kernels/rr_step_i.hlsli:1-33`):
    every E4M3 value is `M * 2^(E - 10)`, M = 0..15; an operand block with exponent span `span <= 14` becomes 2 or 3 signed 7-bit digit planes (1 plane when span <= 3);
    one WMMA per digit pair per K-block; the tile-step is taken only when the int32 bound `L1' * Bmax' < 2^31 - 2^15` holds, else the window is flagged and recomputed by step X.
    Scales are derived exponents (per 16-output tile and 32-input step for weights, per wave and step for activations), not stored scales.
  - Hardware fact H1' for step I: `v_wmma_i32_16x16x16_iu8` is exact in int32 below 2^31 (`clean/rr_f/gpu/probe_i8.py`; MEASURED-IN-REPO per the analysis document).
- The design document's own list of exactness risks: R1 matrix accumulation, R2 Z order (spec-accepted), R4 compiler fusing FP16 ops, R6 NaN code sign (`clean/rr_f/DESIGN-51.md:305-322`).
- Tolerances: per-output in the test vectors; bit-exact where Z is supplied, one code in rows whose f32 Z differs (`clean/rr_f/README.md`, Vectors).

### 4.4 Layer inventory and time (READ, MEASURED-IN-REPO)

- 16 window stages in two groups plus head: IN0, E1..E5 (encoder), D4-F..D0-F (full), D4-R..D1-R, D0-R (recurrent). Channels and heads: 32/2, 64/2, 96/4, 128/4, 160/8.
  Window 8 x 8 (64 tokens), 4 waves (`clean/rr_f/DESIGN-51.md:10`).
- WMMAs per 16-token tile (`clean/rr_f/DESIGN-51.md:340-358`): IN0 90, E1 268, E2 550, E3 856, E4 1518, E5 1418, D4-F 1508, D0-F 96, D0-R 88; VALU per tile from 1,613 to 16,520.
  Estimated WMMA share of the cycle count is only 28-41 % (`:340-358`, DESIGN estimate, +-40 %).
- Measured GPU time at 1440p Quality (render 1706x960, synthetic data, step E, `clean/rr_f/gpu/README.md`): 15 calls 5.31 ms; IN0 1.023, E1 0.580, E2 0.311, D1-F 0.552, D0-F 1.120. D0-R adds
  1.015 ms (`clean/rr_f/head/DESIGN-55.md:187-190`). Level 0 (IN0 + D0-F + D0-R) is about 3.2 of 6.3 ms, 51 %.
  "Levels 0-2 are VALU-bound (the f16 rounding of every step, the attention weight and every Q8 with its NaN-sign handling)" (`clean/rr_f/gpu/README.md`). The exact step X total is 29.70 ms.
- The head: D0-R's head is one dispatch, 88 FP8 WMMAs per wave, 14 KB of weights; its head-only kernel is DRAM-bound (293.7 us) (`clean/rr_f/head/DESIGN-55.md`).

## 5. Frame generation (clean/fg)

### 5.1 Weights and activations (READ)

- "Two networks, float16 elements, ReLU activations, no attention, no FP8" (`clean/fg/FORMATS.md:9-10`).
- net1: 17 blobs, each `[weights][zero area][bias x 16]`; net2: 25 layers (weight blob plus bias blob each). Map: 84 tensors, all `f16`, 1,439,840 elements (`data/fg.v1.map.json`, tally);
  source blocks `A01..A17`, `B01w/b..B25w/b`, element `f16le`.
- No scale tensors, no FP8. The map is "natural order", and "every value checked to be exact in float16" (`clean/fg/FORMATS.md:45-46`).
- Activations: FP16 maps, leaky ReLU `y = max(x, h(s x))` with s = 0.0099945068359375 (FP16 0x211E, "the delivered value"), max-pool, align-corners bilinear upscale with FP16 steps, FP16 add
  for skips (`clean/fg/FORMATS.md:62-82`). net1 input features include a warp error clamped to [0, 1] and a mask (`:47-48`). Range of intermediates: not stated; UNKNOWN.

### 5.2 Accumulators (READ, MEASURED-IN-REPO)

- Delivered rule for net1: "Float16 accumulator started at the bias; 16-term chunks (...) each summed exactly and added with one float16 rounding" (`clean/fg/FORMATS.md:60-62`).
  So the design's matrix step is a 16-deep exact sum, one FP16 rounding per chunk, the accumulator being FP16. Chunk orders per blob are fixed (`:62-66`).
- net2: "the accumulation order" is OPEN item 9 (`clean/fg/FORMATS.md`, OPEN list); only exact accumulation is implemented (`acc="f16"` raises `OpenItem`, `:51-53`).
- MEASURED-IN-REPO (`clean/fg/records/wmma-probe-2026-11-14.json`, `clean/fg/README.md`): gfx12's F16 -> F32 WMMA is not the exact chunk. On ordinary data (62.4 million outputs):
  FP32 accumulate rounded per chunk is more than one step off in 1,216 outputs and more than four in 211; FP16 accumulate in 6.05 million and 1.86 million.
  The tier-B fast path therefore is "Not bit-exact to the CPU reference"; gate is the generated frame against the oracle, not net1's output (`clean/fg/FORMATS.md`, fast path).
- Exact path (integer arithmetic, `clean/fg/kernels/fg_exact16.h`) exists as validation mode and fallback.

### 5.3 Inventory and time (READ, MEASURED-IN-REPO)

- net1: 32 steps (five encoder convolutions with pool, two 1 x 1 bottlenecks, five decoder levels of upscale, add and two 1 x 1) on a 2 x 2 folded 48-channel grid
  (`clean/fg/FORMATS.md:56-62`). net2: 25 3 x 3 layers, two sub-nets, `net2_macs_per_pixel` formula in `clean/fg/layout.py:168-180` (not evaluated; the levels are partly OPEN).
- Time (net1 only, synthetic, RX 9070 XT, `clean/fg/records/net1-fused-2026-11-14.json` and `net1-tierb-2026-11-14.json`):
  fused path (11 launches) 1080p: HIP 2.478 ms, D3D12 5.585 ms (after driver conversion; 10.4 ms before); 1440p HIP 3.566 ms, D3D12 7.233 ms. Conv 2.3 ms, pair 1.3 ms of the 1080p HIP total.
  Tier B (27 launches) 1080p HIP 3.219 ms, 1440p HIP 5.351 ms. These numbers are for the network only, no motion or warp stages.
- Matrix call lines: `fg_mconv.hlsl:53`, `fg_mpair.hlsl:134` (2 F16 lines, analysis 3.1).

## 6. Anything else in the tree

- Converter maps for K, L, M, F and FG exist (`data/`). There is no map or code for any other model; no int8 or FP16-only model of the upscaler; the `clean/wgshared/EXTENDING.md:29` mentions `body: "int8"` only as an example name for a new kernel variant, not as a model.
- `clean/fsr` is the AMD FSR front end (a FidelityFX provider), not a WrongGPU model.
- `clean/equiv` and `EQUIVALENCE-PROTOCOL.md` are test tools comparing WrongGPU with the original on the spec team's side; no format statements beyond the verdict rules.

## 7. What the tree says about the original models' formats (quotes)

The clean-room rules mean the tree describes the original only through the converter maps and spec-derived statements. Everything below is quoted, nothing is added.

| Quote | Where | What it says |
|---|---|---|
| `"sources": {"W": {"element": "f16le", ...}}` for M, L, F, and FG's `A01`.. | `data/preset-M.v1.map.json`, `preset-L`, `preset-F`, `fg.v1.map.json` | The player's DLSS file stores FP16 weights for M, L, F and FG. K's source is base64 text segments (`T`) holding FP16 values after the `f16` op. |
| "e4m3" op on F's 104 tensors; "fp8" op on M's 56 and L's 68 | map `op` entries | The model definition quantises FP16 file values to E4M3 for these tensors. "Weight quantization is part of each FP8 model's definition (the references apply `q8`)" (quoted from the analysis document section 3.2, citing `clean/reference/swin.py:244`; line not re-checked here). |
| "What the original's matrix unit does with them is not specified" (about inf and NaN operands) | `clean/fg/reference/fp16acc.py:14` | The reference calls the original's hardware a "matrix unit" and leaves its inf / NaN behaviour open. |
| "The original sums in float16 in its matrix unit's order (OPEN)" | `clean/fg/FORMATS.md:51-52` | FG net2's accumulation order is not delivered. |
| "can only be judged on hardware" | spec 51.1, quoted in `clean/rr_f/head/DESIGN-55.md` ("Spec 51.1 ('can only be judged on hardware', 51.7 item 3) leaves this open") | RR's spec says the exact rounding of the matrix sum cannot be fixed on paper; it is a hardware question. |
| "for real matrix hardware **[Q1]**" | `clean/rr_f/DESIGN-51.md:203` | Spec 51.1 anticipates "an occasional one-step difference" for long-chain accumulation on real matrix hardware. |
| "an FP8 op takes up to 32 products and an FP16 start value" | `clean/l/reference/oracle.py:78-80`, table `consts` | Model L's FP16 block form follows a spec step with 32-product FP8 ops, FP16 accumulator. "What happens inside one op is not settled" (`:81`). |
| "Model L ... spec side states the order" (float32 blend, fused forms) | `clean/l/FORMATS.md` (post section) | Post stages of L run in FP32 with named fused forms. |
| Delivered slope "0.0099945068359375 (float16 0x211E). Nearest-even rounding of 0.01 would give 0x211F; the delivered value is used as given." | `clean/fg/FORMATS.md:67-68` | FG's leaky ReLU slope is an FP16 constant, i.e. the original runs FP16. |

- There is no sentence in the tree that says the original DLSS runs INT8, or TensorRT, or names a tensor-core format. I searched `int8`, `tensorrt`, `tensor core`, "the original's hardware" and "models the original" across `.md`, `.py`, `.h`, `.hlsl*`, `.hip`, `.json`. The only hits were unrelated container dtypes (`uint8`, `int8` as snorm grid codes).
  The reading "E4M3 weights and FP8 matrix work, FP16 accumulate per 32 (M, L in the oracle; RR spec) or per 16 (FG)" is therefore the spec side's statement through the oracles, not a claim about the vendor's hardware.
- INFERRED, not a quote: M and RR's choice of E4M3 for both operands, saturating at 448, matches a hardware FP8 matrix instruction on the original's side; the tree never says so.

## 8. Suitability ratings (all INFERRED, reasons from sections 1-5)

Definitions. "INT8 matrix path" means `v_wmma_i32_16x16x16_iu8` on RDNA 3, the form AMD uses for FSR 4.1 (analysis option b), with a single int8 plane per operand unless stated. "FP16 matrix path" means
`v_wmma_f32_16x16x16_f16` carrying E4M3 or FP16 values in FP16 (analysis option a). A model suits INT8 when its operands are already bounded and uniformly scaled; unbounded FP16 values and E4M3's wide exponent make single-plane INT8 lossy.

| Model | INT8, single plane | INT8, digit planes (exact) | FP16 | Reasons |
|---|---|---|---|---|
| M | Poor | Feasible, slow | Good | E4M3 operands span 2^-9 to 448 (exponent field 1 to 15): an exact int8 form needs `span` up to 14, so 2 to 3 planes (RR step I `clean/rr_f/kernels/rr_step_i.hlsli:12-20`); one plane covers span <= 3. Weights have no scale tensors, so int8 weights would be a new model (analysis option b). Activations are bounded (per-block static bounds, saturation at 448, soft-capped scores), which helps calibration, but E4M3's relative precision (3 mantissa bits) is not an int8 grid. FP16 holds every E4M3 value exactly (analysis 4.3). |
| L | Poor | Feasible, slow | Good | As M. The FP16 maps `x2` / `y2` are unbounded FP16 (pixel-level path), so int8 would need re-quantisation there too; those maps are already FP16, which suits an FP16 path. The L oracle already uses FP16 accumulate per 32-wide FP8 op, closer to an FP16 matrix semantic than M's. |
| K | Poor | Not applicable (no E4M3 form) | Native, best fit | FP16 weights, FP16 activations rounded after each operation, no quantisation anywhere. Activation is ReLU-like above |x| = 2 (unbounded), norm output depends on data. Accumulation tolerance is documented and generous (FP16 16-deep accumulate passes spec 49), so an FP16 path with FP32 accumulate has margin. K is the one model whose arithmetic already is "FP16 x FP16", gfx12 uses `v_wmma_f32_16x16x16_f16` for all of it. |
| RR F | Fair via digit planes (already built: step I) | Built, exact where bounded, falls back to step X | Good | E4M3 operands, per-step rounding to FP16 designed on exact sums; step I is exact by construction and flags windows where the int32 bound fails. The design is VALU-bound (28-41 % WMMA cycles), so a faster matrix rate alone gives a limited gain (`clean/rr_f/DESIGN-51.md:340-358`). |
| FG | Poor | Not applicable | Native, best fit | FP16 weights and FP16 chunk accumulation of 16 products: the delivered rule is a 16-deep matrix instruction with an FP16 accumulator. Leaky ReLU and warp-error features give unbounded or unscaled FP16 activations; tolerance of the exact sum is "gate is the generated frame", i.e. approximation is accepted at network level. |

Notes on the ratings:

- "Good" for FP16 on M, L and RR is about fidelity of operands only. Analysis section 4.5 gives the cost: matrix peak per WGP-clock about 1/4 of gfx12's FP8, 4 x operand VGPRs. This lane did not test that.
- Where a model already tolerates accumulation approximation (K at spec 49 limits, FG by gating on the generated frame), an FP16 or other approximate matrix path is more likely acceptable than where the contract is byte-identity (M, L, RR; analysis section 4.1).
- An important INFERRED point for any INT8 plan: a single plane works only when the operand block's exponent span is at most 3. The per-block exponent bases (RR step I's `beta` per 16-output tile and 32-input step, per wave and step for activations) are derived at run time; no stored per-channel scale exists in any model, so "INT8 with per-channel scales" (analysis option b) means producing new weights not present in the player's file.

## 9. Upstream: models in progress (read-only `gh`, phantoslayer21/wronggpu, today)

Queried `gh pr list --state all --limit 60` (newest PR #228), `gh issue list --state all --limit 80`, issue and PR bodies for #28, #34, #89, #96, #15, #27, #217, #218, #223, #225. ROADMAP.md and RELEASE-NOTES.md read from the snapshot. I posted nothing.

| Item | State | What it says (format or speed statements quoted) |
|---|---|---|
| Model L, issue #89 | open; code merged in many PRs (#163-#203); runtime option `model=L` exists | Issue: "Players are asking for Model L as a second upscaling model next to Model M"; plan was converter map, CPU reference, kernels bit-exact against the reference, game tests. README says "in development for 0.7.0"; PR #223 (open) rewords to "planned for 0.6.5, off by default and still under test". |
| Model L numerics, PR #218 (merged) | merged | "Model L's oracle: both width-32 blocks now use the FP16 form, and the decoder block reads the encoder's skip. The current C32 kernels fail it, and that's the point of this PR." FP8 op: "up to 32 products and an FP16 start value per op"; `exact` vs `trunc13` "is not settled". Ten shared stages "still run Model M's oracle (`PROVISIONAL["S8"]`, a follow-up)". |
| Model L, PR #223 (open) | open | "the C = 32 block kernels and the oracle follow Model M's block form and are being reworked to the specification's block form, so no check here is a result against the specification yet." |
| Model L speed | PR #188 merged; PR #217 open (speed baseline for 0.6.5: M and L rows; "The Model K rows and the Ray Reconstruction (preset F) rows follow") | Frame span 1080p Quality M 1.366 ms, L 2.276 ms; 1440p Quality 2.273 / 3.833; 1440p Performance 1.596 / 2.783. Speed items "S1 (Model M) and S2 (Model L)" judged on game-route kernels. |
| Model K (Preset K), issue #15 and ROADMAP | open; "Next release (0.6.1)" in ROADMAP; RELEASE-NOTES: "returns in 0.6.1" | PRs #194-#215 merged: K D3D12 ports (narrow, wide, input, output, runtime), "byte-identical to HIP". PR #211: launcher choice K and front-end build state "off by default". Issue #216 (open): launcher builds without K mishandle a saved K choice. PR #213 (open): OptiScaler plugin path honours the launcher's model choice (L, K). |
| Frame generation, issue #96 | open, milestone "Later" (README) / "After Model L and Model K (milestone 0.7.0)" (issue) | Issue comments by the maintainer: "every Frame Generation model configuration the DLSS file carries is in scope, not one of them, and all on Radeon RX 9000 first"; pacing via the Hardware Flip Queue. Merged: PR #173-#205 (map, blobs, CPU reference, net1 wiring, kernels, tier-B matrix path, fused 11-launch path; "D3D12 within 1.5x of HIP" in PR #205 title). PR #193 "probe of the gfx12 matrix instructions against net1's chunk arithmetic". net2 wiring still OPEN in `clean/fg/FORMATS.md`. |
| Ray Reconstruction preset F | shipped; issues #16 (Hogwarts Legacy RR greyed out), #27 (RR speed), #35, #36 | ROADMAP: "Ray Reconstruction speed: Make Ray Reconstruction cheaper per frame on RX 9000 cards without changing its output: faster kernels for the body stages and the output stage, fewer passes, better use of the hardware's matrix instructions." |
| RDNA 3, issue #28 and ROADMAP | open | "Running on RDNA 3 cards needs kernels without RDNA 4's FP8 matrix instructions and a check of what the results and the speed would be. This starts as an investigation." Issue #28: "which kernels change, what arithmetic replaces FP8, how close the results stay, and what the speed could be?" |
| Optional faster Model M mode, issue #34 | open | "An optional, clearly labelled faster mode could allow small output differences within a published quality limit, chosen by the player. ... The default stays exact." Relevant to any approximate RDNA 3 path. |
| Non-RDNA-4 cards, PR #225 (open) | open | "On a GPU that is not an RX 9000-series card, WrongGPU's networks cannot run (their kernels are built for RDNA 4 only)"; sends such cards to the front end's own upscaler at once and removes automatic HIP routes in a game. PCI ids 7550 / 7551 / 7590 / 7591. |
| Linux launcher, PR #221 (open); Linux/Vulkan issues #29, #51 | open | Not a model; context for the Linux target of the RDNA 3 request. |
| Newer DLSS presets or other models | none found | No issue, PR or file mentions a preset other than K, L, M and F, or any model other than frame generation (FG 310.x, `data/fg.v1.map.json`: "frame generation 310.x: DL1Net (17 layers) and DL2Net (25 layers)"). |

Statements about format or speed in upstream text (quoted above): the FP16 form for L (#218, #223); "approximations only if a published quality limit" (#34); "bit-identical output" demanded for RR speed work (ROADMAP, #27); K "byte-identical to HIP" (#204-#215).
UNKNOWN: whether the maintainer plans any model beyond K, L, FG and F; the issue list and PR titles show none. Settling: ask, or watch ROADMAP.md.

## 10. Open questions

1. Are real activation ranges per stage available anywhere? The tree has static formulas only. Without a dump, any INT8 scale choice (and the single-plane claim) is a guess. Who can produce an execute-only range dump on an RX 9070 XT?
2. Does the maintainer want fidelity to RDNA 4's bytes (M, L, RR) or "CLOSE" (K, FG style)? This decides between FP16 carrying E4M3 (analysis option a) and an approximate INT8 path.
3. Which L block form will win: Model M's FP32-accumulate form, or the FP16 "q7" form (`exact` vs `trunc13`)? The answer changes what an RDNA 3 kernel must reproduce (`clean/l/reference/table.py:103-106`; PR #218, #223).
4. K's decoder and block kernels need the gfx12 accumulator layout for `v^T` (`clean/k/FORMATS.md:50-51`). Is K the first RDNA 3 target because its operands are FP16 and need no widening (analysis lane 1 decides)?
5. net2 of frame generation: accumulation order, grid levels, wiring are OPEN (`clean/fg/FORMATS.md` OPEN 3, 5, 9). Not resolvable from the tree.
6. Is the `trunc13` model a measurement of the original's hardware or a guess? The tree calls the op "not settled".

## 11. What to measure on an RX 7900 XT

1. For each of M's 14 stage kinds, K's 11 stages and RR's stage kinds: the distribution of the maximum exponent span per 16-output tile and 32-input step (weights) and per wave and step (activations) from RDNA 4 dumps. This sets how many digit planes an exact IU8 path needs. Method: copy `clean/rr_f/gpu/lever_rates.py`'s logic for M.
2. `v_wmma_f32_16x16x16_f16` on gfx11 with E4M3-valued FP16 inputs: whether the result equals gfx12's FP8 WMMA bits (analysis T1 / T2) and the group size of its FP32 roundings (per 8, 16 or exact).
3. `v_wmma_i32_16x16x16_iu8` exactness below 2^31 on gfx11 (`clean/rr_f/gpu/probe_i8.py`, H1') and its rate against the F16 form.
4. FP16-accumulate form (`v_wmma_f16_16x16x16_f16`) on gfx11: rounding per 16-deep step, as measured on gfx12 for K (`clean/k/README.md`, "Speed approximations"); needed by K's `fused16` and FG's chunk rule.
5. Dump M's per-stage activation histograms (max, 99.9th percentile, count above 8 / 64 / 448) on a real frame, to size int8 headroom.
6. The share of FP16 `A V` in M's time (the 7 F16 call lines) to confirm the 1.5 % estimate; RGA ISA counts per kernel on gfx1100.
7. K's and FG's end-to-end error against RDNA 4 dumps with FP32-accumulate and FP16-accumulate, using the spec 49 limits and the FG probe classes (`clean/fg/runner/fg_wmma_probe.py`).
8. Per-stage time of M (enc0, Swin, dec0, post) on the 7900 XT in the same harness as `clean/l/SPEED.md`, to compare against the 50 % Swin share above.
