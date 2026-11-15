# INT8 matrix multiplies for WrongGPU on RDNA 3: what quality would be lost (CPU study)

Date 2026-11-14. Source tree: WrongGPU upstream main 994f304 (`src/` next to this folder), read-only. No GPU was used.
Labels: **MEASURED** (with the data named), **INFERRED** (reasoning shown), **UNKNOWN**.

Data. Model M: `<WrongGPU data>/model/M/model.wgcm` (the user's converted Model M; on this machine it sits directly in
the `model` folder). RR preset F: `<WrongGPU data>/model/F/model.wgcm`. Both were read in place. Nothing derived
from them was written except the aggregate numbers in `results/`. No other real model (K, L, FG) was found in either
user folder. No number in this file comes from a synthetic model.

## 1. Headline

| Scheme (matrix work per original 16x16x16 multiply) | Multiplies | Est. cycles per layer step | Model M result on probe frames | Verdict |
|---|---|---|---|---|
| FP16 route (E4M3 widened to F16, F16 WMMA, software E4M3 quantizer) | 1 | 322 (measured budget) | CPU proxy (FP32 accumulation of the exact products): every stage MATCHES, 100 % equal | usable as is (quality); the slow one |
| INT8, one plane, any scale granularity tested (S1-S4 weights, with or without int8 activations) | 1 | 59 (74 with a half-wave exchange); +~17 per K tile for per-block scales | every final output DIFFERENT; network-output SNR 17-18 dB; display PSNR 34-35 dB worst case, 45-46 dB mean | not viable as a drop-in; would need calibration frames and a new accuracy contract, and no scale choice fixed it here |
| INT8, two planes on both operands (S5 weights + two-plane activations) | 4 (3.8-3.9 measured) | ~330 (formula in 4) | MATCHES at every stage, bit-identical to the reference | usable as is; exact, but no faster than the FP16 route |
| INT8 adaptive exact (1, 2 or 3 planes per block, rr_step_i rule) | 3.81 (block scales) / 3.92 (row scales) | ~310-330 | exact by construction (equals the line above on this data: no block needed 3 planes) | usable as is; same cost |

The short answer: for Model M, a single INT8 plane loses too much (MEASURED on probe frames, every scheme), and every
exact INT8 form needs two planes per operand, which is four multiplies and costs about what the FP16 route costs
(INFERRED from the measured cycle counts). INT8 does not give RDNA 3 a fast route that keeps Model M's output.

## 2. Method

Scripts (all here; `run.sh <WrongGPU data folder>` runs everything under an 8 GB memory cap, no swap, 8 cores, `nice`):

| File | What it does |
|---|---|
| `qlib.py` | Reads `model.wgcm` (container 1 and 2, e4m3 decoded), picks the matrices from the converter map (op `fp8` / `e4m3`, else FP16 matrices) and orients them as [out, K]; the int8 schemes; exponent spans; digit counts |
| `int8_study.py <model.wgcm> <map.json>` | Step 1 (weights), model-agnostic. Prints a markdown table, writes JSON of aggregates |
| `m_e2e.py <model.wgcm> --src <src>` | Steps 2-4 for Model M: the repo's float64 reference re-composed so that every matrix product goes through one hook. A self-check confirms the re-composition equals the repo's own `encode` / `run_stage` functions bit for bit at every stage (MEASURED: true for all cases) |
| `report.py` | Turns `results/e2e-M.json` into `results/e2e-M.md` |

Weight schemes (on each matrix as [out, K]; integer range +-127 unless noted):

- S1 one real scale per tensor; S2 one real scale per output row; S3 per output row with the clipping point chosen
  from 31 candidates (0.25-1.0 of the row maximum) to minimise squared error;
- S4 one power-of-two scale per 16x16 block, one plane, round to nearest (exact when the block's values fit);
- S5 two planes: per 16x16 block power-of-two scale with integer range +-16383 (= d0 + 128 d1, |d| <= 127);
- S6 the exact digits of `rr_step_i.hlsli` (1, 2 or 3 signed 7-bit digits, count ceil((4 + span) / 7)).

Exponent span = Emax - Emin over the nonzero values of a row or block, with E as in `rr_step_i.hlsli:5-6`
(E = max(exponent field, 1)). Span <= 3 covers 4 binades (one plane exact); span <= 10 covers 11 binades (two planes
exact). A second count from the lowest set bit of the values gave the same plane numbers to three decimals.

Activation schemes (operand [tokens, K]; uint8 0..255 when the whole tensor is non-negative): per tensor (real scale,
from the same input: the optimistic "dynamic" case), per token (real scale per row), one plane per 16-token x 16-K
tile (power-of-two scale), two planes per tile.

End to end: probe inputs from `clean/equiv/probes.py` (cases `pan`, `disocclusion`, `checker`, `exposure_bright`,
seed 1, frame 1), render 128x72 (feature grid 128x96, output 192x108, reconstruction 288x162), exposure from
`clean/reference/exposure.py`, reset frame (no history), exact GELU (the reference default), attention A V kept
FP16 x FP16 as in the reference. Stages enc0 ... dec1, dec0 (h, A, B, C, latent) and the post reconstruction
(`display`) are compared with `clean/equiv/compare.py`'s `compare_stage` (the protocol's MATCHES / CLOSE / DIFFERENT
rules, 2-step band, in memory: nothing is dumped). Runs: chained (the network on its own outputs) and isolated (each
stage fed the baseline's inputs). Also: display PSNR (peak = the baseline display's maximum), largest absolute error,
and the SNR of the network output `dec0.h` against the baseline.

## 3. Results

### 3.1 Weights (MEASURED: Model M 56 matrices, 1,549,568 weights; RR preset F 104 matrices, 2,903,680 weights)

| Model | S1 rel RMS | S2 | S3 | S4 | S5 | nonzero -> 0 (S1 / S2 / S4) | S4 weights kept exact | rows span <= 3 / <= 10 | 16x16 blocks span <= 3 / <= 10 | S6 digits per block |
|---|---|---|---|---|---|---|---|---|---|---|
| M | 1.15e-2 | 6.87e-3 | 6.87e-3 | 6.87e-3 | 0 | 1.2 / 0.2 / 1.4 % | 87 % | 8 / 100 % | 4 / 100 % | 1.96 |
| RR F | 1.12e-2 | 6.69e-3 | 6.68e-3 | 6.90e-3 | 0 | 1.2 / 0.6 / 1.4 % | 87 % | 12 / 100 % | 14 / 100 % | 1.86 |

Block span histograms (span 0, 1, ... 8; none above 8):
M `0, 0, 0, 236, 3769, 1965, 82, 4, 0`; RR F `2, 0, 284, 1347, 704, 3617, 4607, 766, 69`. Per tensor: `results/weights-M.md`,
`results/weights-F.md`.

- MEASURED: every weight block of both models fits two planes exactly (span <= 8 everywhere); none needs a third.
- MEASURED: only 4 % (M) and 14 % (RR) of 16x16 blocks fit one plane exactly. The typical block spans 4-6 binades
  beyond the first, i.e. 5-7 binades in all.
- MEASURED, a surprise: MSE-optimal clipping (S3) gains nothing over plain per-row max scaling (S2): identical to 3
  digits. E4M3 weights have no long tail that clipping could trade away. Per-block power-of-two scales (S4) are no
  better than per-row real scales (S2) in RMS: they keep 87 % of weights exact but flush more small ones to zero.
- MEASURED: the few blocks that fit one plane are concentrated in a few tensors (M: `enc1.Wm`, `dec1.We`,
  `dec2.We` at about 83 %; `b0.Wqk` 44 %; RR: the 3- and 4-output heads `Wh`).

### 3.2 Activations (MEASURED: Model M, four probe cases, all ten Swin stages plus enc0 / dec0)

| Operand | 16x16 tiles span <= 3 / <= 10 | tokens span <= 3 / <= 10 | widest tensor span | rel RMS: per tensor / per token / tile 1 plane / tile 2 planes | nonzero -> 0 (tensor / token / tile) |
|---|---|---|---|---|---|
| enc0 values (>= 0, K = 20) | 100 / 100 % | 100 / 100 % | 2 | 1.6e-3 / 7.3e-4 / 0 / 0 | 0 % |
| a into Wqk, Wv | 0 / 100 % | 0 / 100 % | 8 | 8.6e-3 / 7.0e-3 / 5.0e-3 / 0 | 1.0 / 0.8 / 1.1 % |
| u u^T (attention scores) | 0 / 100 % | 1 / 100 % | 7 | 8.6e-3 / 5.4e-3 / 5.0e-3 / 0 | 1.1 / 0.6 / 1.2 % |
| o into Wp | 2 / 100 % | 0 / 100 % | 8 | 8.1e-3 / 6.2e-3 / 4.3e-3 / 0 | 1.0 / 0.7 / 1.0 % |
| m into W1 | 0 / 100 % | 0 / 100 % | 7 | 7.6e-3 / 6.2e-3 / 4.6e-3 / 0 | 1.0 / 0.8 / 1.2 % |
| GELU output h into W2 | 0 / 100 % | 0 / 100 % | 9 | 1.9e-2 / 1.5e-2 / 1.2e-2 / 0 | 9.6 / 8.1 / 6.5 % |
| merge input into Wm | 0 / 100 % | 0 / 100 % | 8 | 8.3e-3 / 7.0e-3 / 4.8e-3 / 0 | 1.4 / 1.1 / 1.4 % |
| expand input into We | 0 / 100 % | 0 / 100 % | 9 | 9.0e-3 / 6.5e-3 / 4.8e-3 / 0 | 1.2 / 0.8 / 1.2 % |
| dec1 output into dec0 W | 0 / 100 % | 0 / 100 % | 9 | 7.9e-3 / 5.8e-3 / 6.7e-3 / 0 | 1.4 / 1.0 / 1.8 % |

- MEASURED: apart from the encoder's input values, no activation tile fits one plane; every tile and every whole
  tensor fits two (widest span 9, the bound for two planes is 10).
- MEASURED: the GELU output feeding W2 is the worst operand for one plane: 6-10 % of its nonzero values become 0
  (GELU's small negative tail sits many binades below the tile maximum). W2 also carries the most products (K = 4C).
- INFERRED: Model M's activations are bounded (FP8 saturation at 448, RMSNorm inputs, the soft cap), yet that does
  not help a single plane: the trouble is the spread inside a tile, not the maximum.

### 3.3 End to end, Model M (MEASURED: four probe cases at 128x72; full table in `results/e2e-M.md`)

| Run | display PSNR dB min / mean | display max abs | network-output SNR | worst final verdict (chained) | display equal | isolated stages DIFFERENT (of 19) |
|---|---|---|---|---|---|---|
| (i) baseline vs itself; FP32-accumulation control | inf | 0 | exact | MATCHES | 100 % | 0 |
| (ii) W:S1, exact activations | 35.2 / 45.3 | 0.063 | 17.5 dB | DIFFERENT | 28.8 % | 18 |
| (ii) W:S2 | 34.4 / 45.7 | 0.063 | 17.8 dB | DIFFERENT | 32.6 % | 18 |
| (ii) W:S3 | 34.4 / 45.8 | 0.066 | 17.8 dB | DIFFERENT | 32.9 % | 18 |
| (ii) W:S4 | 35.0 / 45.9 | 0.064 | 18.0 dB | DIFFERENT | 31.9 % | 18 |
| (ii) W:S5 (two planes) | inf | 0 | exact | MATCHES | 100 % | 0 |
| (iii) W:S2 + A per tensor | 34.6 / 45.4 | 0.062 | 17.3 dB | DIFFERENT | 32.2 % | 18 |
| (iii) W:S2 + A per token | 35.3 / 45.3 | 0.071 | 17.1 dB | DIFFERENT | 32.1 % | 18 |
| (iii) W:S3 + A per token | 35.4 / 45.3 | 0.058 | 17.1 dB | DIFFERENT | 32.0 % | 18 |
| (iii) W:S4 + A tile, one plane | 35.5 / 46.0 | 0.068 | 18.0 dB | DIFFERENT | 32.0 % | 18 |
| (iii) W:S5 + A tile, two planes | inf | 0 | exact | MATCHES | 100 % | 0 |
| (iv) adaptive exact planes | inf (by construction) | 0 | exact | MATCHES | 100 % | 0 |

- MEASURED: weight-only int8 at one plane already makes every stage DIFFERENT, isolated as well as chained. After the
  first stage only 67-74 % of enc0 features are equal; from enc2 on, 10-20 % of FP8 values are equal in the chained
  run. Isolated stages keep 26-57 % equal. Adding int8 activations changes little: the weight error dominates.
- MEASURED: the 18 isolated DIFFERENT stages are every stage with a multiply; the 19th is the post stage, which has
  none.
- MEASURED: FP32 accumulation of the exact products (BLAS order, a CPU stand-in for an F16 or FP8 WMMA that keeps
  products exact) gave bit-identical outputs at every stage on this data. That is a proxy for the FP16 route, not a
  measurement of gfx11.
- INFERRED: the display PSNR flatters every scheme. The probes are reset frames, so the display is mostly the current
  colour filtered by the network's kernel coefficients; the worst case (34-35 dB) is the honest number. The
  network-output SNR (17-18 dB, i.e. about 13 % relative error in `dec0.h`) is the better measure of what the network
  computes differently. With history over many frames the error could build up (UNKNOWN; not tested).
- Exact adaptive planes (iv), WMMAs per original WMMA (MEASURED on the same runs; attention scores sampled on 1/64 of
  the windows): per-block scales 3.81, per-row scales over the whole K 3.92. Per site: enc0 1.0-2.0 (half its K tiles
  are zero on reset frames), Wqk 3.8, Wv / W1 / W2 / Wp / scores 4.0, Wm 2.9, We 2.9-3.4, dec0 3.6. No output's
  integer bound reached 2^31 (largest 2^24), so int32 never overflowed with per-row scales.

### 3.4 Scale granularity: what is free (INFERRED from the instruction's arithmetic)

`v_wmma_i32_16x16x16_iu8` computes D[i][j] = sum over k of A[i][k] B[k][j] exactly. A scale that does not depend on k
can be applied after the multiply:

| Scale | Cost |
|---|---|
| per output row of the weight matrix (real or power of two) | free in multiplies: folds into the one int32 -> float conversion per output |
| per token (row of the activation operand) | free in multiplies; one more multiply per output if kept separate from the weight scale |
| per activation tensor | free (one constant) |
| per 16-wide K tile (scale depends on k) | each K tile's int32 partial sum must be converted and scaled before adding: about 2 VALU per output per K tile (~17 cycles per step, from the 8-9 cycles per slot-per-value rate of `RESULTS-VALU.md`); exact only if that float sum is exact |
| two or three digit planes | one extra WMMA per digit pair, plus a shift per extra level (Horner) |

Because the measured spans of whole rows and whole tokens are as narrow as those of 16x16 tiles (3.3: 3.92 against
3.81 WMMAs), the free per-row and per-token scales are the right choice: the per-tile rescale buys almost nothing.

## 4. Cost

Formula (INFERRED; cycles per wave for one 16x16x16 step, from `probe-vk-valu/RESULTS-VALU.md` section 6:
I8 or F16 WMMA 32.1-33.1, int32 -> int8 requantize 27.3, FP32 -> E4M3 quantize 133, E4M3 -> F16 widen 35 per 8 values,
half-wave exchange 14.7): cycles = 32.1 x (WMMAs per step) + output glue.

| Scheme | WMMAs | Output glue per step | Cycles | Verdict | Model M PSNR (display, worst) |
|---|---|---|---|---|---|
| FP16 route (MEASURED budget) | 1 | widen 141 + quantize 133 + exchange 15 | 322 | usable as is | inf on the CPU proxy |
| INT8 one plane, row / tensor scales (S1-S3) | 1 | requantize 27 (+15 exchange) | 59-74 | not viable without calibration; DIFFERENT | 34.4-35.4 dB |
| INT8 one plane, block scales (S4, tile activations) | 1 | requantize 27 + ~17 per K tile (+15) | 76-91 | not viable; DIFFERENT | 35.0-35.5 dB |
| INT8 two planes on both operands (S5 + two-plane activations), row scales | 4 (3.92 measured) | E4M3 quantize 133 (the model rounds to E4M3 between products) + digit split ~35 + Horner shifts ~18 + exchange 15 | 128 + 201 = ~330 | usable as is (exact) | inf |
| INT8 adaptive exact (S6 / rr_step_i) | 3.81-3.92 | as above, plus per-tile scale handling for block scales | ~310-330 | usable as is (exact) | inf |

Why the exact routes cost the same as FP16 (INFERRED): Model M rounds every product's result to E4M3, so an exact INT8
route still needs the 133-cycle software E4M3 quantizer that dominates the FP16 route; it trades the 141 cycles of
widening for about three extra WMMAs (96 cycles) plus digit handling. The only cheap INT8 route (59 cycles) is the
one that changes the model's arithmetic, and that one is DIFFERENT on every stage.

## 5. Verdict per scheme (Model M)

| Scheme | Verdict | Basis |
|---|---|---|
| S1 per tensor | not viable | MEASURED: DIFFERENT, SNR 17.5 dB |
| S2 per output channel | needs calibration frames (and a new contract); as tested, not viable | MEASURED: DIFFERENT, SNR 17.8 dB |
| S3 per channel, MSE clipping | same as S2 | MEASURED: identical to S2 |
| S4 per 16x16 block, power of two, one plane | not viable | MEASURED: DIFFERENT, SNR 18.0 dB; 4 % of blocks exact |
| S5 two planes | usable as is | MEASURED: MATCHES, exact (every weight and activation span <= 9) |
| S6 exact digits (adaptive) | usable as is | MEASURED: 3.8-3.9 WMMAs per product on this data; INFERRED cost ~ FP16 route |
| int8 activations, per tensor / token / tile one plane | not viable (on top of any one-plane weights) | MEASURED: DIFFERENT |

"Needs calibration frames": the prior art (`findings/7-int8-prior-art.md` 3.1-3.3) recovers single-plane INT8 with
AdaRound-style rounding or per-layer reconstruction from a few hundred unlabelled frames. Nothing here tested that.
Whether it could bring Model M from DIFFERENT to CLOSE is UNKNOWN; given 10-20 % equal FP8 values per chained stage,
CLOSE (99 % equal) looks out of reach for any single-plane scheme (INFERRED).

## 6. Other models

- RR preset F (MEASURED, weights only, 3.1): the same picture as M. 14 % of blocks fit one plane, all fit two;
  S6 needs 1.86 digits per block on the weight side. Its activations were not measured (the RR CPU reference was not
  run here). Step I already uses up to three activation planes (`rr_step_i.hlsli:8-10`); expect about 2 x 2 WMMAs per
  product as in M (INFERRED from M's activation spans, not measured for RR).
- K, L, FG: no real model data on this machine (MEASURED: only `model/model.wgcm` (M) and `model/F/model.wgcm`).
  `int8_study.py` runs on any of them unchanged: it takes the matrices from the map (op `fp8`/`e4m3`, else FP16
  matrices). For FP16 weights the span rule uses 11 significant bits, so even span 0 needs two planes (INFERRED from
  the format): FP16-weight models (K, FG) are poor INT8 candidates unless re-quantized.

What makes a model cheap for INT8 (INFERRED from the results):

1. Single-plane exactness needs every 16x16 weight block and every 16-token activation tile (or, with the free scales,
   every weight row and every token over the whole K) to span at most 3 binades beyond the first, i.e. 4 binades in
   all. Model M and RR have 4 % and 14 % of weight blocks there and 0 % of activation tiles.
2. Bounded activations are not enough; the spread inside a tile matters. Operands after a GELU-like function
   (a long small negative tail) are the worst.
3. A model trained or quantized for INT8 (as AMD did for FSR 4.1) has values on a uniform grid by construction; a model
   defined in E4M3 does not.
4. A model whose contract is "close in image quality" rather than "same values" could accept one plane after
   calibration. WrongGPU's contract (equivalence protocol, FP8 steps) cannot.

What must be true for single-plane INT8 to match FP16 quality (INFERRED): the model's own definition must use an
integer grid between products (int8 weights and activations with scales), not E4M3. For an E4M3-defined model
the only INT8 forms that match are the exact two-plane ones, at FP16-route cost.

## 7. Limits

- Synthetic inputs: the probes are generated scenes (textures, edges, a checkerboard, a 1024x brighter pan), single
  reset frames, 128x72. Game frames could have wider activation spans (more planes) or narrower ones. UNKNOWN.
  To rerun on captured frames: replace `probe_input()` in `m_e2e.py` with a reader of the `.npz` frames that
  `clean/equiv/model_m.py` reads (`color`, `depth`, `motion`, `mv_scale`, `jitter`, ...); the rest is unchanged.
  Captured frames are the user's data: run on the user's machine and keep only the aggregate output.
- No temporal accumulation: history, latent and motion history were zero (reset). Error build-up over frames is UNKNOWN.
- Activation scales were taken from the same input ("optimistic"); a static calibration would do worse (INFERRED).
- Attention A V stayed FP16; enc0's FP16 running sums and dec0's FP16 partial sums kept the reference's order.
- Cycle numbers are throughput budgets from the probe (MEASURED there), combined by formula here (INFERRED). The digit
  split (~35) and Horner shifts (~18) are estimates by analogy with the widening cost, not measured.
- The FP16-route quality claim rests on a CPU proxy (FP32 accumulation) and on the open question T1 in
  `RDNA3-ANALYSIS.md` 4.3; gfx11's real accumulation was not measured here.
- Only Model M has end-to-end numbers; RR has weight numbers only.

## Files

Not published with this copy: the `results/` folder (per-tensor tables of a model converted from the tester's own
DLSS files). `run.sh` recreates it from any converted model.

`run.sh`, `qlib.py`, `int8_study.py`, `m_e2e.py`, `report.py`; `results/weights-M.{md,json}`,
`results/weights-F.{md,json}`, `results/e2e-M.{md,json}` (aggregate numbers only). Runtime: weights under 1 s per
model; end to end about 1 minute for four cases on 8 cores (peak memory not measured; the 8 GB cap was never hit).
