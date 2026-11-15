# Lane 6: how AMD built FSR 4's INT8 model, and what that teaches WrongGPU

Researched 2026-11-14. Labels: MEASURED-IN-REPO, READ, INFERRED, UNKNOWN (as in COMMON.md). Two more for outside
work: MEASURED-ELSEWHERE (a named outlet measured it) and IMPRESSION (a named outlet's or AMD's judgement, not a
number). Repo paths are relative to the repo root of `src` (upstream main, 994f304).

Clean-room note. Only AMD's public SDK repository, GPUOpen pages, AMD product pages and release notes, Intel's public
SDK and slides, arXiv papers and press articles were read. The leaked 2025 FSR 4 source and any mirror of it were not
opened. Skipped on purpose because they may contain or derive from leaked code, and not read: the community issue
"FSR 4 (INT8): three bit-exact speedups found on RDNA3 under Linux/RADV"
(https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/issues/210), the repository `lhl/fsr4-rdna3-optimization`,
and the Reddit post behind the gamegpu article in section 3.4. Only the gamegpu press article's own text is used.
The shipped signed DLLs were not disassembled.

## Summary

1. AMD ships three FSR 4 models: "Full FP8" (RDNA 4), "Full INT8" (RX 7000 since FSR 4.1.1 / June 2026; RX 6000,
   Strix Halo and Gorgon Halo since FSR 4.1.2 / 28 Oct 2026) and "Lightweight INT8" (Strix Point, Gorgon Point,
   Z2 Extreme, Z2 A; "less than half the size of the original INT8 model"). READ, section 1.
2. AMD's own table: RX 7900 XTX INT8 4K 3070 us, RX 6900 XT INT8 4K 3190 us, RX 9070 XT FP8 4K 1316 us. Memory:
   one working-set table only (318 MB at 4K), measured on an RX 9070 XT, "do not change significantly between GPUs".
   READ.
3. AMD calls the INT8 model "slightly different" from the FP8 one, "fully adapted" to the new format and checked for
   image errors, with quality "identical" to RDNA 4. It has published nothing on how: no layer list, no quantization
   method, no scales. FSR 4 ships only as a signed DLL; the public SDK has no FSR 4 shader source. READ.
4. Per CU per clock, AMD lists the same 8-bit peak for RDNA 2 (dot4, no matrix unit) and RDNA 3 (WMMA): 512. RDNA 3's
   F16 WMMA peak is also 512. So on RDNA 3 the INT8 format buys no matrix throughput over F16. READ (GPUOpen table).
5. Normalised to CU-cycles per 1080p-equivalent frame, AMD's own timings give RDNA 4 FP8 about 58-67 M, RDNA 2 INT8
   141-152 M, RDNA 3 INT8 168-184 M. RDNA 3 is no faster per CU-cycle than RDNA 2; RDNA 4 FP8 needs about 2.7x fewer.
   INFERRED from READ timings and specs, assuming the published boost clocks (section 3.2).
6. Reviewers who compared INT8 on RX 7000 with FP8 on RX 9070 XT in the same games (TechSpot/Hardware Unboxed, PCGH,
   Club386) found them nearly identical; the FP8 model is slightly better in fine detail at low render resolutions.
   The 2025 leaked INT8 build was visibly worse; the shipped one is better. MEASURED-ELSEWHERE (side-by-side captures).
7. Generic recipe (literature): per-channel INT8 weights, per-tensor activations, calibration on unlabeled inputs is
   usually enough for 8 bits; quantization-aware training needs the training pipeline and data. An FP8-trained net
   converts to INT8 well unless it has outliers (Qualcomm FP8-vs-INT8 paper). READ.
8. For WrongGPU: an INT8 model can be calibrated from captured game inputs without training data, but it is a new
   model, cannot meet "byte-identical" or "MATCHES", and gains no matrix speed on RDNA 3 over F16 WMMA carrying exact
   E4M3 values. Per-channel INT8 holds E4M3 weights exactly only within 4 binades of the channel maximum. INFERRED.
9. XeSS precedent: Intel says its DP4a path uses a "smaller model, less parameters" than the XMX one, trained
   separately. AMD instead kept one network size for discrete RDNA 2/3 and built a separate lightweight one for iGPUs.
   READ.

## 1. Facts: versions, GPUs, dates, timings, memory, quality statements

### 1.1 Which versions ship an INT8 model, for which GPUs, since when

| FSR version / SDK | Date | INT8 targets | Source | Label |
|---|---|---|---|---|
| 4.0.2 (SDK 2.0.0) | 2025-08-20 | none official; the accidental source release that day contained an INT8 variant, later built by the community | press: https://www.techspot.com/news/109164-amd-accidentally-leaks-fsr-4-source-code-teasing.html , https://www.club386.com/amd-fsr-4-rdna-3-tested/ ("references to 'i8' folders") | READ (press) |
| 4.0.3 (SDK 2.1.x), 4.1.0 (SDK 2.2.0) | 2025-12-10, 2026-03-18 | none; doc says "FSR4 requires an AMD 9000 series GPU or later" | SDK git history, `Kits/FidelityFX/docs/techniques/super-resolution-ml.md` at commits 0836aa7, e236f23 | READ |
| 4.1.1 (SDK 2.3.0), driver 26.6.2 | SDK 2026-06-24; driver 2026-06-22 | RX 7000 discrete ("Added support for AMD FSR Upscaling 4.1 on AMD Radeon RX 7000 Series (AMD RDNA 3 architecture) discrete GPUs") | `docs/whats-new/version_2_3_0.md` line 61; https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-6-2.html ("AMD FSR Upscaling 4.1 support for AMD Radeon RX 7000 Series Graphics cards"); date: https://www.tomshardware.com/pc-components/gpu-drivers/amd-brings-official-fsr-4-1-support-to-rx-7000-series-gpus-int8-model-now-available-in-300-games-rdna-3-apus-also-getting-fsr-4-1-soon | READ |
| 4.1.2 (SDK 2.4.0) | 2026-10-28 | adds RX 6000 discrete and Ryzen AI Max 300/400 ("Strix Halo", "Gorgon Halo") on the full INT8 model; a "highly optimized" lightweight INT8 model for Ryzen AI 9 300/400 ("Strix Point", "Gorgon Point"), Ryzen AI Z2 Extreme, Z2 Extreme and Z2 A ("AMD RDNA 2 architecture iGPU") | `docs/whats-new/index.md` lines 18-30 at 7539ae8; https://gpuopen.com/amd-fsr-sdk/ | READ |

All SDK paths: https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK (blobless clone, `main` = 7539ae8,
"AMD FSR SDK 2.4.0", 2026-10-28). Further facts:

- Model selection is automatic: "Model selection is automatic and based on the target hardware"
  (`super-resolution-ml.md` line 87). READ.
- The lightweight model "is less than half the size of the original INT8 model and purpose-built for this class of
  hardware" (`whats-new/index.md` line 29). Absolute sizes are not published. READ.
- Not listed in SDK 2.4.0: RDNA 3 APUs (Phoenix / Hawk Point, e.g. Radeon 780M) and RDNA 1. In June AMD said it was
  "working on lightweight machine learning models" for RDNA 3 APUs (Tom's link above); heise reports RDNA 1 is
  excluded as it lacks "a suitable data type" (https://heise.de/-11295174). READ (absence in SDK; press).
- The doc's Limitations line still says "FSR4 requires an AMD 7000 series discrete GPU or AMD 9000 series GPU or
  later" (`super-resolution-ml.md` line 347) while its tables list RX 6000 and APUs. READ; a stale line.
- AMD says FSR 4 is "Trained on high-quality game data using AMD Instinct GPUs" (https://gpuopen.com/amd-fsr-sdk/).
  READ.
- Press reports Sony's updated PSSR on PS5 Pro "corresponds to an FSR 4.1 implementation with INT8"
  (https://heise.de/-11295174). READ (press); not confirmed by AMD in anything read here.

### 1.2 AMD's timing and memory tables, quoted in full

From `Kits/FidelityFX/docs/techniques/super-resolution-ml.md` lines 68-110 at 7539ae8 (FSR 4.1.2), verbatim:

> Below you can find reference performance numbers for running FSR4 in performance mode for recommended target
> resolution. Upscaling from a lower resolution is slightly faster than upscaling from a higher one, but performance
> is mostly bound by the target resolution AMD FSR™ Upscaling upscales to.
> Performance numbers are measured with RCAS sharpening disabled and peak GPU clocks.

| AMD Radeon GPU         | ML Model  | Target Resolution | Microseconds (us)|
| ---------------------- | --------- | ----------------- | ---------------- |
| AMD Radeon RX 9070 XT  | Full FP8  | 3840x2160         |  1316us          |
| AMD Radeon RX 9070 XT  | Full FP8  | 1920x1080         |   352us          |
| AMD Radeon RX 9060 XT  | Full FP8  | 3840x2160         |  3433us          |
| AMD Radeon RX 9060 XT  | Full FP8  | 1920x1080         |   583us          |
| AMD Radeon RX 7900 XTX | Full INT8 | 3840x2160         |  3070us          |
| AMD Radeon RX 7800 XT  | Full INT8 | 2560x1440         |  2050us          |
| AMD Radeon RX 7600     | Full INT8 | 1920x1080         |  1998us          |
| AMD Radeon RX 6900 XT  | Full INT8 | 3840x2160         |  3190us          |
| AMD Radeon RX 6700 XT  | Full INT8 | 2560x1440         |  2460us          |
| AMD Radeon RX 6600     | Full INT8 | 1920x1080         |  2183us          |
| AMD Radeon RX 6500 XT  | Full INT8 | 1920x1080         |  3130us          |

| AMD Ryzen APU                         | System Memory | ML Model         | Target Resolution | Microseconds (us)|
| ------------------------------------- | ------------- | ---------------- | ----------------- | ---------------- |
| AMD Ryzen AI MAX+ 395 w/ Radeon 8060S | LPDDR5X-8000  | Full INT8        | 2560x1440         |  3022us          |
| AMD Ryzen AI MAX+ 395 w/ Radeon 8060S | LPDDR5X-8000  | Full INT8        | 1920x1080         |  1668us          |
| AMD Ryzen AI 9 HX 370 w/ Radeon 890M  | LPDDR5X-8000  | Lightweight INT8 | 2560x1440         |  4222us          |
| AMD Ryzen AI 9 HX 370 w/ Radeon 890M  | LPDDR5X-8000  | Lightweight INT8 | 1920x1080         |  2423us          |
| AMD Ryzen AI 9 HX 470 w/ Radeon 890M  | DDR5-5600     | Lightweight INT8 | 2560x1440         |  5311us          |
| AMD Ryzen AI 9 HX 470 w/ Radeon 890M  | DDR5-5600     | Lightweight INT8 | 1920x1080         |  3187us          |
| AMD Ryzen Z2 Extreme                  | LPDDR5X-8000  | Lightweight INT8 | 1920x1080         |  2520us          |
| AMD Ryzen Z2 A                        | LPDDR5-6400   | Lightweight INT8 | 1920x1080         |  5620us          |

| Resolution | Working set (MB) |
| -----------|------------------|
| 7680x4320  | 1274MB           |
| 3840x2160  |  318MB           |
| 1920x1080  |   81MB           |

> Figures are approximations, rounded to nearest MB using an RX 9070XT GPU, and are subject to change. These do not
> change significantly between GPUs.

History of the table (READ, git): SDK 2.2.0 had only the two RX 9070 XT rows. SDK 2.3.0 added RX 7900 XTX 3070,
RX 7800 XT 2050, RX 7600 1998 with no model column. SDK 2.4.0 added the "ML Model" column, the RX 9060 XT, RX 6000 and
APU rows. The RX 7000 numbers are unchanged since 2.3.0, so they were probably measured on FSR 4.1.1 and the RX 6000
numbers on 4.1.2 (INFERRED; matters for section 3.3). AMD publishes no per-model memory figure and no model sizes.

### 1.3 What AMD says about INT8 quality against FP8

| Statement | Where | Label |
|---|---|---|
| Computex 2026 interview (Zdravkovic, Makedon), reported by PCGH: AMD "fully adapts the model to the new data format and checks it for image errors before shipping"; AMD "explicitly speaks of a slightly different AI model whose end result is qualitatively identical" (translated: "einem leicht abweichenden KI-Modell, dessen Endergebnis aber qualitativ deckungsgleich sei") | https://www.pcgameshardware.de/FidelityFX-Super-Resolution-Software-277617/News/AMD-FSR-41-RDNA-3-Bildqualitaet-INT8-1545397 (2026-06-16) | READ (press quoting AMD); IMPRESSION by AMD |
| Same interview via OC3D: "FSR 4.1 will use a different model on RDNA 3 GPUs, but will deliver the same levels of image quality"; AMD "ensured that the final visual output for FSR 4.1's new INT8 version matches its output on FP8 RDNA 4 GPUs" | https://overclock3d.net/news/gpu-displays/amd-is-targeting-fsr-4-1-quality-parity-on-rdna-3-gpus/ | READ (press) |
| Club386: "Reportedly, AMD had to completely modify the FSR 4.1 model for this new data type to maintain identical image quality"; RDNA 2 "required a great deal of optimisation" to "reduce the number of shader cycles"; training "starts with general training on Instinct MI accelerators, before moving to workstation-grade Radeon Pro GPUs for further refinements", then testing on "hundreds of thousands of PC configurations" | https://www.club386.com/?p=147588 (2026-06-16) | READ (press, "reportedly") |
| Jack Huynh: FSR 4.1 for RDNA 3 verified "on hundreds of PC configurations without compromises in quality and performance" (PCGH paraphrase, translated) | PCGH link above | READ (press paraphrase) |
| Lightweight model: "a balance of image reconstruction quality and performance optimized for these devices" | `whats-new/index.md` line 29 | READ |

No AMD number for the INT8 vs FP8 difference (PSNR, error, or anything else) was found. UNKNOWN.

## 2. How the INT8 model differs from the FP8 model

### 2.1 What is public

- Format and delivery. FSR 4 "requires integration using the AMD FSR API ... and use of the signed binary
  distribution" (`super-resolution-ml.md` line 51). The tree has `Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll`
  and `..._vk.dll` and no FSR 4 shader, weight or model file (git tree listing of 7539ae8; only FSR 2/3 HLSL is
  present under `Kits/FidelityFX/upscalers/fsr3/`). READ. So no public shader source exists to show how one INT8
  layer is computed; question 2's "read the shader" step cannot be done within the clean-room rule.
- Same network or retrained? AMD's words: "slightly different model", "fully adapted", "completely modify"
  (section 1.3). Nothing says whether this was quantization-aware training, post-training quantization, distillation
  or a new architecture. UNKNOWN.
- Architecture of FSR 4 itself: at launch AMD said only that it "uses a proprietary model that takes the best of CNN
  and transformer" and "utilizes FP8 capabilities of RDNA 4's 2nd gen AI accelerators"
  (https://www.notebookcheck.net/AMD-talks-FSR-4-Hypr-RX-and-new-Adrenalin-software-FSR-4-uses-proprietary-model-and-is-limited-to-RDNA-4-cards-for-now.969986.0.html).
  Layer structure, channel counts, activation functions, clamp points, scale granularity and scale storage: not
  published for either model. UNKNOWN.
- Press description of the shipped INT8 shaders (not AMD): a Linux enthusiast who profiled the vkd3d-proton SPIR-V of
  FSR 4 INT8 on an RX 7900 XT reports "int8 rounding instructions between layers" that could be "replaced with cheaper
  and equivalent hardware commands", and "internal loops whose boundaries are always zero in practice"
  (https://en.gamegpu.com/news/zhelezo/modifikatsiya-shejderov-fsr-4-int8-uskorila-videokarty-amd-radeon-rx-7000-pod-linux-na-31 ,
  2026-10-27). READ (press, second-hand). INFERRED from it: activations are re-quantized to 8-bit integers between
  layers, as in the standard integer scheme of section 5.1. Whether scales are per channel, and whether they are
  applied as float multiplies or integer multiply-and-shift, is UNKNOWN.

### 2.2 What can be inferred, and how strongly

- Probably not a pure cast of the FP8 weights. AMD's own wording ("adapt", "slightly different") and the fact that
  reviewers saw the 2025 leaked INT8 build lose detail at low resolutions while the 2026 build does not (section 4)
  point to further work on the model or its quantization between August 2025 and June 2026. INFERRED, medium
  confidence: the improvement could also come from non-network changes (PCGH sees MIP-bias differences, section 4).
- The lightweight model is a different network. "Less than half the size" of the full INT8 model means fewer
  parameters, so it had to be trained (or distilled) anew, which needs AMD's training pipeline. INFERRED, high
  confidence; AMD did not say how it was made.
- The full INT8 model probably keeps the full model's size. AMD calls it "Full INT8" next to "Full FP8", TechSpot's
  2026 review judged it "the same AI upscaling model as the FP8 version, or one that is virtually identical"
  (IMPRESSION), and the RX 6000 path is labelled "full INT8 model" too (GPUOpen). INFERRED.
- One INT8 model for both RDNA 2 and RDNA 3. Same label, same SDK, and RDNA 2 has no matrix unit, so the format
  choice served both generations; on RDNA 3 alone INT8 has no matrix-rate advantage over F16 (section 3.1). INFERRED.

### 2.3 What one standard INT8 layer looks like (generic, not AMD's)

Since AMD's layer is not public, this is the textbook integer-only layer for reference (Jacob et al. 2018,
https://arxiv.org/abs/1712.05877 ; Wu et al. 2020, https://arxiv.org/abs/2004.09602). READ (papers).

1. Weights W stored as int8 with a scale per output channel s_w[c] (symmetric, zero point 0).
2. Input activations x stored as int8 (or uint8 after ReLU) with one scale s_x per tensor.
3. Accumulate sum_k W[c,k] * x[k] in int32 (dot4 or IU8 matrix instruction; exact below 2^31).
4. Add the bias, pre-quantized to int32 with scale s_w[c] * s_x.
5. Requantize: y = round(acc * M[c]), M[c] = s_w[c] * s_x / s_y, done either as a float multiply or as an int32
   fixed-point multiplier M0 and a right shift n (M = M0 * 2^-n), then saturate to [-128, 127]. The activation
   function is often fused here (ReLU becomes a clamp at 0; bounded activations become a clamp at their range).
6. Nonlinear functions with no integer form (softmax, GELU, layer norm in transformer parts) usually run in FP16 or
   FP32 between dequantize and requantize steps.

## 3. Which GPU instructions the INT8 path uses

### 3.1 What each generation offers for 8-bit products

| Generation | 8-bit integer instructions | Peak 8-bit per CU per clock (AMD) | F16 per CU per clock (AMD) | Source |
|---|---|---|---|---|
| RDNA 2 (gfx103x, RX 6000) | packed 4-way 8-bit dot products (LLVM `v_dot4_i32_i8` / `v_dot4_u32_u8`, HLSL SM 6.4 `dot4add_i8packed`; names not re-checked in this lane); no WMMA | 512 | 256 | GPUOpen RDNA 4 article table (RX 6950 XT column): https://gpuopen.com/learn/using_matrix_core_amd_rdna4/ ; LLVM gating in findings/4-isa-driver.md section 5 |
| RDNA 3 (gfx11, RX 7000) | `v_dot4_i32_iu8` and `v_wmma_i32_16x16x16_iu8` (16x16x16 per 32 cycles in wave32, same as F16 WMMA) | 512 | 512 (WMMA) | same table (RX 7900 XTX column); findings/4-isa-driver.md lines 144-145 |
| RDNA 4 (gfx12, RX 9000) | as RDNA 3 plus faster IU8 and FP8 WMMA (8 cycles), SWMMAC | 2048 (I8; FP8 same rate) | 1024 | same table (RX 9070 XT column) |

AMD's product pages agree: RX 7900 XTX "Peak 8-bit Precision (INT8 Matrix) Performance 123 TOPs" = 96 CU x 512 x
2.5 GHz; RX 9070 XT "389 TOPs" = 64 x 2048 x 2.97 GHz
(https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html ,
https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html). READ.

Consequence. On RDNA 3 the IU8 matrix instruction has the same peak per CU-clock as RDNA 2's dot4 and as RDNA 3's own
F16 WMMA. What IU8 WMMA saves on RDNA 3 is registers and bandwidth (A/B fragment 4 VGPRs instead of 8 for F16,
findings/4-isa-driver.md line 129), not arithmetic rate. READ (rates) + INFERRED (consequence).

Which of these AMD's shipped INT8 shaders use on RX 7000, and whether they go through the WaveMatrix driver
intrinsics there at all: not published. UNKNOWN. Press evidence only: the Linux enthusiast above measured "a maximum
of 52 trillion operations per second" for WMMA and "55 trillion" for dot4 on an RX 7900 XT, stating that on RDNA 3
"matrix units share execution resources with regular shader processors" (gamegpu link, 2.1). READ (press, one
person, op-count convention not stated). Consistent with the equal-peak table.

### 3.2 AMD's timings normalised per CU-cycle

Inputs: AMD's timing table (1.2); CUs and boost clocks from AMD's product pages (RX 6900 XT 80 CU / 2250 MHz, RX 6700
XT 40 / 2581, RX 6600 28 / 2491, RX 6500 XT 16 / 2815, RX 7900 XTX 96 / 2500, RX 7800 XT 60 / 2430, RX 7600 32 / 2655,
RX 9070 XT 64 / 2970, RX 9060 XT 32 / 3130; e.g.
https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6900-xt.html , same path pattern
for the others). Assumption: AMD's "peak GPU clocks" equal the rated boost clock. CU-cycles = time x CUs x clock;
"per 1080p-eq" divides by target pixels / (1920x1080). INFERRED (arithmetic).

| GPU | Model | Target | us | 8-bit peak TOPS | M CU-cycles per frame | M CU-cycles per 1080p-eq |
|---|---|---|---|---|---|---|
| RX 9070 XT | FP8 | 4K | 1316 | 389 | 250 | 63 |
| RX 9070 XT | FP8 | 1080p | 352 | 389 | 67 | 67 |
| RX 9060 XT | FP8 | 4K | 3433 | 205 | 344 | 86 |
| RX 9060 XT | FP8 | 1080p | 583 | 205 | 58 | 58 |
| RX 7900 XTX | INT8 | 4K | 3070 | 123 | 737 | 184 |
| RX 7800 XT | INT8 | 1440p | 2050 | 75 | 299 | 168 |
| RX 7600 | INT8 | 1080p | 1998 | 44 | 170 | 170 |
| RX 6900 XT | INT8 | 4K | 3190 | 92 | 574 | 144 |
| RX 6700 XT | INT8 | 1440p | 2460 | 53 | 254 | 143 |
| RX 6600 | INT8 | 1080p | 2183 | 36 | 152 | 152 |
| RX 6500 XT | INT8 | 1080p | 3130 | 23 | 141 | 141 |

Readings (INFERRED):
- Each family is internally consistent (RDNA 3: 168-184; RDNA 2: 141-152; RDNA 4: 58-67), which supports the
  clock assumption and AMD's "mostly bound by the target resolution".
- RDNA 3 INT8 costs about 2.7x the CU-cycles of RDNA 4 FP8, although RDNA 4's 8-bit peak per CU-clock is 4x. So
  RDNA 4 FP8 does not run at its matrix peak either; part of the frame is not matrix work.
- RDNA 3 needs about 1.2x the CU-cycles of RDNA 2 for the "same" full INT8 model.
- The RX 9060 XT at 4K costs 1.4x its own 1080p per pixel; the 32 MB-cache card falls off at 4K. The RX 7600 (also
  32 MB) is only listed at 1080p.

### 3.3 The RX 7900 XTX (3070 us) next to the RX 6900 XT (3190 us)

The sources do not settle it. What they do settle: per CU-clock the 8-bit peak is equal (3.1), so the only expected
RDNA 3 advantage is CUs x clock, 1.33x for this pair (1.41x for 7800 XT / 6700 XT, 1.22x for 7600 / 6600). The
observed time ratios are 1.04x, 1.20x and 1.09x. Hypotheses for the remaining shortfall, none confirmed:

| # | Hypothesis | For | Against / test |
|---|---|---|---|
| H1 | The RX 6000 rows come from FSR 4.1.2, after AMD's RDNA 2 work to "reduce the number of shader cycles"; the RX 7000 rows are 4.1.1 numbers not re-measured | rows unchanged since SDK 2.3.0 (1.2); AMD said RDNA 2 needed extra optimisation (1.3) | whether 4.1.2 changed RX 7000 time: time FSR 4.1.1 vs 4.1.2 DLL on one RX 7000 card |
| H2 | RDNA 3's WMMA path pays for its layout: duplicated A/B lanes, interleaved accumulator rows, lane exchanges between chained products, no co-issue with VALU | known gfx11 layout costs (findings/4-isa-driver.md lines 136-150) | RGP capture: instruction mix of FSR 4.1 on an RX 7000 |
| H3 | Both generations run the same dot4 shader; RDNA 3 then gains only CUs x clock, minus H4/H5 | equal dot4 rate; one model for both | RGP/RGA: presence of `v_wmma_i32_16x16x16_iu8` vs `v_dot4_i32_iu8` |
| H4 | Real clocks differ from rated boost (RDNA 3 big chips often sit below rated boost under load; AMD only says "peak GPU clocks") | RDNA 3 gap is similar on all three pairs | log clocks during an FSR 4.1 capture |
| H5 | Memory/cache-bound passes: the RX 6900 XT has 128 MB Infinity Cache, the RX 7900 XTX 96 MB | 9060 XT 4K cliff shows cache sensitivity | weak: the 7600 / 6600 pair has equal 32 MB and still shows a gap |
| H6 | Non-matrix work (requantization, packing, input/output passes) dominates and runs at similar per-CU rates | RDNA 4 also far below its matrix peak (3.2); the Linux report found one output pass taking 1.55 of 3.61 ms on RADV | per-pass timing in RGP |

Cross-check (INFERRED): RDNA 3 at 184 M CU-cycles per 1080p-eq predicts 737 M / (84 CU x 2.4 GHz) = 3.66 ms for an RX
7900 XT at 4K. The Linux report measured 3.61 ms baseline on an RX 7900 XT for 1440p -> 4K Quality under
vkd3d-proton/RADV (gamegpu link). Different driver and mode, so only an order-of-size agreement.

## 4. What independent reviewers measured

### 4.1 Image quality, INT8 (RX 7000) against FP8 (RX 9070 XT), shipped FSR 4.1

| Outlet, date | Setup | Conclusion (quoted) | Label |
|---|---|---|---|
| TechSpot / Hardware Unboxed, 2026-07-02, https://techspot.com/article/3142-amd-fsr-4-upscaling-rdna-3 | RX 7900 XTX vs RX 9070 XT; 21 examples, 9 games; 4K output from 1080p (Performance) | "virtually identical"; "extremely difficult to distinguish"; "At worst ... the INT8 version may be slightly worse in a handful of edge cases at low render resolutions"; differences "smaller and more difficult to notice than those between the two DLSS 4.5 models"; "AMD has not downgraded the upscaling model" | MEASURED-ELSEWHERE (captures), conclusion is IMPRESSION |
| PCGH, 2026-06-29, https://www.pcgameshardware.de/FidelityFX-Super-Resolution-Software-277617/Specials/FSR-4-RDNA-3-Release-Test-Benchmarks-1546566/2/ | RX 7800 XT vs RX 9070 XT; six games (Death Stranding 2, Forza Horizon 6, Gothic Remake, ...) | "very difficult to spot differences between FP8 and the stripped-down Int8 model"; at WQHD "only extremely minor differences"; "In Full HD and when using sharper upscaling levels, it's a little easier to see differences"; "slightly different settings for the texture MIP bias"; 4.1.1 INT8 is "yet another improvement over FSR 4.0.2c Int8" (the leaked build) | MEASURED-ELSEWHERE / IMPRESSION |
| Club386, 2026-07-03, https://www.club386.com/amd-fsr-4-rdna-3-tested/ | RX 7800 XT vs RX 9070 XT; four games | "practically identical"; minor differences "in fine line detail and foliage", slightly more visible jaggies, billboard text "read more clearly on the RX 9070 XT" | MEASURED-ELSEWHERE / IMPRESSION |
| ComputerBase (via igor'sLAB, 2026-07-08, https://www.igorslab.de/en/fsr-4-1-rdna-3-better-image-quality-rx-7900-xtx-up-to-14-percent-performance/) | RX 7900 XTX / 7800 XT / 7600, RX 9070 XT reference, nine games | final INT8 shows "clear progress compared with the build leaked last year, which was still struggling with visible rendering errors" | READ second-hand; ComputerBase article not fetched |

Earlier, on the leaked 2025 build (FSR 4.0.2 INT8, not what ships): TechSpot found "The FP8 model delivers
progressively better image quality than the INT8 model as render and output resolutions decrease"; at 1440p
Performance "the original FP8 image is less blurry, more detailed, and sharper overall"; stability comparable;
their guess "The AI model size has likely been reduced" (https://www.techspot.com/article/3040-amd-fsr4-upscaling-older-radeon-gpu/ ,
2025-10-07). MEASURED-ELSEWHERE / IMPRESSION.

Summary: no outlet published a numeric image metric. All agree the shipped INT8 is near-identical at 4K and WQHD
Quality and slightly behind at low render resolutions. "Same model" (TechSpot) is an impression; AMD says "slightly
different".

### 4.2 Performance cost per GPU (FSR 4.1 INT8 vs FSR 3.1 on the same card unless stated)

| Outlet | GPU, game, setting | Result | Label |
|---|---|---|---|
| ComputerBase via igor'sLAB | RX 7900 XTX, 9-game average, 4K | gain over native: Quality +40 % (FSR 3.1 +57 %), Performance +73 % (FSR 3.1 +103 %); "around 11 percent" loss in Quality, "up to 14 percent" in Performance vs FSR 3.1 | MEASURED-ELSEWHERE (second-hand) |
| ComputerBase via Tom's forum summary | RX 7800 XT | 3440x1440: 59.4 -> 54.8 FPS (~8 %); 4K: 43.4 -> 39.2 FPS (~10 %); per game 2 % to 18 % (Call of Duty: Black Ops 7) | MEASURED-ELSEWHERE (third-hand, a forum post quoting ComputerBase; weak) |
| TechSpot 2026 | Cyberpunk 2077 4K RT Ultra | RX 7900 XTX: Quality +82 % (FSR 3.1 +92 %), Performance +168 % (+200 %); RX 9070 XT FP8: +86 % / +176 % | MEASURED-ELSEWHERE |
| TechSpot 2026 | Mafia: The Old Country 4K | RX 7900 XTX Quality +52 % (FSR 3.1 +70 %), Performance +83 % (+114 %); RX 9070 XT FP8 Quality +68 %, Performance +108 % | MEASURED-ELSEWHERE |
| TechSpot 2026 | Horizon Zero Dawn Remastered | RX 7900 XTX Quality +9 % vs FSR 3.1 +29 % | MEASURED-ELSEWHERE |
| Club386 | RX 7800 XT, Assassin's Creed Shadows 4K | FSR 4.1 +33 % vs FSR 3 +75 % | MEASURED-ELSEWHERE |
| PCGH | RX 7900 XTX / 7800 XT / 7600, three games | FSR 4.1 costs more than FSR 3.1 in "the vast majority of cases", most in Ultra Performance; VRAM allocation "slightly lower" than FSR 3.1 (e.g. Crimson Desert 4K Quality 9.0 GiB) | MEASURED-ELSEWHERE |
| TechSpot 2025 (leaked build) | RX 7900 XTX | "about two tiers slower" than FSR 3 (FSR 4 Performance ~ FSR 3 Quality uplift); RX 6750 XT up to three tiers | MEASURED-ELSEWHERE (unofficial build) |

## 5. What generalises

### 5.1 The standard recipe for an INT8 variant

READ unless marked; sources: Nagel et al., "A White Paper on Neural Network Quantization"
(https://arxiv.org/abs/2106.08295); Wu et al. (NVIDIA) (https://arxiv.org/abs/2004.09602); Jacob et al.
(https://arxiv.org/abs/1712.05877); van Baalen et al. (Qualcomm), "FP8 versus INT8 for efficient deep learning
inference" (https://arxiv.org/abs/2303.17951); Nagel et al., "Data-Free Quantization"
(https://arxiv.org/abs/1906.04721); Li et al., "PAMS: Quantized Super-Resolution" (https://arxiv.org/abs/2011.04212).

| Step | What | Data needed |
|---|---|---|
| 1. Choose granularity | weights symmetric per output channel; activations per tensor (asymmetric after ReLU) | none |
| 2. Weight PTQ | range setting (min-max or MSE) per channel; optional cross-layer equalization and bias correction (data-free) or AdaRound (small unlabeled set) | none or a few hundred unlabeled inputs |
| 3. Activation calibration | run representative inputs, record ranges per tensor (max, percentile, entropy or MSE) | unlabeled representative inputs |
| 4. Evaluate | compare to the float model on held-out inputs | unlabeled inputs; the float model as reference |
| 5. If not good enough: QAT | fine-tune with fake quantization in the forward pass | the training pipeline, a loss, and training data (labels, or a teacher model's outputs) |

- Typical loss. Nagel: "In most cases, PTQ is sufficient for achieving 8-bit quantization with close to
  floating-point accuracy. QAT requires fine-tuning and access to labeled training data". Wu: an 8-bit workflow
  "able to maintain accuracy within 1% of the floating-point baseline on all networks studied".
- Image restoration is harder. PAMS: super-resolution models "without batch normalization have a large dynamic
  quantization range", a cause of "significant performance drop"; they fix it with a trainable clipping bound and
  a knowledge-transfer loss from the float model, i.e. training. Upscalers like FSR 4 are of this kind (INFERRED).
- FP8-trained to INT8. van Baalen: "INT8 can exactly represent roughly 90% of the range covered by the FP8-E4 format
  without any quantization error" (with a free scale); ResNet18 and MobileNetV2 trained in FP8-E4 lose nothing when
  converted naively with MSE range setting (69.84 / 71.59 vs FP8 QAT 69.79 / 71.59); networks with outliers (BERT)
  lose more (79.38 vs 83.91) and needed INT8 QAT to recover (83.79). "for networks that were easy to quantize from FP32
  to INT8, the conversion is expected to be smooth".
- What needs the original training data: QAT with ground-truth targets, and any new or smaller architecture (like
  AMD's lightweight model). What does not: PTQ, data-free equalization, calibration, and distillation against the
  original model's own outputs (needs inputs, a differentiable reimplementation and compute, not labels).

### 5.2 E4M3 weights on an INT8 grid: exactness arithmetic

INFERRED, arithmetic shown. An E4M3 value is (8 + m) x 2^(e-3) with m in 0..7 (normal) or m x 2^-9 (subnormal). With
a per-channel power-of-two scale 2^t and |k| <= 127, the largest value in the channel, exponent E, needs
15 x 2^(E-3-t) <= 127, so t >= E-6. With t = E-6, every value in binades E, E-1, E-2, E-3 is exact (k up to 120);
values one binade lower lose one mantissa bit, and so on; anything below 2^(E-7) rounds to 0 or +-1 step. A
non-power-of-two scale does not help in general. So per-channel INT8 holds an E4M3 weight tensor exactly only if
each channel's nonzero weights span at most a factor of 16. E4M3 activations between stages span up to 2^18, so
per-tensor INT8 activations will round small values; that, not the weights, is the likely main loss. This matches
van Baalen's "only the smaller numbers will suffer some loss".

WrongGPU's FP8 weights carry no separate scale: M's tensors are float32 values put on the E4M3 grid by the converter's
`fp8` op (`data/preset-M.v1.map.json:105-117`, 56 tensors per RDNA3-ANALYSIS.md 3.2), encoded at load
(`clean/runtime/model.cpp:87-89`); RR stores E4M3 codes directly (`data/preset-F.v1.map.json`, 208 `"e4m3"` dtype
entries by grep). Activation maps between stages are 1-byte FP8 (`clean/FORMATS.md:12-13`). READ.

### 5.3 What a project with only the trained FP8 weights can and cannot do

Can (INFERRED from 5.1 and the repo):
- Re-quantize weights to per-channel INT8 with no data at all, and measure exactly how many weights change (5.2).
- Calibrate activation scales from representative inputs. WrongGPU does not need training data for this: real game
  inputs can be captured, and the existing stage-dump and equivalence machinery (`clean/EQUIVALENCE-PROTOCOL.md:8-12`)
  already produces per-stage FP8 activations from which ranges follow.
- Use the FP8 model as a teacher: score the INT8 variant against FP8 outputs on captured frames, and, with a
  differentiable reimplementation, fine-tune scales or weights by distillation. That is model development, not
  porting.
- Keep the FP8 model's arithmetic instead (option (a) of RDNA3-ANALYSIS.md 4.5): F16 WMMA fed exact E4M3 values.

Cannot:
- Claim the INT8 variant is the same model. By the repo's own rules it is a new computation: MATCHES and CLOSE require
  at least 99 % of stage values equal and differences of at most 2 steps (`clean/EQUIVALENCE-PROTOCOL.md:10-11`); an
  INT8 activation grid with a different spacing than E4M3 will put most stage values on different steps, so the
  verdict would be DIFFERENT by construction (INFERRED). It needs a new end-to-end image-quality contract.
- Do QAT against ground truth, retrain, shrink the network (an AMD-style lightweight model), or prove "no visible
  loss" the way AMD claims to: AMD had the data, the pipeline and a large validation farm (1.3).
- Get matrix speed on RDNA 3 from the format change. IU8 and F16 WMMA have the same rate there (3.1). INT8 gains only
  operand registers and bandwidth on RDNA 3, plus RDNA 2 support through dot4.

Lesson for WrongGPU (INFERRED): AMD's INT8 choice is evidence that an 8-bit-integer network can match an FP8 one
visually after work by the model's owner. It is not evidence that a third party can convert one without loss, and
for RDNA 3 alone it is not a speed argument. AMD's result does give a speed reference: its INT8 model costs 2.33x the
RX 9070 XT's FP8 time on an RX 7900 XTX at 4K, about 2.7x per CU-cycle. Scaled to an RX 7900 XT (84 CU, 2.4 GHz)
against an RX 9070 XT (64 CU, 2.97 GHz) that is about 2.6x the RDNA 4 time for a network that behaves like FSR 4,
in line with RDNA3-ANALYSIS.md's "roughly 3x" for option (a).

## 6. Intel XeSS as precedent

- Intel ships two networks: "XMX-accelerated neural network on Intel ARC graphics" and "DP4a-based neural network on
  Intel iGPU and compatible Nvidia/AMD hardware"; "The XeSS API automatically selects the appropriate model". READ,
  Intel GameDev 2023 slides (https://game.intel.com/wp-content/uploads/2024/02/GameDev2023_XeSS_v1.pdf).
- Same slides, "Building XeSS for iGPUs": "No high-throughput matrix engine such as XMX"; "Smaller model, less
  parameters"; "Deeply optimized kernels"; "Leverage DP4a acceleration"; "The same model is a good fit for a large
  set of GPUs". So the DP4a path is a smaller, separately trained network, not the XMX weights re-run. READ.
- XeSS 1.1 shipped a "New Advanced XeSS Upscaling Model" (XMX) and a "New XeSS Upscaling Model" (others): two models
  retrained in parallel. READ (same slides).
- SDK today: "An HLSL-based cross-vendor implementation that runs on any GPU supporting SM 6.4. Hardware acceleration
  for DP4a or equivalent is recommended" and "An Intel implementation optimized to run on Intel Arc Graphics"
  (`doc/xess_sr_developer_guide_english.md`, https://github.com/intel/xess at de0fb9c). READ.
- XeSS 1.3 updates "apply to both the XMX ... and DP4a (INT8) modes"; "The original XeSS 1.0 was generally lacking in
  image quality, particularly when running in DP4a mode" (Tom's Hardware, 2024-04-05,
  https://www.tomshardware.com/pc-components/gpus/xess-13-improves-performance-by-up-to-28-with-refined-image-quality-adds-ultra-quality-plus-and-native-aa-modes).
  READ (press).
- Cost and quality against FSR 4: TechSpot measured FSR 4 INT8 ahead of XeSS DP4a on an RX 7900 XTX at equal mode
  and when performance-normalised (2025 article, 4.1). MEASURED-ELSEWHERE. TechSpot 2026 contrasts the two: "This is
  not a situation like Intel XeSS, where a higher-quality model is used for the XMX implementation ... while a
  lower-quality DP4a version runs on other hardware". IMPRESSION.
- Difference in approach: Intel cut network size for the no-matrix path; AMD kept a full-size INT8 model for discrete
  RDNA 2/3 and cut size only for small iGPUs. Neither publishes its quantization method. READ + INFERRED.

## Open questions

1. Did AMD obtain the INT8 model by QAT, PTQ plus tuning, or distillation, and with which scale granularity? Not
   public; only AMD can answer.
2. Do AMD's RX 7000 shaders use `v_wmma_i32_16x16x16_iu8`, `v_dot4_i32_iu8`, or both, and through which driver path?
   Settle with an RGP capture or RGA disassembly of the shipped pipeline on an RX 7000 (inspection of a shipped
   binary's ISA; check the DLL license before doing it).
3. Were the RX 7000 timings re-measured for FSR 4.1.2, and does 4.1.2 run faster than 4.1.1 on RX 7000 (H1)?
4. What share of an FSR 4 frame on RDNA 3 is matrix work versus requantization, packing and I/O (H6)? Same capture.
5. For WrongGPU: which contract would an INT8 variant be judged by, given it is DIFFERENT under the equivalence rules?
   A product decision for the maintainer.
6. How many of M's and RR's weights per output channel fall outside 4 binades of the channel maximum (5.2)? A CPU
   script over the model files answers it; no GPU needed.
7. Is AMD's lightweight model, or AMD's statement that RDNA 3 APUs get "lightweight models", relevant to WrongGPU's
   smaller targets? Only if WrongGPU ever trains models, which today it does not (INFERRED from 3.2: it ports
   existing weights).

## What to measure on an RX 7900 XT

1. Throughput of `v_wmma_i32_16x16x16_iu8`, `v_wmma_f32_16x16x16_f16` and `v_dot4_i32_iu8` loops per CU per clock at
   a fixed clock (expect all three near 512 ops per CU-clock if AMD's table holds; a gap would change option (b)'s
   case). Same test as findings/4-isa-driver.md item 6.
2. FSR 4.1 INT8 per-pass timing and instruction mix in a game, with RGP, at 4K Performance, with clocks logged: tests
   H2-H6 and gives a per-CU-cycle figure to compare with 184 M per 1080p-eq.
3. Same capture with the FSR 4.1.1 and 4.1.2 DLLs (H1).
4. Offline, no GPU: per-channel E4M3 -> INT8 exactness of M's and RR's weights (open question 6), and per-stage
   activation ranges from existing FP8 stage dumps, to size the INT8 rounding before anyone builds kernels.
5. If an INT8 variant is ever tried: image metrics (PSNR or the repo's step statistics) of INT8 vs FP8 outputs over
   captured frames at 1080p and 4K, since reviewers found the gap grows as render resolution falls.
