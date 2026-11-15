# Lane 7: running a float-format network on INT8 matrix instructions — prior art

Scope: what already exists for running an FP8 (E4M3) or FP16 network on integer matrix hardware, how well it works,
and what fits WrongGPU's case on RDNA 3 (`v_wmma_i32_16x16x16_iu8`, exact int32 accumulation, same rate as F16 WMMA).
Paths are relative to the repo root (upstream main 994f304). External facts carry their URL.

Labels: MEASURED-IN-REPO, READ (code, paper or docs read), MEASURED-ELSEWHERE (a number someone else published),
INFERRED (reasoning shown), UNKNOWN (with the test that settles it).

Clean-room note: no leaked FSR 4 source, mirror or DLL was opened. Facts about FSR 4's INT8 model come from AMD's
public SDK docs and press only. Web pages were read through searches and plain HTTP fetches of public pages.

Overlap: lane 4 (`findings/4-isa-driver.md` section 4) and lane 5 (`findings/5-linux-stack.md` section 2) already
cover vkd3d-proton's code and AMD's FSR 4.1 timings in detail. This file cites them and adds the quantization side.

## 1. Bottom line

1. **Re-quantizing to INT8 (what AMD did for FSR 4.1 on RX 7000) is a different model, and WrongGPU has no data to
   check it.** Post-training INT8 quantization of image-to-image networks usually costs 0.01-0.3 dB PSNR when a few
   hundred unlabelled inputs are available for calibration (section 3.3). The fully data-free methods (DFQ, ZeroQ)
   were made and measured on classifiers with batch norm. WrongGPU's FP8 models are already on a coarse grid, and
   Qualcomm reports FP8-E4 to INT8 conversion as "smooth" for convolutional networks (section 4.1). The fit is still
   poor for WrongGPU: every model would need a new accuracy contract, and the result cannot be bit-identical to RDNA 4.
2. **The exact digit-plane idea has strong prior art: the Ozaki scheme.** It splits floating-point matrices into
   integer slices, multiplies the slices on INT8 tensor cores with exact int32 sums, and recombines them. It has been
   used for FP64 GEMM since 2012 and on INT8 units since 2023 (section 4.4). WrongGPU's step "I"
   (`clean/rr_f/kernels/rr_step_i.hlsli:1-33`) is an Ozaki-I-style scheme specialised to E4M3. Block floating point
   (MSFP, OCP MX) is the approximate, one-slice version of the same idea.
3. **Nobody has published FP8 run on integer instructions for a game network.** Only two routes are public:
   - vkd3d-proton widens E4M3 to F16 and uses F16 WMMA. Its author measured 3 ms against 0.85 ms native on an
     RX 9070 XT, and the slowness was blamed mostly on software FP8 quantization and LDS round trips.
   - AMD shipped a separately quantized INT8 model. Its own table gives 3070 us on an RX 7900 XTX against 1316 us on
     an RX 9070 XT at 4K, and the press called the quality close to FP8.
4. **E4M3's "4 significant bits" do not make INT4 cheaper on RDNA 3.** A signed E4M3 mantissa needs 5 bits. gfx11's
   IU4 WMMA has K=16 like IU8 and twice its rate, so even with all exponents equal it costs the same as one IU8 plane
   (section 4.6). What the 4 bits do buy: one IU8 plane is exact whenever a block's exponent span is at most 3.
5. **Three prototypes worth building** (section 8): (P1) count the digit planes WrongGPU's real data needs, with the
   existing `clean/rr_f/gpu/lever_rates.py`; (P2) a one-plane-or-two-plane IU8 step with a bounded error, judged by
   the equivalence protocol; (P3) a calibration-from-game-frames INT8 path for one model, judged CLOSE / DIFFERENT, as
   a measured "what AMD's route would cost us" baseline.

## 2. WrongGPU's starting point (READ)

| Fact | Source |
|---|---|
| E4M3 value = M 2^(E-10), M = 0..15, E = max(exponent field, 1); range 2^-9 .. 448 | `clean/rr_f/kernels/rr_step_i.hlsli:5-6` |
| Weights of M, L: float32 tensors that the converter rounds to E4M3 (`fp8` op on 56 / 68 tensors); RR stores E4M3 codes | `RDNA3-ANALYSIS.md` 3.2; `data/preset-M.v1.map.json:113` |
| K and FG: FP16 weights and activations (F16 WMMA exists on gfx11, so nothing in this lane is needed for them) | `RDNA3-ANALYSIS.md` 3.2 and 1 |
| Weight quantization is part of the model definition (`q8` in the reference) | `clean/reference/swin.py:244` (cited in `RDNA3-ANALYSIS.md` 3.2) |
| Step I: an operand block is scaled to the integer v' = M 2^(E-beta), split into signed 7-bit digits (1, 2 or 3 int8 planes); one IU8 WMMA per digit pair; Horner over levels; exact while partial sums stay inside int32; one rounding to f16 | `clean/rr_f/kernels/rr_step_i.hlsli:7-33` |
| Block granularity: weights per (16-output tile, 32-input step), activations per (wave, step) | `clean/rr_f/kernels/rr_step_i.hlsli:11-13` |
| Weight layout for step I: two int8 digit planes plus a (beta, L1') table per (tile, step) | `clean/rr_f/gpu/layout.py:150-152` |
| The repo already models IU4 digits: 3-bit signed digits, count ceil((4 + span)/3) | `clean/rr_f/gpu/lever_rates.py:9-17, 47-52` |
| IU8 WMMA exact in int32 below 2^31 (on gfx12) | `clean/rr_f/kernels/rr_step_i.hlsli:35-36`, `clean/rr_f/gpu/probe_i8.py` (MEASURED-IN-REPO on RDNA 4) |
| Not committed: the lever rate results note (`results/2026-10-21-body-levers.md` is cited at `lever_rates.py:2` but is not in the tree), and no step I GPU timing in `clean/rr_f/gpu/README.md:60-100` | READ (absence) |

What WrongGPU does not have: training data, labels, or a training pipeline. It does have, or can capture, game frames
and the intermediate activations of its own runs (the RR proxy capture `clean/rr_f/proxy/CAPTURE.md`, the body dump
`sequence.py --dump-body` used by `lever_rates.py:3`). Those can serve as an unlabelled calibration set (INFERRED).

## 3. Quantization without training data

### 3.1 Methods

| Method | What it does | Data it needs | Reported loss | Source |
|---|---|---|---|---|
| Per-channel weight scales, min-max or MSE range setting | one scale per output channel instead of per tensor | weights: none; activations: a calibration pass | baseline of every tool below | ONNX Runtime docs (below) |
| DFQ: cross-layer equalization (CLE) + bias correction | rescales consecutive layers so per-channel weight ranges match (uses ReLU's scale equivariance); corrects the mean output shift that quantization causes, analytically from batch-norm statistics | **none** (analytic bias correction uses BN statistics) | "near-original" INT8 on MobileNet family; also segmentation and detection | https://arxiv.org/abs/1906.04721 (READ, abstract) |
| AIMET's version of the above | BN fold, CLE, high-bias absorption; bias correction "analytical" (no data) or "empirical" (data) | none or some | CLE + BC within ~1 % of FP32 INT8 on ResNet / MobileNet | https://quic.github.io/aimet-pages/AimetDocs/examples/torch/quantization/cle_bc.html (READ via search) |
| ZeroQ | builds a "distilled" synthetic input batch that matches the batch-norm statistics of each layer, then calibrates on it; also picks per-layer bit widths (Pareto frontier) | **none**, but needs BN layers | +1.71 % over DFQ on MobileNetV2; under 30 s for ResNet50 | https://arxiv.org/abs/2001.00281 (READ via search) |
| GZNQ and other generative zero-shot methods | a generator trained to reproduce BN statistics produces calibration images | none, but needs BN | classifier results only | https://arxiv.org/abs/2101.08430 (READ via search) |
| AdaRound | learns, per weight, whether to round up or down, minimising each layer's output error | unlabelled inputs, "typically 1024 images" | AIMET example: W8A8 nearest 49.85 %, AdaRound 81.21 %, FP32 82.20 % | https://quic.github.io/aimet-pages/releases/1.35.0/user_guide/adaround.html ; https://alphaxiv.org/overview/2004.10568v2 (READ via search) |
| BRECQ | AdaRound-style reconstruction per block of layers instead of per layer | unlabelled inputs (this family typically uses 128-1024) | strong at 4 bit; 8 bit is near-lossless in this family | https://arxiv.org/abs/2102.05426 (READ via search; exact sample count not checked) |
| Bit-Split and Stitching | splits each integer weight into bit planes, optimises them one by one, stitches them back | unlabelled inputs | near-original accuracy at INT3 without fine-tuning (classifiers, detection) | https://mlanthology.org/icml/2020/wang2020icml-accurate (READ via search) |
| Mixed precision | keeps sensitive layers at 16 bits (for example W8A16) | sensitivity analysis: a few inputs, or none (ZeroQ) | Qualcomm: "W8A16 mixed-precision for some layers solves most quantization issues in PTQ" | https://arxiv.org/html/2303.17951v2 section 7 (READ) |
| PTQ4SR (super resolution) | density-based dual clipping of long-tailed activations, pixel-aware calibration supervised by the float model | 100 unlabelled images | +2.091 dB over earlier PTQ on Urban100 for EDSR x4 at **4** bit; 8-bit table only in the supplement (not checked) | https://openaccess.thecvf.com/content/CVPR2023/html/Tu_Toward_Accurate_Post-Training_Quantization_for_Image_Super_Resolution_CVPR_2023_paper.html (READ, abstract) |

Which need zero data (READ, from the sources above): DFQ / CLE + analytic bias correction, ZeroQ, GZNQ, and plain
weight-only per-channel quantization. Every method that sets activation ranges or learns rounding needs inputs. Those
inputs are unlabelled, so frames or captured activations are enough.

Fit to WrongGPU (INFERRED): the zero-data methods rely on batch-norm statistics to stand in for data. WrongGPU's
models are transformer-like (Swin blocks in M / L per `clean/reference/swin.py`; RR's attention and MLP sites per
`lever_rates.py:6`) and are not BN-based convolutional networks. So "zero data" in practice means weight-only
methods (per-channel scales, CLE where the activation is ReLU-like). Activation ranges would come from game frames.

### 3.2 Tools

| Tool | Data it needs for INT8 activations | Notable | Source |
|---|---|---|---|
| ONNX Runtime | dynamic quantization: none (activation ranges computed at run time); static: a calibration set | calibration MinMax, Entropy, Percentile; per-channel; "reduce-range" 7-bit weights for U8S8 on AVX2 | https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html (READ) |
| AMD Quark (ONNX) | calibration set | built on ONNX Runtime's tool; MinMax, Percentile, MSE, Entropy, NonOverflow; CLE, bias correction, SmoothQuant, AdaRound, AdaQuant; "XINT8" preset uses power-of-two scales | https://quark.docs.amd.com/release-0.5.1/onnx/user_guide_feature_description.html ; https://quark.docs.amd.com/v0.11.1/supported_accelerators/ryzenai/ryzen_ai_best_practice.html (READ via search) |
| Intel NNCF / OpenVINO | calibration set; `subset_size` default **300** samples | `fast_bias_correction`; MIXED preset for non-ReLU activations | https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html (READ) |
| Qualcomm AIMET | CLE / analytic BC: none; AdaRound: unlabelled data | the reference implementation of DFQ and AdaRound | links in 3.1 |
| TensorRT | calibration for implicit INT8, or explicit Q/DQ nodes | formats listed: INT8, INT4 (weight-only), FP8 E4M3, FP4 E2M1 | https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html (READ) |
| TFLite / LiteRT | dynamic range: none (weights only); full integer: "around ~100-500 samples" | float16 weight quantization also offered | https://ai.google.dev/edge/litert/models/post_training_quantization (READ) |
| ncnn | calibration table from images; docs suggest "more than 5000 images" | `ncnn2table` | https://github.com/Tencent/ncnn/wiki/quantized-int8-inference (READ) |
| Brevitas | PTQ and QAT; equalization, bias correction, GPTQ | supports OCP and FNUZ FP8 and custom minifloats, so it can model E4M3 networks directly | https://huggingface.co/docs/optimum/amd/brevitas/api_reference ; https://arxiv.org/abs/2311.12359 (READ via search) |

None of these tools emits RDNA 3 WMMA code. Each would give WrongGPU only an offline recipe (scales, rounding,
equalisation) that its own kernels then use. That is how the converter already works today (INFERRED).

### 3.3 Accuracy at 8 bits for image-to-image networks

| Network / setting | Loss | Data used | Source |
|---|---|---|---|
| EDSR x4, W8A8, RDO-PTQ | about 0.01-0.02 dB on Set5 / Urban100; the abstract says "negligible performance loss" for SR | tiny calibration set | https://arxiv.org/abs/2211.02854 (abstract READ; dB figures from a search excerpt of its tables, not checked in the PDF) |
| EDSR x4, W8A6, PTQ4SR / AdaBM / OAPTQ | -0.17 to +0.15 dB on Test2K / Test4K | calibration images | https://arxiv.org/abs/2511.00682 (search excerpt) |
| ABPN (mobile SR), full INT8 PTQ | 30.22 -> 30.09 dB (-0.13) | calibration images | https://github.com/NJU-Jet/SR_Mobile_Quantization (search excerpt) |
| XLSR (Mobile AI 2021 winner), uint8 PTQ | about -0.3 dB vs float | calibration images | https://arxiv.org/abs/2105.10288 (search excerpt) |
| Mobile AI 2021 challenge | "the rest of the teams were not able to fight the accuracy drop resulted from model quantization" | trained models + DIV2K | https://arxiv.org/abs/2105.07825 (search excerpt) |
| On-device SR | a 0.1 dB drop "can still cause significant problems" on NPUs | — | https://arxiv.org/abs/2212.09501 (search excerpt) |
| Data-free compression of SR (distillation, not quantization) | -0.16 dB Set5 x2 | synthetic data from the teacher | https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Data-Free_Knowledge_Distillation_for_Image_Super-Resolution_CVPR_2021_paper.html (READ via search) |

Found nothing for: a data-free (zero-input) INT8 quantization paper on super resolution, denoising or frame
interpolation; any PTQ result on a temporal (history-feedback) upscaler. Two searches ("data-free quantization
super-resolution", "zero-shot quantization super-resolution") returned none. UNKNOWN whether one exists.

INFERRED for WrongGPU: (1) Those losses are against ground-truth images. WrongGPU's contract is against its own
reference (`RDNA3-ANALYSIS.md` 4.1), and a 0.1 dB average shift fails every bit-exact or per-output-tolerance test.
(2) Temporal models feed their output back as history, so a small per-frame bias can build up over frames. None of the
cited numbers covers that. (3) SR activations are "long-tailed, asymmetric and highly dynamic" (PTQ4SR abstract),
which is exactly where a single INT8 scale per tensor hurts and E4M3 helps.

## 4. Converting a float format rather than a float model

### 4.1 FP8 networks converted to INT8 (Qualcomm, READ)

van Baalen et al., "FP8 versus INT8 for efficient deep learning inference", https://arxiv.org/html/2303.17951v2 ,
section 6 and Table 4. Networks were trained with FP8-E4 quantization-aware training, then converted "naively" to
INT8 with MSE range setting and no retraining:

| Model | FP32 | INT8 PTQ | FP8 PTQ | FP8-E4 QAT | -> INT8 | -> W8A16 |
|---|---|---|---|---|---|---|
| ResNet18 | 69.7 | 69.55 | 68.49 | 69.79 | 69.84 | 70.00 |
| MobileNetV2 | 71.7 | 70.94 | 64.22 | 71.59 | 71.59 | 71.54 |
| BERT (GLUE avg) | 83.06 | 71.03 | 83.08 | 83.91 | 79.38 | 83.54 |
| SalsaNext (SemanticKITTI) | 55.8 | 52.8 | 55.1 | 55.2 | 54.1 | 54.2 |

- Their explanation: "INT8 can exactly represent roughly 90% of the range covered by the FP8-E4 format without any
  quantization error"; activation distributions depend on training settings, not on the number format.
- Their caveat: networks with outliers (transformers: BERT -4.5 points) lose accuracy in the conversion; W8A16 or INT8
  QAT recovers it.
- Bias: Qualcomm sells INT8 hardware, and the paper concludes FP8 is not worth building. Treat it as one party's view.

INFERRED for WrongGPU: its M, L and RR models are transformer-like (attention, MLP), the class that lost accuracy.
The table also covers classification only, with no image-to-image network.

### 4.2 What one INT8 scale does to E4M3 values (INFERRED, arithmetic)

Take a block (a channel, or a step-I block) whose largest E4M3 exponent is Emax, and an INT8 grid k·s with
s = 2^(Emax-6) (|k| <= 127; 127·s = 1.98·2^Emax covers the block's largest value 1.875·2^Emax). E4M3 values with
exponent E are spaced 2^(E-3) apart, so:
- values with E >= Emax-3 (the top 4 binades) land exactly on the grid: the same rule as
  `rr_step_i.hlsli:9` ("d1 = d2 = 0 when span <= 3");
- smaller values are rounded with absolute error <= 2^(Emax-7), about 1/240 of the block's largest value;
- a second int8 digit (14 bits) makes every value with E >= Emax-10 exact. Smaller ones then err by at most
  2^(Emax-14), about 1/30,000 of the block's largest value. A third digit makes all of E4M3 exact (span <= 14).

So "INT8 with per-block power-of-two scales" is E4M3 with flush-to-grid below 4 binades. How often that matters
depends on the exponent spans of real blocks. Those spans are what `lever_rates.py` measures, and its results are not
committed (section 2).

### 4.3 Logarithmic, power-of-two and block floating point

| Work | Idea | Relation to WrongGPU | Source |
|---|---|---|---|
| LogQuant / power-of-two weights | weight = ±2^k, a multiply becomes a shift | E4M3 with M fixed at 8; E4M3 is finer. Useful only as evidence that exponent-only formats work for CNNs | https://arxiv.org/abs/1603.01025 (Miyashita et al.; known reference, not re-read); FPGA comparison https://ijcai.org/proceedings/2022/219 (search excerpt) |
| APoT (additive powers of two) | each level a sum of 2-3 powers of two; 4-bit ResNet-50 76.6 % vs 76.4 % FP | E4M3 is itself "8 + mantissa" times a power of two; APoT's sum of shifted terms is the same decomposition as digit planes | https://arxiv.org/abs/1909.13144 (READ via search; the 76.6 % comes from a summary page) |
| MSFP (Microsoft, block floating point) | 16 elements share one 8-bit exponent; each keeps a sign and a few mantissa bits (MSFP12: 4.5 bits per element) | exactly "per-block exponent band, one integer plane": integer dot product inside the block, float between blocks | described in https://arxiv.org/abs/2510.14557 ; MX lineage https://arxiv.org/abs/2302.08007 (READ via search) |
| OCP MX (MXINT8, MXFP8) | block of 32, shared E8M0 scale = floor(log2 max) - emax_elem | MXINT8 with a block of 16-32 along K is exactly the one-plane variant of step I | https://arxiv.org/abs/2310.10537 (READ via search) |

Block floating point is approximate: values far below the block's largest lose low bits. It is the same trade as
4.2's single plane.

### 4.4 Exact integer-slice products: the Ozaki scheme (prior art for digit planes)

| Work | What it does | Cost reported | Source |
|---|---|---|---|
| Ozaki scheme I (2012) | error-free split of each float matrix into slices with a shared exponent per row (A) or column (B); slice products are exact in the hardware's accumulator; sum the slice products | 7-8 slices per operand for FP64, so tens of GEMMs | https://www.r-ccs.riken.jp/assets/uploads/sites/2/2025/12/FugakuNEXT_4th_ApplicationSeminar.pdf (READ via search) |
| Ootomo, Ozaki, Yokota: "DGEMM on Integer Matrix Multiplication Unit" (2023/24) | Ozaki-I with **INT8 slices and exact int32 accumulation** on INT8 tensor cores | faster than cuBLAS DGEMM on consumer NVIDIA GPUs; one deck reports 7x on L40s | https://arxiv.org/abs/2306.11975 (READ via search) |
| NVIDIA, "Guaranteed DGEMM accuracy ... extensions of the Ozaki scheme" (2025) | unsigned slicing that removes redundant sign bits; an "Exponent Span Capacity" estimator chooses the slice count per call | FP64-equivalent accuracy at under 10 % overhead | https://arxiv.org/abs/2511.13778 (READ via search) |
| Ozaki scheme II (2025) | Chinese-remainder method: products modulo several int8-sized moduli, no input slicing; the GEMM count grows **linearly** with precision instead of quadratically | 15-19 INT8 GEMMs for FP64 vs 28-35 for scheme I; faster than native DGEMM on GH200 (secondary source) | https://arxiv.org/abs/2504.08009 ; https://arxiv.org/abs/2508.03984 (READ via search) |
| Ozaki-II on FP8 units (2026) | CRT scheme adapted to FP8 MMA | each modulus needs 3 FP8 GEMMs vs 1 INT8 GEMM | https://arxiv.org/abs/2603.10634 (READ via search) |
| Hardware caveat | Hopper's FP16 / BF16 tensor cores truncate aligned significand bits that fall out of range | a reason to prefer integer units for exactness | https://arxiv.org/abs/2512.07004 (READ via search) |

Mapping to WrongGPU (INFERRED):
- Step I is Ozaki-I with three specialisations: the input is E4M3, so a value needs at most 18 integer bits and at most
  3 digits of 7 bits (`rr_step_i.hlsli:6-10`); the scale is per (tile, step) or (wave, step) instead of per row; and
  the result is rounded once per 32-product step to f16 (`rr_step_i.hlsli:23-33`) instead of accumulated to FP64.
- The 2025 NVIDIA work's span estimator is the same idea as `lever_rates.py`'s `span()` (`lever_rates.py:55-60`),
  which picks the digit count per block from the exponent span.
- Ozaki-II's linear scaling: 9 WMMAs for a 3x3-digit block would become about 5 residue WMMAs (moduli near 2^8,
  product > 2^32 covers |N| < 2^31). But every output of every step then needs a CRT reconstruction on VALU, and
  WrongGPU rounds after every 32-product step (`rr_step_i.hlsli:2-3`). The reconstruction cost per output per step,
  roughly 10-20 integer ops (an estimate, not counted), likely outweighs the saved WMMAs except in rare 3-digit
  blocks. UNKNOWN; an RGA instruction count for gfx1100 would settle it.

### 4.5 Error-compensated two-term splits (approximate and exact)

| Work | Idea | Source |
|---|---|---|
| Markidis et al. 2018; Ootomo & Yokota 2022 | x = hi + lo with both FP16; A·B ≈ hi·hi + hi·lo + lo·hi on FP16 tensor cores; with careful rounding it matches FP32 SIMT accuracy (51 TFLOP/s on A100 for a limited exponent range) | https://arxiv.org/abs/2203.03341 (READ via search) |
| WrongGPU step X | e = WMMA(a, b, -p): a second WMMA to capture the error of the first; 4.5x step E's time on RDNA 4 (29.70 vs 5.31 ms at 1440p) | `clean/rr_f/gpu/README.md:60-62`, table at `:79-96` (MEASURED-IN-REPO) |

The "two planes with bounded error" variant asked about is the integer form of this split: hi = the first 7-bit digit,
lo = the second, and dropping lo·lo (or the third digit) gives a bounded error (4.2). No published work states this
variant for FP8 specifically. Ozaki-I with a capped slice count is the closest (INFERRED).

### 4.6 Bit-serial and bit-plane products; the INT4 question

| Work | Idea | Source |
|---|---|---|
| APNN-TC (2021) | p-bit x q-bit products built from 1-bit tensor-core operations: decompose into bit planes, multiply, recombine with shifts | https://arxiv.org/abs/2106.12169 (READ via search) |
| Bit Fusion (ISCA 2018), Stripes (bit-serial) | hardware that fuses 2-bit units into wider products | https://arxiv.org/abs/1712.01507 (known reference, not re-read) |
| Bit-Split and Stitching | software bit planes for quantization, not for speed | 3.1 |

RDNA 3 rates (READ): per CU per clock, F16 512, IU8 512, IU4 1024 ops; https://gpuopen.com/learn/wmma_on_rdna3/
(via search). gfx11 has only `v_wmma_i32_16x16x16_iu4` (K = 16). The K=32 form is gfx11_7 / gfx12 only
(`findings/5-linux-stack.md` table in section 3).

INFERRED cost of E4M3 on IU4, in units of one IU8 WMMA:
- A signed 4-bit digit holds |d| <= 7 (`lever_rates.py:13`). E4M3's M = 8..15 needs 2 digits even at span 0, so
  digits4(span) = ceil((4 + span)/3) (`lever_rates.py:51-52`) gives at least 2 x 2 = 4 IU4 WMMAs = 2 IU8 units,
  against 1 IU8 unit at span <= 3.
- Unsigned IU4 holds M = 0..15 in one digit, but the sign then needs A = A+ - A- (and the same for B): 2 x 2 = 4
  IU4 WMMAs = 2 IU8 units again. With unsigned A and signed B (the instruction has per-operand sign flags,
  `findings/5-linux-stack.md`), 2 IU4 WMMAs = 1 IU8 unit, but only at exponent span 0.
- Conclusion: on gfx11, IU4 never beats IU8 for E4M3. It ties only in the degenerate span-0 case. The "4 significant
  bits" pay off through IU8's 7-bit digits: span <= 3 needs one plane.

### 4.7 FP8 (E4M3 / E5M2) inference on hardware without FP8

| Route | Who | Exact? | Source |
|---|---|---|---|
| Widen E4M3 to F16 in registers, F16 x F16 -> F32 matrix op | vkd3d-proton / dxil-spirv (Vulkan, RDNA 3 and RDNA 4 without FP8); also the common weight-only route on GPUs without FP8 (dequantize, then F16 GEMM) | products exact; accumulation differs from FP8 hardware | `findings/5-linux-stack.md` 2.2 |
| Quantize to INT8 offline | AMD FSR 4.1 on RX 7000 / RX 6000 | no: a different model | `findings/4-isa-driver.md` section 4 |
| Integer slices (step I) | WrongGPU only, as far as found | exact sum, one rounding | `clean/rr_f/kernels/rr_step_i.hlsli` |

Brevitas and the FPGA minifloat study (https://arxiv.org/abs/2311.12359) simulate minifloat formats in software for
PTQ research. They report that minifloats typically beat integers at 4 bits and above (READ via search). Those are
accuracy studies, not fast kernels.

## 5. Game and driver projects

| Project | What it does | Numbers published | Accuracy stated | Source |
|---|---|---|---|---|
| vkd3d-proton + dxil-spirv, FP8 emulation | E4M3 codes stay uint8 in memory; widened to F16 by bit tricks ("Sign-extend, shift, bit-and, and an fmul"); F16 WMMA; F32 -> FP8 by an F16 bit trick (round to nearest, ties toward zero, after an F32 -> F16 step: double rounding); accumulator -> B through LDS. RDNA 3 needs a source build flag plus `DXIL_SPIRV_CONFIG=wmma_rdna3_workaround` | RX 9070 XT, 1440p: native AMD 0.85 ms; RADV FP16 emulation ~3 ms; RADV native FP8 1.3 ms; after removing LDS round trips ~1 ms | "significant performance cost (and some cost to theoretical correctness)"; tests allow 8 ULP slack on RDNA 3; inf/NaN not handled | https://themaister.net/blog/2025/05/ (READ, fetched); `findings/5-linux-stack.md` 2.1-2.3 |
| Same, on RDNA 3 | as above | the author: "Given the terrible performance I got in FP16 emulation, I can understand why RDNA3 is not supported ... RDNA3's lesser WMMA grunt is simply not strong enough"; users: about 12 % fewer FPS than the INT8 DLL (forum, RUMOR) | — | blog above; https://discuss.cachyos.org/t/fsr-4-1-0-with-vanilla-wine-on-rdna3-yes-it-actually-works/23546 |
| Blog note on the conversion | "Doing accurate RTE with denorm handling in soft-float is GPU sadism. It explodes driver compile times and runtime performance is borderline unusable" | — | hence the inexact fast rounding | blog above (READ) |
| Mesa RADV | exposes `VK_KHR_cooperative_matrix` on GFX11+ only (ACO). Found no software lowering for gfx10.3 (RDNA 2) | — | — | https://gitlab.neroreflex.duckdns.org/NeroReflex/mesa/commit/653bcbcb091d4619bca6edfc44a0a2cfc4f30d50 (a fork's copy of the upstream commit, MR 24683); `findings/5-linux-stack.md` 3.3 |
| AMD FSR 4.1 on RX 7000 (official) | a separate INT8 model; how it was made (PTQ or retraining) is not stated publicly | AMD SDK table: RX 7900 XTX INT8 4K 3070 us vs RX 9070 XT FP8 4K 1316 us; RX 6900 XT INT8 4K 3190 us | PCGH: RX 7800 XT INT8 vs RX 9070 XT FP8 "nearly indistinguishable", even at low resolution | `findings/4-isa-driver.md` section 4 (AMD's `super-resolution-ml.md`); https://www.pcgameshardware.de/FidelityFX-Super-Resolution-Software-277617/Specials/FSR-4-1-Download-RDNA-3-7900-xtx-1546068/2/ |
| FSR 4.1 vs FSR 3.1 frame rate | — | ComputerBase (via Igor's Lab): up to 14 % loss on the RX 7900 XTX; 9-12 % on an RX 7800 XT and 10-13 % on an RX 6800 XT in Performance mode | — | https://www.igorslab.de/en/fsr-4-1-rdna-3-better-image-quality-rx-7900-xtx-up-to-14-percent-performance/ ; https://www.tomshardware.com/pc-components/gpus/optiscaler-team-fixes-int8-fsr-4-ghosting-on-rx-6000-series-gpus-adds-support-for-the-latest-adrenalin-drivers (via search) |
| OptiScaler + the 2025 INT8 files (unofficial) | loads AMD's INT8 FSR 4 build through OptiScaler on RX 6000 / RX 7000 | early: 10-20 % below FSR 3.1, up to 40 % in the first ports (HotHardware); TechSpot RX 6750 XT 1440p: FSR 4 INT8 Performance slightly slower than FSR 3 Quality | ghosting on RX 6000 fixed in OptiScaler 4.2.0b "much less of a problem" | https://wccftech.com/amd-fsr-4-enabled-rx-6800-xt-rdna-2-gpu-performance-hit-higher-image-quality/ ; https://hothardware.com/news/amd-fsr-41-on-radeon-rx-6000-tested ; https://www.techspot.com/article/3040-amd-fsr4-upscaling-older-radeon-gpu/ (via search) |
| Shader mod of the INT8 path on Linux | a report of WMMA peak ~52 TOPS vs dot4 ~55 TOPS on a 7900 XT | RUMOR (one report) | — | `findings/4-isa-driver.md` section 4 |
| Intel XeSS DP4a path | an INT8 network on `dp4a` (4 int8 products into int32) on any SM 6.4 GPU; XMX path on Arc | Intel: no "major performance loss" vs XMX; early reviews found DP4a often gave no FPS gain, and XeSS 1.0 DP4a quality was weaker; XeSS 1.3 retrained both paths | press calls the DP4a model "a little simpler"; Intel never named a separate model | https://www.pcgamer.com/intel-xess-compatibility-nvidia-amd ; https://www.tomshardware.com/pc-components/gpus/xess-13-improves-performance-by-up-to-28-with-refined-image-quality-adds-ultra-quality-plus-and-native-aa-modes (via search) |
| DLSS-derived networks on DP4a / other vendors | none found; vkd3d-proton's author: DLSS uses CUDA interop, "0% chance anyone outside of NVIDIA can deal with that" | — | — | blog above |

What these say about WrongGPU (INFERRED):
- Both shipping vendors (AMD for RX 7000, Intel for non-Arc GPUs) solved "no fast low-precision float" by shipping a
  **separately trained or quantized integer model**, not by emulating the float format. Both have the training data.
- The only float-format emulation in the field (vkd3d-proton) is about 3.5x native on RDNA 4. From the blog's own
  sequence, the F16 path's extra cost splits roughly into F16 vs FP8 rate plus widening (3 -> 1.3 ms when native FP8
  arrived) and LDS use conversions (1.3 -> 1 ms). WrongGPU's option (a) already plans in-register widening and a lane
  exchange instead of LDS (`RDNA3-ANALYSIS.md` 1). Its exact software quantizer costs 8-10 VALU per value
  (`RDNA3-ANALYSIS.md` 4.4), the step the blog called "borderline unusable" in RTE soft-float form.
- The FSR 4.1 RX 7900 XTX time (2.3x the RX 9070 XT) is a rough floor for an equally sized INT8 network on RDNA 3
  without emulation overhead. A route that does more integer work per product (digit planes) or uses F16 WMMA with
  widening sits above that floor.

## 6. Mixed designs

| Design | What it is | Fits WrongGPU? (INFERRED) | Source |
|---|---|---|---|
| Integer weights, float activations (weight-only, W8A16) | weights int8 or int4 with scales, activations FP16; the GEMM runs in FP16 after dequantization | gfx11 has no int8 x f16 WMMA, so this is option (a) with int8 instead of E4M3 storage: no gain over E4M3 storage, which already widens exactly | Qualcomm W8A16 column (4.1); TensorRT INT4 weight-only (3.2) |
| Residual (two-pass) quantization | q1 = Q(w), q2 = Q(w - s·q1) at a finer scale; w ≈ s·q1 + s'·q2 | the same as two digit planes, with rounding instead of exact digits; exact for E4M3 when span <= 10 | Bit-Split (3.1); Ozaki-I (4.4) |
| Error compensation | a second product to recover the first's rounding error | step X today (4.5); on IU8 the products are exact and need no compensation | 4.5 |
| Lookup-table multiply | table of products indexed by codes (T-MAC: bit-wise lookups, up to 4x llama.cpp on CPU; LUT-GEMM for weight-only GPU kernels) | poor: an E4M3 x E4M3 product is a 4x4-bit mantissa product plus an exponent add, so a 64 K-entry table saves nothing on a GPU with matrix units. LUT methods win when memory-bound on CPUs | https://arxiv.org/abs/2407.00088 (READ via search); LUT-GEMM https://arxiv.org/abs/2206.09557 (known reference, not re-read) |
| "4 significant bits" by exponent class | A = sum over classes c of 2^c A_c, each A_c holding 4-bit mantissas | 16 x 16 class pairs, far too many; grouping consecutive classes into 7-bit windows is exactly step I's digits, the efficient form of this idea | INFERRED from `rr_step_i.hlsli:5-16` |
| Per-row / per-column exponent bands | one scale per row of A and per column of B (the WMMA sums over K only) | allowed and already used (`lever_rates.py:10-12`): finer bands (per output row, per token) shrink the span and so the digit count. Scales must be uniform along K inside one WMMA | `lever_rates.py:10-14` |

## 7. Summary table

"Mults" = matrix-instruction work per original product, in units of one IU8 or F16 WMMA on gfx11 (same rate);
dA, dB = digit planes of the weight and activation block (1-3 each).

| Approach | Needs training data | Exact? | Mults per original | Reported quality loss | Fit to WrongGPU |
|---|---|---|---|---|---|
| (a) E4M3 widened to F16, F16 WMMA | no | exact products, float accumulation (RDNA 4-like) | 1 + widening VALU + exact quantizer (8-10 VALU per value) | vkd3d-proton tests: 8 ULP slack (inexact quantizer); with an exact quantizer, UNKNOWN until T1 | good: keeps the model; speed is the question |
| Digit planes, exact (step I, Ozaki-I) | no | exact sum, one rounding | dA x dB (1-9); 1 when both spans <= 3 | none against the reference | good where spans are small; cost depends on data (UNKNOWN until P1) |
| Digit planes capped at 2 (bounded error) | no | error <= 2^-14 of the block max per dropped value | <= 4 | none published; bound by arithmetic (4.2) | good if the contract allows CLOSE; needs P1 numbers |
| One plane (MXINT8-like, per-block power-of-two scale) | no | values 4+ binades below the block max rounded (<= 1/240 of the max) | 1 | MX / MSFP papers: small on their models; none on SR | a cheap "CLOSE" candidate; error uncontrolled on wide-span blocks |
| Ozaki-II (CRT residues) | no | exact | ~4-5 + CRT VALU per output per step | none | weak: per-step reconstruction cost |
| PTQ to INT8 (per-channel, CLE, bias correction) | no for weights; a few hundred frames for activations | no | 1 | 0.01-0.3 dB on SR (with data) | weak: new model, new contract, no ground truth to check against |
| PTQ + AdaRound / BRECQ / PTQ4SR | a few hundred to ~1000 unlabelled frames | no | 1 | best PTQ results; near-lossless at 8 bit | as above; frames are available, labels are not needed |
| FP8-E4-trained network -> INT8 (Qualcomm) | a few frames for range setting | no | 1 | CNNs: none; transformers: up to -4.5 points (BERT) | the most relevant published number; WrongGPU's models are the risky class |
| Retrained INT8 model (AMD FSR 4.1, XeSS DP4a) | yes (full training) | no | 1 | press: close to FP8 | not possible: no training data |
| IU4 planes | no | exact | >= 1 at span 0, 2+ otherwise | none | worse than IU8 on gfx11 (4.6) |
| Bit-serial / 1-bit planes (APNN-TC) | no | exact | many | none | no: gfx11 has no 1-bit WMMA |
| LUT products | no | exact | VALU per product | none | no |
| (d) VALU FMA / `v_dot2_f32_f16` | no | depends on rounding model | 16-32x VALU | — | slowest (`RDNA3-ANALYSIS.md` 4.5) |

## 8. Three approaches most worth a prototype

**P1. Measure the digit-plane counts first (cheap, no GPU).** `clean/rr_f/gpu/lever_rates.py` already computes the
distribution of dA, dB and dA x dB per site, and the share of outputs past 2^24 and 2^31, from a body dump (READ,
`lever_rates.py:1-23`). Its results note is not in the tree. Run it on real captured frames (M, L and RR, several
scenes) and commit the output. It decides whether exact integer steps cost close to 1x (most blocks span <= 3),
about 4x (span <= 10), or 9x, against option (a)'s 1x plus widening. It also sizes the bounded-error variant. Every
other decision in this lane depends on these numbers. Prior art says the approach works (Ozaki-I on INT8 units),
so the open question is WrongGPU's own data, not the method.

**P2. Adaptive 1-or-2-plane IU8 step with a stated error bound.** Like step I, but each (tile, step) block uses one
plane when its span is <= 3 (exact), two when it is <= 10 (exact), and two with the third digit dropped otherwise:
bounded error, flagged for the existing fixup if exactness is required. It reuses step I's packing, Horner and
rounding code (`rr_step_i.hlsli:7-33`), needs no training data, and gives exact results on most blocks. Its cost
range is known after P1. Judge it two ways: bit-exact against the CPU reference where no block was capped, and
EQUIVALENCE-PROTOCOL CLOSE / MATCHES elsewhere. It is the strongest candidate if gfx11's F16 WMMA fails T1
(`RDNA3-ANALYSIS.md` 4.3), because its correctness does not depend on undocumented float accumulation.

**P3. A calibration-only INT8 baseline for one model, offline first.** Take one FP8 model (M is the smallest
contract surface), apply per-channel weight scales + CLE (zero data) and activation ranges from ~300 captured frames
(NNCF's default size; Brevitas or AIMET can simulate it on the CPU reference). Compare its outputs to the FP8
reference with the equivalence protocol. This answers "what would AMD's route cost us in quality" with WrongGPU's own
metric, before any kernel work. If it comes out CLOSE, a 1-plane IU8 kernel (the fastest RDNA 3 option) becomes a
valid, explicitly approximate mode. If it comes out DIFFERENT, that settles the matter and the lane can drop it.

Not recommended for prototyping: IU4 planes (4.6), Ozaki-II CRT (4.4), LUT multiplies (6), and retraining (no data).

## Open questions

1. Which contract must RDNA 3 meet: bit-identical to RDNA 4, exact against the reference, or CLOSE? It decides
   between (a), P2 and P3 (`RDNA3-ANALYSIS.md` 4.5 asks the same).
2. Would the maintainer accept an explicitly approximate integer mode (P2 capped, or P3) as a user-visible option?
3. Where is `results/2026-10-21-body-levers.md`? `lever_rates.py:2` cites it, but it is not in the tree at 994f304.
   Its numbers would answer P1 at once.
4. Did AMD make FSR 4.1's INT8 model by PTQ from the FP8 one, or by retraining? Not public. It bounds how far PTQ alone
   can get for this kind of network.
5. Do RR's activation blocks (per wave, per step) have wider exponent spans than its weights? Step I gives weights two
   planes (`layout.py:150`) but activations up to three (`rr_step_i.hlsli:8-10`).
6. Is there any published PTQ result for a temporal upscaler or denoiser with history feedback? None found. Drift
   over frames is the unmeasured risk of P3.

## What to measure on an RX 7900 XT

1. `probe_i8.py` (H1'): is `v_wmma_i32_16x16x16_iu8` exact in int32 on gfx11, with the same no-saturation two's
   complement behaviour as on gfx12? Everything in P1 and P2 assumes it.
2. Throughput per CU per clock at a fixed clock: IU8 WMMA, IU4 WMMA (16x16x16), F16 WMMA, and a
   `v_dot4_i32_iu8` VALU loop, in dependent and independent chains. This checks GPUOpen's 512 / 1024 / 512 table and
   the 52-vs-55 TOPS rumor.
3. Step I's real cost against option (a) on one RR stage (E1 at 1440p): time per stage with 1, 2 and 3 forced planes,
   and the extra VALU for digit extraction and the int32-to-f16 rounding (`rr_step_i.hlsli:23-33`).
4. The cost of the CRT reconstruction per output, counted in RGA for gfx1100, only if P1 shows many 3-digit blocks.
5. For P3: the same INT8 kernel on RDNA 4 and RDNA 3 should give byte-identical outputs, since integer sums are
   exact on both. Check that; it would give RDNA 3 a cross-family bit-exact path for any integer mode.
