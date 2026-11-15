# RDNA 3 F16 WMMA: how often do its FP32 errors change a quantized value?

CPU-only analysis of the saved RX 7900 XT probe outputs (`probe-vk/results/*.npz`). MEASURED = counted here from those files;
INFERRED = reasoning. Script: `post_quant.py` (compute, 3 min on one core), `render.py` / `make_post.py` (tables), `selfcheck.py`.

## Headline (MEASURED)
- F32-accumulate, p1b data: **451 per million outputs** change their E4M3 code vs the exact sum (945 of 2,097,152; 941 of them by one code) and **1,472 per million** change their FP16 value. Across the 8 data modes: 4 to 1,030 (q8), 259 to 3,345 (RN16).
- Against M-g8 (RDNA 4 fit) the rates are the same: 450 (q8), 1,455 (RN16). M-g8 and the exact sum themselves differ by only 9 (q8) / 66 (RN16) per million, so this card vs RDNA 4 is about the same question.
- Uniform-random E4M3 (e4_random): 2 per million (q8; 61 of 33,554,432), 212 (RN16). q8 changes are one code in 99 % of cases (p1b 941 of 945); RN16 changes are one step in 95 %.
- F16-accumulate instruction: 1.75e5 per million of p1b outputs differ from RN16(exact) (4.2e5 for e4_random), but most of that is the instruction rounding after every pair of products (M-g2), not the artefact; vs M-g2 it is 5.0e4 (p1b).
- The rate is set by how many results sit near a rounding boundary, so it depends on cancellation depth and on exact ties (section 2).

Method: D_exact = RN32(C + exact sum) from `exact.py` (float64 two-sum, equal to the Python-int backend on a 16,384-output check); D_model likewise (`models.py`).
q8 = `fp8.q8` (RNE, saturate 448); RN16 = float16 RNE. A *flip* = quantized values compare unequal (+0 and -0 count as equal; they differ in code only on 54 of 16,384 sampled adv outputs).
Size = distance in E4M3 codes / FP16 steps. Non-finite inputs and FP16 overflow excluded. Not measured: a bias add or activation before the quantizer.

## 1. Flip rates, F32-accumulate (`v_wmma_f32_16x16x16_f16`)

#### A. 7900 XT vs exact sum (modes: 0,1,2,5 sparse full-range; 3 dense full-range; 4 large exponents; 6 sparse small; 7 dense small with wide C)

| data | outputs | q8 differs: count (per M) | 1 code / 2+ | RN16 differs: count (per M) | 1 step / 2+ |
|---|---|---|---|---|---|
| e4_random | 33,554,432 | 61 (2) | 61 / 0 | 7,119 (212) | 6,756 / 363 |
| **p1b all modes** | 2,097,152 | 945 (451) | 941 / 4 | 3,087 (1,472) | 2,997 / 90 |
| p1b mode 0 | 262,144 | 122 (465) | 122 / 0 | 200 (763) | 194 / 6 |
| p1b mode 1 | 262,144 | 196 (748) | 196 / 0 | 372 (1,419) | 344 / 28 |
| p1b mode 2 | 262,144 | 105 (401) | 105 / 0 | 450 (1,717) | 442 / 8 |
| p1b mode 3 | 262,144 | 1 (4) | 1 / 0 | 68 (259) | 63 / 5 |
| p1b mode 4 | 262,144 | 66 (252) | 62 / 4 | 877 (3,345) | 845 / 32 |
| p1b mode 5 | 262,144 | 181 (690) | 181 / 0 | 298 (1,137) | 291 / 7 |
| p1b mode 6 | 262,144 | 270 (1,030) | 270 / 0 | 340 (1,297) | 337 / 3 |
| p1b mode 7 | 262,144 | 4 (15) | 4 / 0 | 482 (1,839) | 481 / 1 |
| **adv all** | 2,097,152 | 71,279 (33,988) | 55,583 / 15,696 | 344,030 (164,046) | 23,813 / 320,217 |
| adv sub-kind 0 | 262,144 | 58,298 (222,389) | 45,972 / 12,326 | 261,874 (998,970) | 1,596 / 260,278 |
| adv sub-kind 1 | 262,144 | 12,802 (48,836) | 9,432 / 3,370 | 81,503 (310,909) | 21,584 / 59,919 |
| adv sub-kind 5 | 262,144 | 1 (4) | 1 / 0 | 93 (355) | 91 / 2 |
| adv sub-kind 7 | 262,144 | 178 (679) | 178 / 0 | 548 (2,090) | 530 / 18 |

adv sub-kinds 2, 3, 4, 6 (ties, big C, near overflow): 0 q8 flips; sub-kind 0 (C cancels the big product) flips 22 % of outputs. adv is a stress set, not real data.

#### B. 7900 XT vs M-g8 (stand-in for RX 9070 XT) and C. exact vs M-g8 (no 7900 XT)

| data | outputs | q8 differs: count (per M) | 1 code / 2+ | RN16 differs: count (per M) | 1 step / 2+ |
|---|---|---|---|---|---|
| B e4_random | 33,554,432 | 62 (2) | 62 / 0 | 7,256 (216) | 6,887 / 369 |
| B p1b all | 2,097,152 | 943 (450) | 939 / 4 | 3,051 (1,455) | 2,964 / 87 |
| B adv all | 2,097,152 | 71,791 (34,233) | 55,920 / 15,871 | 344,966 (164,493) | 23,832 / 321,134 |
| C e4_random | 33,554,432 | 7 (0) | 7 / 0 | 1,492 (44) | 1,420 / 72 |
| C p1b all | 2,097,152 | 18 (9) | 18 / 0 | 139 (66) | 114 / 25 |
| C adv all | 2,097,152 | 6,082 (2,900) | 5,453 / 629 | 59,354 (28,302) | 15,405 / 43,949 |

Per-mode versions of B and C are in `fragments.md`. For p1b the 7900 XT sits no closer to M-g8 than to the exact sum.

## 2. Why that rate: cancellation depth and ties (MEASURED)

Depth d = exponent of the largest of the 17 terms minus exponent of the exact result. Error is about u x 2^(E-24) with u = 1.0-1.4 (mean) once terms have mixed signs.

| depth | e4 share | e4 q8/M | e4 RN16/M | p1b share | p1b q8/M | p1b RN16/M | adv share | adv q8/M | adv RN16/M |
|---|---|---|---|---|---|---|---|---|---|
| <=0 | 79.3 % | 0 | 41 | 84.6 % | 188 | 882 | 67.1 % | 30 | 229 |
| 1 | 13.2 % | 0 | 151 | 7.6 % | 1,127 | 3,411 | 2.5 % | 935 | 3,237 |
| 2 | 4.0 % | 2 | 365 | 2.6 % | 3,403 | 5,777 | 1.1 % | 1,792 | 5,375 |
| 3 | 1.8 % | 8 | 755 | 1.2 % | 4,185 | 8,167 | 0.8 % | 1,035 | 7,242 |
| 4-5 | 1.3 % | 12 | 2,260 | 0.9 % | 2,827 | 12,082 | 1.3 % | 868 | 9,729 |
| 6-8 | 0.4 % | 104 | 10,013 | 0.2 % | 14,174 | 20,363 | 1.9 % | 1,281 | 23,005 |
| 9-12 | 0.1 % | 832 | 84,661 | 0.0 % | 6,974 | 51,604 | 2.5 % | 4,868 | 145,779 |
| 13-16 | 0.0 % | 12,071 | 645,311 | 0.0 % | 0 | 578,125 | 2.9 % | 16,607 | 833,395 |
| >16 | 0.0 % | 72,289 | 975,904 | 0.0 % | 274,510 | 882,353 | 13.6 % | 244,493 | 996,672 |
| exact 0 | 0.0 % | 0 | 1,000,000 | 3.0 % | 113 | 516 | 6.2 % | 145 | 1,261 |
| all | 100 % | 2 | 212 | 100 % | 451 | 1,472 | 100 % | 33,988 | 164,046 |

Two mechanisms, both visible in the table:
1. **Away from ties** (random position between two grid values): flip probability = error / grid spacing = u x 2^(d-21) for E4M3 (3 mantissa bits) and u x 2^(d-14) for FP16 (10 bits).
   Check on e4_random (q8, u = 1.19): d=7: formula 73 vs measured non-tie 64 per M; d=10.5: formula 822 vs measured non-tie 595 per M; d=14.5: formula 13,148 vs measured non-tie 11,184 per M. Little cancellation (d <= 3) gives 0-8 per M; 10 binades of cancellation gives about 1,000 per M.
2. **Exact ties**: E4M3 products have few mantissa bits, so with sparse or low-precision data the exact sum often lies exactly halfway between two q8 values. A tie flips only when the GPU error is nonzero and points the wrong way: 6 % of p1b ties, 25 % of e4_random, 28 % of adv (more at larger depth)
   (p1b: 13,440 of 2,097,152 outputs are q8 ties, 6.0 % of them flip; 811 of the 945 q8 flips are at ties). Random E4M3 has almost no ties (2 per M), so its rate is tiny.

Predicting for other data: rate_q8 = sum over depth of share(d) x u x 2^(d-21), plus tie_fraction x tie_flip_rate (6-28 % above), and the same with 2^(d-14) for RN16. RN16 ties are also common (p1b: 2,525 of 3,087 flips).
p1b has 85 % of outputs at d <= 0, which is why its rate is dominated by ties; adv has 14 % at d > 16, which is why it flips 3-4 % of outputs.

## 3. F16-accumulate (`v_wmma_f16_16x16x16_f16`, C = RN16(C)) (MEASURED)

| data | outputs | q8 differs: count (per M) | 1 code / 2+ | RN16 differs: count (per M) | 1 step / 2+ |
|---|---|---|---|---|---|
| vs RN16(exact) e4_random | 25,406,205 | 73,034 (2,875) | 61,963 / 11,071 | 10,702,293 (421,247) | 8,846,734 / 1,855,559 |
| vs RN16(exact) p1b all | 1,607,032 | 2,816 (1,752) | 2,537 / 279 | 281,133 (174,939) | 233,761 / 47,372 |
| vs RN16(exact) p1b mode 0 | 220,746 | 66 (299) | 60 / 6 | 6,401 (28,997) | 6,035 / 366 |
| vs RN16(exact) p1b mode 7 | 179,868 | 1,214 (6,749) | 1,165 / 49 | 85,220 (473,792) | 67,879 / 17,341 |
| vs RN16(exact) adv all | 1,076,608 | 185,276 (172,092) | 69,317 / 115,959 | 566,200 (525,911) | 106,138 / 460,062 |
| vs M-g2 e4_random | 25,406,205 | 10,894 (429) | 9,346 / 1,548 | 1,551,983 (61,087) | 1,274,880 / 277,103 |
| vs M-g2 p1b all | 1,607,032 | 767 (477) | 696 / 71 | 79,558 (49,506) | 64,374 / 15,184 |
| vs M-g2 p1b mode 0 | 220,746 | 21 (95) | 21 / 0 | 3,548 (16,073) | 3,437 / 111 |
| vs M-g2 p1b mode 7 | 179,868 | 393 (2,185) | 370 / 23 | 23,495 (130,624) | 17,206 / 6,289 |
| vs M-g2 adv all | 1,076,608 | 58,193 (54,052) | 47,354 / 10,839 | 318,412 (295,755) | 33,673 / 284,739 |

Outputs with an FP16 overflow in C or the result are excluded (e4_random 8.1 M, p1b 0.49 M, adv 0.50 M), so counts are on fewer outputs. In this table the first (q8) columns apply q8 to the FP16 result.
Against RN16(exact) the card is far off because it rounds to FP16 after every pair of products; the part M-g2 does not explain is 6.1 % of e4_random outputs and 5.0 % of p1b.

## 4. Scale-up (INFERRED, back of the envelope)

- Differing codes per stage = N x r x 1e-6, with r = flips per million outputs. Per frame = sum over the stages: N_total x r x 1e-6.
- Worked, N = 1e6 values per layer, p1b rates: q8 451 per M gives **about 450 codes per stage** (4 to 1,030 depending on the data mode); RN16 1,472 gives about 1,500 FP16 values. A 20-layer frame: about 9,000 q8 codes (range 80 to 21,000).
- Rate 451 per M is 1 in 2,200 outputs; random E4M3 would be 2 per M (about 2 per stage). Real weights and activations are neither, so measure a real layer before relying on either number.
- If a ReLU or a clamp zeroes about half the outputs, the count roughly halves (not measured). A bias add moves the sum off its tie, which would lower the tie share (not measured).
- Cascade: a differing code is 1 step (6-12 % of the value for E4M3) in the next layer's input. It touches every output that reads it (fan-out F, for example K x K x channels), and each of those can flip with probability about weight x step / grid spacing.
  If F x that probability is above 1 per stage the difference spreads; below 1 it dies out. Not measured: replay a real network on CPU with injected one-code errors to find which.
- 1 in 2,200 is a different claim from "bit identical": the frame can still differ in a few thousand codes yet stay visually close, which a CLOSE verdict would allow and a byte-exact check would not.

## Limits
- Rates are for these probe data sets (uniform E4M3, P1b mimic, adversarial), 16x16x16 tiles, no bias/activation. The p1b rate varies 250x across modes, so treat 451 per M as a central guess, not a bound.
- Exact reference is RN32 first, then the quantizer (the pipeline's order). Rounding the exact sum straight to E4M3 would change only the rare double-rounding cases (not counted).
- Only 7900 XT outputs were available; the RX 9070 XT numbers use M-g8 as a proxy, which fit 2 of 51 unexplained samples.
