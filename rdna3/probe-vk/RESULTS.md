# RDNA 3 coopmat probe: does `v_wmma_f32_16x16x16_f16` round like RDNA 4's FP8 WMMA?

Short answer (MEASURED unless marked):

- **F16 -> F32 (`v_wmma_f32_16x16x16_f16`): no model fits.** When terms have mixed signs, the result is off by a few
  units of 2^(E-24), where E is the exponent of the largest term. Even 1 + (-1) returns -2^-24, not 0 (section 3.1). The
  best of the 13 models (M-g2) still mismatches 25 % (p1b), 37 % (adv), 48 % (random E4M3) and 63 % (full-range FP16) of
  outputs. M-exact and the repo's RDNA 4 best fit M-g8 both mismatch about 52-53 % of all 33.5 M random-E4M3 outputs.
  The errors stay small: over 99 % are within 4 units.
- **F16 -> F16 (`v_wmma_f16_16x16x16_f16`): close to M-g2.** M-g2 means: start from C, add the products two at a time
  in K order (k0+k1, k2+k3, ...), and round to FP16 after each pair. It explains all but 38 of 63,000 full-range FP16
  outputs, but only about 95 % of E4M3 outputs and 72 % of adversarial ones. INFERRED: the mixed-sign artefact also
  shows up when it reaches FP16 precision.
- Deterministic across runs; does not depend on lane or dispatch position; **does** depend on the K order (26-39 % of
  outputs change under a K permutation).
- NaN and inf propagate as IEEE expects; FP16 subnormal inputs are honoured; an all-zero sum gives +0. The iu8
  instruction is exact (wrapping); the saturating variant differs from "clamp the exact sum" on 2,241 outputs, where
  2,200 overflow.

So feeding E4M3 values to RDNA 3's F16 WMMA cannot be bit-identical to RDNA 4 unless RDNA 4's FP8 WMMA shows the same
mixed-sign artefacts with the same K-slot dependence. The RDNA 4 replay (section 5) settles that (INFERRED).

Labels: MEASURED = observed on this card in this probe; INFERRED = reasoning from those observations.

## 1. Setup

| Item | Value | Label |
|---|---|---|
| Device | AMD Radeon RX 7900 XT (RADV NAVI31), vendor 0x1002, device 0x744c, Vulkan API 1.4.366 | MEASURED |
| Driver | radv / Mesa 27.0.0-devel (conformance 1.4.5.3) | MEASURED |
| Excluded | the integrated GPU 1002:13c0 (the harness only accepts 1002:744c) | MEASURED |
| Subgroup | required subgroup size 32, full subgroups, 4 subgroups per workgroup, one 16x16x16 problem per subgroup | MEASURED (pipeline reports subgroup size 32) |
| Harness | `wmma_probe.c`: plain Vulkan compute, no surface, swapchain or WSI extension; host-visible buffers, under 900 MiB | MEASURED |
| Shader | `mm.comp` (GLSL, `GL_KHR_cooperative_matrix`): `coopMatLoad` A (use A), B (use B), C (accumulator) row-major with stride 16, `coopMatMulAdd`, `coopMatStore` | MEASURED |

### Advertised `VkCooperativeMatrixPropertiesKHR` (MEASURED, `out/props.txt`)

All 16x16x16, subgroup scope; supported stage: compute only (0x20).

| A | B | C = Result | saturatingAccumulation |
|---|---|---|---|
| u8 | u8 | u32 | 0 |
| u8/s8 (all 4 sign mixes) | u8/s8 | s32 | 0 and 1 (both listed) |
| u8/s8 (the 3 other mixes) | u8/s8 | u32 | 0 |
| f16 | f16 | f16 | 0 |
| f16 | f16 | f32 | 0 |

14 entries in total. No BF16 and no FP8 entries. INFERRED: this matches the lane-5 reading of Mesa (FP8 only for gfx11_7 and later; BF16 behind an option).

### Disassembly evidence (MEASURED)

Captured two ways: through `VK_KHR_pipeline_executable_properties` (`disasm/*.txt`, "Final Assembly") and with
`RADV_DEBUG=shaders` (`disasm/radv_debug_shaders_f16_f32.txt`). Each pipeline contains exactly one matrix instruction:

```
f16 -> f32 : v_wmma_f32_16x16x16_f16 v[24:31], v[8:15], v[16:23], v[24:31] ; cc404018 1c622108
f16 -> f16 : v_wmma_f16_16x16x16_f16 v[24:31], v[8:15], v[16:23], v[24:31] ; cc424018 1c622108
s8 -> s32  : v_wmma_i32_16x16x16_iu8 v[40:47], v[4:7], v[8:11], v[40:47] neg_lo:[1,1,0] ; cc444028 7ca21104
s8 -> s32 saturating : v_wmma_i32_16x16x16_iu8 v[40:47], v[4:7], v[8:11], v[40:47] neg_lo:[1,1,0] clamp ; cc44c028 7ca21104
u8 -> u32  : v_wmma_i32_16x16x16_iu8 v[40:47], v[4:7], v[8:11], v[40:47] ; cc444028 1ca21104
```

The F16 instructions carry no neg/abs modifiers (MEASURED). On the iu8 instruction, `neg_lo` marks A and B as signed. The loads
(MEASURED, "Final Assembly"): A and B are read by lane & 15, so lanes 16-31 repeat lanes 0-15 (the gfx11
replication); C and D use 8 VGPRs, with lanes 0-15 holding even rows and lanes 16-31 odd rows. Nothing else touches
the values between the loads and the stores.

## 2. Inputs (fixed seeds, `gen.py`)

| Class | Problems | Content |
|---|---|---|
| `e4_random` | 131072 | A, B uniform over the 254 finite E4M3 codes; C as in P1b (zero, full-mantissa f32 2^-24..2^21, FP16-like, x1024) |
| `p1b` | 8192 | P1b's 8 data modes (`clean/rr_f/gpu/probe_p1b.py`) on 16x16 operands; mode 7 also scales C by 2^10 ("wide C") |
| `adv` | 8192 | 8 sub-kinds, E4M3 operands: cancel_acc, cancel_pair, ties_one, ties_split, big_c, spread, near_ovf, sgn_zero (docstring in `gen.py:adv`) |
| `specials` | 2048 | C = +-inf, C = NaN, a NaN E4M3 code in A, C = +-inf everywhere |
| `f16_full` | 4096 | full-range FP16 operands with subnormals (|x| < 2^8), uniform(-2, 2), subnormal A, lowest binades |
| `kperm` | 16384 | 1024 `e4_random` + 1024 `p1b` problems, each with 8 permutations of the 16 K slots (A columns and B rows together) |
| `pos_shift`, `pos_perm` | 16384 each | the first 16384 `e4_random` problems moved by one problem slot, or with A rows and B columns permuted |
| `int_s8`, `int_u8` | 4096 each | random int8 / uint8, random int32 / uint32 C; a quarter near the int32 limits |

For the F16-accumulate instruction, C is RN16(C) (`C16`); values above 65504 become inf.

Expected values: E4M3 classes use float64, which is provably exact here. Every product is a multiple of 2^-18 below 2^17.6, so any
sum of up to 16 products fits in 40 bits; one addition of C is handled exactly with TwoSum plus a midpoint check
(`exact.py`). The FP16 class uses Python integers in units of 2^-200. Product sums and all models use the K order
of the instruction (k = 0..15 = column k of A).

## 3. Headline measurements

### 3.1 Two-term examples (MEASURED, `out/terms.txt`, B = all ones so D[m, n] = sum_k A[m, k] + C[m, n])

| Terms (A row) | C | GPU result | Exact (= RN32) |
|---|---|---|---|
| 3, 5 | 0 | 8 | 8 |
| 1, -1 | 0 | **-2^-24** | 0 |
| 8, -1 | 0 | 7 - 2^-21 (0x40DFFFFF) | 7 |
| -8, 1 | 0 | -7 - 2^-21 | -7 |
| 2, -1 | 0 | 1 - 2^-23 | 1 |
| 4, -3 | 0 | 1 - 2^-22 | 1 |
| 1, -0.5 | 0 | 0.5 - 2^-24 | 0.5 |
| 1 (k=0), -0.5 (k=8) | 0 | 0.5 - 2^-25 | 0.5 |
| 1, -1, 1 | 0 | 1 | 1 |
| 8 | -1 | **7 + 2^-20** | 7 |
| 1 | -0.5 | 0.5 + 2^-23 | 0.5 |
| 1 | -2^-60 | 1 + 2^-23 | 1 |
| -1 | -2^-30 | -(1 - 2^-24) | -1 |
| 32768 | -1 | 32767 + 2^-8 | 32767 |
| 0 or -0 | -1 / 1 | -1 / 1 | exact |

INFERRED: when terms have mixed signs, the product path loses about one unit at 2^(E-24) toward minus infinity. That
looks like negation by bit inversion (one's complement) with no carry-in, followed by truncation. A negative C goes
through a different path and lands **above** the exact value. Where a term sits matters too: the same pair loses
2^-24 in K slots 0 and 1 but 2^-25 in slots 0 and 8. Same-sign terms are mostly, but not always, exact. On `p1b`,
take the C = 0 outputs whose exact sum fits in f32 (`explore5.py`, MEASURED). Of 66,800 with no negative product,
142 differ, all 1 unit low. Of 36,052 with only negative products, 3,757 differ. With one negative term, 9,753 of
53,447 differ.

### 3.2 Determinism and position (MEASURED, `out/analysis.txt`)

- Three runs of every class are bit-identical (both F16 instructions and all three iu8 variants).
- Moving every problem to the next subgroup slot: 0 of 4,194,304 outputs change.
- Permuting A rows and B columns (each value then sits in other lanes and registers): 0 of 4,194,304 outputs change.
- Permuting the 16 K slots: 26-39 % of outputs change (f32 accumulate) and 26-29 % (f16 accumulate), for every one
  of the 7 permutations tried (reverse, swap halves, even/odd, 4 random). The K order is part of the arithmetic.

INFERRED: there is no hidden per-lane or per-dispatch state. The result is a function of (A, B, C) with the K order fixed.

### 3.3 Model fits

MEASURED (`out/analysis.txt`). Mismatch = output bits differ (NaN equals NaN); ulp columns = ordered-bit distance (1 also counts +0 vs -0); rows sorted best first. Non-finite inputs are excluded.

#### e4_random (all 131072 problems): v_wmma_f32_16x16x16_f16, two headline models
| model | mismatches | of | ulp 0 | 1 | 2 | 3-4 | 5-16 | >16 |
|---|---|---|---|---|---|---|---|---|
| M-g8 | 17517128 | 33554432 | 16037304 | 12765931 | 3010956 | 889154 | 627530 | 223557 |
| M-exact | 17888390 | 33554432 | 15666042 | 13552491 | 2626748 | 854155 | 627332 | 227664 |

#### e4_random: v_wmma_f32_16x16x16_f16 (f32 accumulate), first 16384 problems
| model | mismatches | of | ulp 0 | 1 | 2 | 3-4 | 5-16 | >16 |
|---|---|---|---|---|---|---|---|---|
| M-g2 | 2007740 | 4194304 | 2186564 | 1364741 | 415579 | 120772 | 78723 | 27925 |
| M-seq | 2076663 | 4194304 | 2117641 | 1394512 | 444200 | 129894 | 79918 | 28139 |
| M-g4 | 2106948 | 4194304 | 2087356 | 1477910 | 403323 | 119155 | 78674 | 27886 |
| M-g8 | 2189346 | 4194304 | 2004958 | 1596118 | 375627 | 111393 | 78250 | 27958 |
| M-g8-pre | 2205432 | 4194304 | 1988872 | 1579075 | 403886 | 114841 | 79321 | 28309 |
| M-exact | 2235344 | 4194304 | 1958960 | 1693854 | 327769 | 106838 | 78352 | 28531 |
| M-g8-rev | 2237128 | 4194304 | 1957176 | 1646367 | 371792 | 111327 | 79024 | 28618 |
| M-g4-then-C | 2237577 | 4194304 | 1956727 | 1614865 | 394325 | 117490 | 81713 | 29184 |
| M-g8-evenodd | 2240399 | 4194304 | 1953905 | 1651789 | 369140 | 111987 | 79034 | 28449 |
| M-sum16-then-C | 2257148 | 4194304 | 1937156 | 1693048 | 348678 | 108300 | 78611 | 28511 |
| M-g8-then-C | 2263174 | 4194304 | 1931130 | 1656976 | 382826 | 114068 | 80313 | 28991 |
| M-tree | 2273931 | 4194304 | 1920373 | 1648528 | 393997 | 119352 | 82454 | 29600 |
| M-seq-rev | 2285983 | 4194304 | 1908321 | 1591278 | 443799 | 137843 | 83611 | 29452 |

#### p1b: v_wmma_f32_16x16x16_f16 (f32 accumulate)
| model | mismatches | of | ulp 0 | 1 | 2 | 3-4 | 5-16 | >16 |
|---|---|---|---|---|---|---|---|---|
| M-g2 | 534610 | 2097152 | 1562542 | 377265 | 95996 | 30958 | 22295 | 8096 |
| M-seq | 545749 | 2097152 | 1551403 | 381685 | 100163 | 33180 | 22555 | 8166 |
| M-g4 | 550290 | 2097152 | 1546862 | 393551 | 95497 | 30882 | 22268 | 8092 |
| M-g8 | 564196 | 2097152 | 1532956 | 412634 | 92037 | 29185 | 22290 | 8050 |
| M-g8-pre | 565376 | 2097152 | 1531776 | 411411 | 94074 | 29476 | 22338 | 8077 |
| M-seq-rev | 570277 | 2097152 | 1526875 | 404391 | 100073 | 34584 | 22914 | 8315 |
| M-g8-rev | 571535 | 2097152 | 1525617 | 419783 | 91611 | 29723 | 22337 | 8081 |
| M-g8-evenodd | 572214 | 2097152 | 1524938 | 420792 | 91374 | 29564 | 22338 | 8146 |
| M-exact | 577299 | 2097152 | 1519853 | 434645 | 83210 | 29051 | 22163 | 8230 |
| M-g4-then-C | 577532 | 2097152 | 1519620 | 428997 | 88145 | 29703 | 22420 | 8267 |
| M-sum16-then-C | 578850 | 2097152 | 1518302 | 434260 | 85021 | 29126 | 22214 | 8229 |
| M-g8-then-C | 579280 | 2097152 | 1517872 | 431879 | 87370 | 29447 | 22331 | 8253 |
| M-tree | 580024 | 2097152 | 1517128 | 431330 | 88144 | 29746 | 22486 | 8318 |

#### adv: v_wmma_f32_16x16x16_f16 (f32 accumulate)
| model | mismatches | of | ulp 0 | 1 | 2 | 3-4 | 5-16 | >16 |
|---|---|---|---|---|---|---|---|---|
| M-g2 | 781858 | 2097152 | 1315294 | 257746 | 40948 | 18692 | 27211 | 437261 |
| M-seq | 787170 | 2097152 | 1309982 | 261891 | 42389 | 19110 | 27247 | 436533 |
| M-g4 | 792888 | 2097152 | 1304264 | 270116 | 40047 | 18619 | 27133 | 436973 |
| M-seq-rev | 795930 | 2097152 | 1301222 | 273566 | 41937 | 19738 | 27723 | 432966 |
| M-g8-pre | 810267 | 2097152 | 1286885 | 294535 | 40016 | 18903 | 27408 | 429405 |
| M-g8-rev | 814850 | 2097152 | 1282302 | 299423 | 38201 | 18670 | 27523 | 431033 |
| M-g8 | 815781 | 2097152 | 1281371 | 295233 | 38190 | 18455 | 27135 | 436768 |
| M-g8-evenodd | 830621 | 2097152 | 1266531 | 311038 | 37259 | 18232 | 27247 | 436845 |
| M-g4-then-C | 898544 | 2097152 | 1198608 | 381643 | 38297 | 18546 | 27677 | 432381 |
| M-g8-then-C | 900035 | 2097152 | 1197117 | 383735 | 37683 | 18537 | 27595 | 432485 |
| M-sum16-then-C | 900386 | 2097152 | 1196766 | 386005 | 36145 | 18240 | 27507 | 432489 |
| M-tree | 900588 | 2097152 | 1196564 | 383350 | 38215 | 18759 | 27668 | 432596 |
| M-exact | 902765 | 2097152 | 1194387 | 385965 | 34608 | 18014 | 27317 | 436861 |

#### kperm: v_wmma_f32_16x16x16_f16 (f32 accumulate), first 4096 problems
| model | mismatches | of | ulp 0 | 1 | 2 | 3-4 | 5-16 | >16 |
|---|---|---|---|---|---|---|---|---|
| M-g2 | 385668 | 1048576 | 662908 | 265415 | 75927 | 23010 | 15670 | 5646 |
| M-seq | 396832 | 1048576 | 651744 | 270074 | 80518 | 24713 | 15810 | 5717 |
| M-g4 | 402099 | 1048576 | 646477 | 283657 | 74367 | 22783 | 15644 | 5648 |
| M-g8 | 415811 | 1048576 | 632765 | 303199 | 69969 | 21408 | 15595 | 5640 |
| M-g8-pre | 418528 | 1048576 | 630048 | 301358 | 73884 | 21875 | 15663 | 5748 |
| M-g8-rev | 423272 | 1048576 | 625304 | 310748 | 69592 | 21412 | 15792 | 5728 |
| M-g8-evenodd | 423817 | 1048576 | 624759 | 311807 | 69073 | 21581 | 15659 | 5697 |
| M-exact | 424521 | 1048576 | 624055 | 320401 | 62047 | 20725 | 15629 | 5719 |
| M-g4-then-C | 425101 | 1048576 | 623475 | 309573 | 71428 | 22157 | 16039 | 5904 |
| M-sum16-then-C | 427750 | 1048576 | 620826 | 320417 | 65047 | 20876 | 15693 | 5717 |
| M-g8-then-C | 428779 | 1048576 | 619797 | 315749 | 69627 | 21683 | 15893 | 5827 |
| M-seq-rev | 428867 | 1048576 | 619709 | 300350 | 80251 | 25958 | 16448 | 5860 |
| M-tree | 430119 | 1048576 | 618457 | 314370 | 71310 | 22358 | 16129 | 5952 |

#### f16_full: v_wmma_f32_16x16x16_f16 (f32 accumulate), first 256 problems
| model | mismatches | of | ulp 0 | 1 | 2 | 3-4 | 5-16 | >16 |
|---|---|---|---|---|---|---|---|---|
| M-g2 | 41138 | 65536 | 24398 | 21251 | 9078 | 5013 | 4228 | 1568 |
| M-g4 | 43534 | 65536 | 22002 | 23545 | 9067 | 5140 | 4208 | 1574 |
| M-seq | 44114 | 65536 | 21422 | 21791 | 10334 | 5836 | 4508 | 1645 |
| M-g8 | 44137 | 65536 | 21399 | 24739 | 8708 | 4961 | 4158 | 1571 |
| M-exact | 44198 | 65536 | 21338 | 25272 | 8490 | 4782 | 4111 | 1543 |
| M-g8-pre | 44330 | 65536 | 21206 | 23925 | 9627 | 4997 | 4213 | 1568 |
| M-g8-evenodd | 44405 | 65536 | 21131 | 24778 | 8855 | 5031 | 4171 | 1570 |
| M-sum16-then-C | 44486 | 65536 | 21050 | 25137 | 8855 | 4834 | 4107 | 1553 |
| M-g8-rev | 44749 | 65536 | 20787 | 25146 | 8972 | 4889 | 4158 | 1584 |
| M-g8-then-C | 44920 | 65536 | 20616 | 24504 | 9549 | 5070 | 4219 | 1578 |
| M-g4-then-C | 45028 | 65536 | 20508 | 23936 | 9797 | 5263 | 4420 | 1612 |
| M-tree | 45668 | 65536 | 19868 | 24203 | 10006 | 5360 | 4431 | 1668 |
| M-seq-rev | 47536 | 65536 | 18000 | 23536 | 11518 | 6135 | 4648 | 1699 |

#### e4_random: v_wmma_f16_16x16x16_f16 (f16 accumulate, C = RN16(C)), first 16384 problems
| model | mismatches | of | ulp 0 | 1 | 2 | 3-4 | 5-16 | >16 |
|---|---|---|---|---|---|---|---|---|
| M-g2 | 193815 | 3555814 | 3361999 | 159342 | 21328 | 6913 | 4694 | 1538 |
| M-g4 | 1100627 | 3555814 | 2455187 | 874873 | 131613 | 46635 | 31959 | 15547 |
| M-seq | 1184432 | 3555814 | 2371382 | 894477 | 180861 | 59934 | 36666 | 12494 |
| M-g8 | 1312715 | 3555814 | 2243099 | 1057417 | 133655 | 53642 | 40493 | 27508 |
| M-exact | 1367524 | 3555814 | 2188290 | 1106473 | 122754 | 52481 | 42480 | 43336 |
| M-sum16-then-C | 1417221 | 3555814 | 2138593 | 1144589 | 131778 | 54470 | 43396 | 42988 |
| M-tree | 1584850 | 3555814 | 1970964 | 1228282 | 201612 | 75858 | 55383 | 23715 |

#### p1b: v_wmma_f16_16x16x16_f16 (f16 accumulate, C = RN16(C))
| model | mismatches | of | ulp 0 | 1 | 2 | 3-4 | 5-16 | >16 |
|---|---|---|---|---|---|---|---|---|
| M-g2 | 79571 | 1735493 | 1655922 | 64387 | 9107 | 3136 | 2176 | 765 |
| M-g4 | 215232 | 1735493 | 1520261 | 163436 | 25186 | 9147 | 6182 | 11281 |
| M-seq | 225397 | 1735493 | 1510096 | 169246 | 34216 | 12239 | 7243 | 2453 |
| M-g8 | 270575 | 1735493 | 1464918 | 204133 | 25913 | 10447 | 7856 | 22226 |
| M-exact | 312200 | 1735493 | 1423293 | 233995 | 24755 | 10889 | 8962 | 33599 |
| M-sum16-then-C | 325324 | 1735493 | 1410169 | 244711 | 26765 | 11346 | 9176 | 33326 |
| M-tree | 344588 | 1735493 | 1390905 | 263749 | 36624 | 14821 | 11111 | 18283 |

#### adv: v_wmma_f16_16x16x16_f16 (f16 accumulate, C = RN16(C))
| model | mismatches | of | ulp 0 | 1 | 2 | 3-4 | 5-16 | >16 |
|---|---|---|---|---|---|---|---|---|
| M-g2 | 318413 | 1121114 | 802701 | 33674 | 11329 | 13564 | 41945 | 217901 |
| M-seq | 489660 | 1121114 | 631454 | 87421 | 25735 | 21907 | 55634 | 298963 |
| M-g4 | 498062 | 1121114 | 623052 | 88758 | 22703 | 22243 | 58075 | 306283 |
| M-g8 | 556646 | 1121114 | 564468 | 105630 | 22780 | 23438 | 61636 | 343162 |
| M-exact | 580674 | 1121114 | 540440 | 106227 | 21321 | 22087 | 59799 | 371240 |
| M-sum16-then-C | 583501 | 1121114 | 537613 | 106394 | 17021 | 12424 | 23838 | 423824 |
| M-tree | 591290 | 1121114 | 529824 | 123849 | 22942 | 14423 | 24042 | 406034 |

#### f16_full: v_wmma_f16_16x16x16_f16 (f16 accumulate, C = RN16(C)), first 256 problems
| model | mismatches | of | ulp 0 | 1 | 2 | 3-4 | 5-16 | >16 |
|---|---|---|---|---|---|---|---|---|
| M-g2 | 38 | 63000 | 62962 | 28 | 7 | 1 | 1 | 1 |
| M-g4 | 28664 | 63000 | 34336 | 21147 | 4565 | 1602 | 1070 | 280 |
| M-g8 | 30547 | 63000 | 32453 | 23493 | 4131 | 1521 | 1094 | 308 |
| M-exact | 30785 | 63000 | 32215 | 24164 | 3729 | 1466 | 1125 | 301 |
| M-sum16-then-C | 32048 | 63000 | 30952 | 25040 | 3982 | 1553 | 1155 | 318 |
| M-seq | 34363 | 63000 | 28637 | 21289 | 7556 | 3297 | 1737 | 484 |
| M-tree | 39151 | 63000 | 23849 | 25212 | 7769 | 3448 | 2111 | 611 |

Model definitions (`models.py`; R = round to nearest even in the result type, products exact):
M-exact = R(C + exact sum); M-seq / M-seq-rev = R after each product in K order / reversed, starting from C;
M-g8 = R(R(C + sum k0-7) + sum k8-15) (the repo's best fit for RDNA 4 FP8); M-g8-rev = the 8..15 group first;
M-g8-evenodd = groups of even and odd k; M-g4, M-g2 = groups of 4 or 2 in K order; M-tree = pairwise tree with R at
every node, C last; M-sum16-then-C = R(R(sum) + C); M-g8-pre = each group's sum rounded alone, then added to C in turn;
M-g8-then-C, M-g4-then-C = rounded group sums added together, C last.
Also tried in exploration (`explore3.py`, `explore6.py`, p1b): align every term to the largest exponent and floor or
truncate it at 23-28 bits, with C inside or outside the alignment and a final RN, RZ or floor; and one's complement
negation (`-floor(|t|/u)u - u`), optionally with -0 products counted as negative, in each of those combinations.
Best on p1b C = 0 outputs: floor at 26 bits with final RN, 51,737 of 261,893 still mismatching. **None fits.**

### 3.4 Error size (MEASURED)

Error is |D - exact| in units of 2^(E-24), where E is the exponent of the largest of the 17 terms. The "differs from exact" counts below
include outputs where RN32 itself cannot be exact.

- e4_random: max |err| = 23.072 units; outputs with err != 0: all terms same sign 90 of 101, mixed signs 3976076 of 4194203
- p1b: max |err| = 13.942 units; outputs with err != 0: all terms same sign 130441 of 612997, mixed signs 1023260 of 1484155
- adv: max |err| = 21.469 units; outputs with err != 0: all terms same sign 179715 of 336310, mixed signs 1396819 of 1760842
- f16_full: max |err| = 108.871 units; outputs with err != 0: all terms same sign 16 of 16, mixed signs 1037544 of 1048560

On `p1b`, 87 % of C = 0 outputs are within 1 unit and 99.8 % within 4 units (`explore8.py`). The errors are bounded
relative to the largest term (never above 1e-4 of it), so this is a precision loss, not wrong data (MEASURED).

### 3.5 Signed zero, NaN and inf, subnormals, integers (MEASURED)

#### Signed zero (adv sub-kind 7: C = +-0, products +-0 or nonzero)
- every term -0 (C = -0, all products -0): 0 outputs -> +0: 0, -0: 0, nonzero: 0
- all terms zero, some +0: 130721 outputs -> +0: 130721, -0: 0, nonzero: 0
- nonzero products cancelling to an exact 0: 17 outputs -> +0: 1, -0: 0, nonzero: 16 (values seen: [-0.0001220703125, -7.62939453125e-06, -2.384185791015625e-07, -1.1920928955078125e-07, -5.960464477539063e-08, -2.9802322387695312e-08])
- f16 accumulate, every term -0: +0 0, -0 0

#### Specials (C = +-inf, NaN, NaN operands)
- C = +-inf (30% of outputs): outputs where expected NaN: 0, GPU NaN there: 0; expected +-inf: 39393, GPU same inf: 39393; GPU NaN elsewhere: 0
- C = NaN (30%): outputs where expected NaN: 39204, GPU NaN there: 39204; expected +-inf: 0, GPU same inf: 0; GPU NaN elsewhere: 0
- one NaN code in A: outputs where expected NaN: 8192, GPU NaN there: 8192; expected +-inf: 0, GPU same inf: 0; GPU NaN elsewhere: 0
- C = +-inf everywhere: outputs where expected NaN: 0, GPU NaN there: 0; expected +-inf: 131072, GPU same inf: 131072; GPU NaN elsewhere: 0

#### FP16 subnormal operands (f16_full sub-kinds 2 and 3)
- A subnormal, B normal: outputs 262144; |D - exact| / |exact| median 1.13e-07, max 4.84e-03; D equal to the flush-to-zero result where that differs from exact: 0 of 262144
- A, B in the lowest binades (subnormals), C = 0: outputs 262144; |D - exact| / |exact| median 6.52e-08, max 1.28e-02; D equal to the flush-to-zero result where that differs from exact: 0 of 262044

#### Integer (iu8) instruction
- int_s8 wrap: mismatches vs exact (wrapping) = 0 of 1048576; outputs that overflow int32: 2200
- int_s8 sat: mismatches vs exact (saturating) = 2241 of 1048576; outputs that overflow int32: 2200
- int_u8 wrap: mismatches vs exact (wrapping) = 0 of 1048576; outputs that overflow int32: 65114

## 4. What this means for the F16-emulation idea (INFERRED)

1. RDNA 3's F16 WMMA cannot match an "exact sum, one rounding" reference, and it cannot match any simple f32 grouping
   model either. Bit-identity with RDNA 4's `v_wmma_f32_16x16x16_fp8_fp8` holds only if RDNA 4 shows exactly the same
   mixed-sign artefacts with the same K-slot dependence. The repo's RDNA 4 fit (M-g8 with 2 of 51 unexplained) does
   not suggest that, but it was fitted on few samples. Replay these inputs on RDNA 4 (section 5) to settle it.
2. The errors are small: a few units at 2^(E-24), within 4 units for over 99 % of outputs. After `q8` that rarely
   changes an FP8 code. The equivalence verdicts (CLOSE / MATCHES) may still hold; byte identity will not.
3. Because results depend on the K order, a pack-time K permutation cannot fix the mismatch as long as no model
   explains the arithmetic. Only a measured RDNA 4 bit-for-bit match would tell which order to use.
4. The iu8 instruction is the exact alternative (option e in lane 3). Wrapping results equal the exact sum mod 2^32
   on every output tested (3.5). The 41 saturating mismatches beyond the 2,200 overflowing outputs suggest that
   saturation applies to an intermediate partial sum, not only to the final one (INFERRED; not investigated).
5. For Model K / FG (real FP16 weights, FP16 accumulate), M-g2 is a near-fit. Compare RDNA 4's F16 WMMA on the same
   `f16_full` inputs before concluding anything (INFERRED).

## 5. Replaying these inputs on RDNA 4

Files: `results/<class>.npz` (inputs plus the GPU outputs of this card). Keys:

| Key | Shape, type | Meaning |
|---|---|---|
| `A` | [N,16,16] float16 | A of problem p: row m, column k (row-major, M x K) |
| `B` | [N,16,16] float16 | B: row k, column n (row-major, K x N) |
| `C` | [N,16,16] float32 | accumulator, row m, column n |
| `C16` | [N,16,16] float16 | RN16(C), used for the F16-accumulate runs |
| `A8`, `B8` | [N,16,16] uint8 | E4M3 codes of A and B (E4M3 classes only; `A == E4M3(A8)` exactly) |
| `kind` | [N] int8 | sub-kind (P1b mode, adversarial sub-kind, ...) |
| `D_f32` | [N,16,16] float32 | this card: D = A @ B + C via `v_wmma_f32_16x16x16_f16` (run 0; `D_f32_3runs_identical`) |
| `D_f16` | [N,16,16] float16 | this card: `v_wmma_f16_16x16x16_f16` on `C16` |
| `perms` (kperm), `mperm` / `nperm` (pos_perm) | | the permutations used |
| `int_*.npz`: `A`, `B` int8/uint8, `C` int32/uint32, `D_wrap`, `D_sat` | | iu8 runs |

Mathematically D[p, m, n] = C[p, m, n] + sum_k A[p, m, k] * B[p, k, n], with k in the order stored. For gfx12's FP8
instruction, pack `A8` and `B8` into the gfx12 wave32 fragments: k = 0..7 and 8..15 split across the half-waves as in
`clean/rr_f/gpu/probe_p1.py` / `rr_probe.hlsl`; WrongGPU's own layout is the gfx12 one. Keep the stored K order. Then
compare D bit for bit with `D_f32`, and also run `models.py` against the RDNA 4 output (`analyse.py`'s `fit` takes any D
array). The FP16 classes (`f16_full`) replay through gfx12's `v_wmma_f32_16x16x16_f16` for Model K / FG. Raw
little-endian `.bin` files with the same layout are in `inputs/` for a C harness.

## 6. Reproduce

`./run.sh` (about a minute of GPU time, a few seconds per run; the analysis takes a few minutes of CPU). It builds the
harness and shaders, generates the inputs if missing, runs every class 3 times, captures the disassembly, runs the
two-term probe, packs `results/`, and writes `out/analysis.txt`. Disk: about 2.5 GB (`inputs/`, `out/`, `results/`).
The run removes DISPLAY and WAYLAND_DISPLAY, uses only the RADV ICD and disables the shader cache.

## 7. Limits and open points

- The full model table on `e4_random` covers the first 16384 problems; all 131072 problems are fitted only against
  M-exact and M-g8. The FP16 class is fitted on its first 256 problems (the slow exact-integer backend).
- No model explains the arithmetic. The likely next step (INFERRED) is a structured term probe like 3.1:
  vary one negative term's K slot, exponent gap and count, to rebuild the adder (pair adders, end-around carry,
  where C enters).
- Signed zero: the generator never produced an output whose 17 terms are all -0 (the rows report 0 such outputs),
  so IEEE's "-0 only when every term is -0" rule is untested. Every all-zero sum that includes a +0 term returned +0.
  Of 17 exact cancellations to 0 with nonzero products, 16 returned a small negative value (for example
  -1.19e-07), consistent with 3.1.
- `explore*.py` are the exploration scripts behind 3.1 and 3.3 (C = 0 subsets, alignment and truncation variants).
- RADV sets the shader float mode (RNE, denormals). Whether another driver or mode register setting changes the
  WMMA behaviour is not tested. INFERRED: the WMMA has no rounding-mode operand, so it probably does not.
