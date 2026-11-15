# Lane 9: what the documentation says about RDNA 3 throughput (RX 7900 XT, Navi 31, gfx1100)

Scope: the documented expectation, not a measurement. Part I covers the matrix instructions (`v_wmma_*`), Part II
the ordinary vector instructions, Part III what AMD says about using them for neural networks. The last section is
a one-page table to check measurements against. Lane 4 (`findings/4-isa-driver.md`) already covers the instruction
lists, opcodes, register layouts and the driver channel. This lane repeats those only where a number depends on them.

Sources were fetched on 2026-11-14:

| short name | what | URL |
|---|---|---|
| ISA3 | "RDNA3" Instruction Set Architecture Reference Guide (AMD, 600 pages, PDF built Aug 2023) | https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0 |
| ISA4 | "RDNA4" Instruction Set Architecture Reference Guide (AMD, PDF built Mar 2025), for comparison only | https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture |
| GO3 | GPUOpen, "How to accelerate AI applications on RDNA 3 using WMMA" (2023) | https://gpuopen.com/learn/wmma_on_rdna3/ |
| GO4 | GPUOpen, "Using the Matrix Cores of AMD RDNA 4 architecture GPUs" (July 2025) | https://gpuopen.com/learn/using_matrix_core_amd_rdna4/ |
| GOocc | GPUOpen, "Occupancy explained" | https://gpuopen.com/learn/occupancy-explained/ |
| CALC | ROCm amd_matrix_instruction_calculator, commit 94b1f16 (2026-11-11), run on the CPU | https://github.com/ROCm/amd_matrix_instruction_calculator |
| LLVM | llvm-project `main`, AMDGPU backend | https://github.com/llvm/llvm-project/tree/main/llvm/lib/Target/AMDGPU |
| ACO | Mesa `main`, commit c9ce420 (2026-11-14), `src/amd/compiler` | https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/amd/compiler |
| SPEC-XT | AMD product page, RX 7900 XT | https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xt.html |
| SPEC-9070 | AMD product page, RX 9070 XT | https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html |
| ROCPROF | ROCm Compute Profiler, RDNA 3.5 "System Speed-of-Light" | https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/rdna/system-speed-of-light.html |
| DECK | AMD, "RDNA 3: Beyond the current gen" (Reboot Develop 2023) | https://gpuopen.com/presentations/2023/RDNA3_Beyond-the-current-gen-v4.pdf |
| C&C | Chips and Cheese, "Microbenchmarking AMD's RDNA 3 Graphics Architecture" (third-party analysis) | https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture |
| TH | Tom's Hardware, RDNA 3 architecture deep dive (launch-deck coverage) | https://www.tomshardware.com/news/amd-rdna-3-gpu-architecture-deep-dive-the-ryzen-moment-for-gpus |

Labels: READ (seen in a document or source, cited), MEASURED-IN-REPO (the repo records a measurement), INFERRED
(my arithmetic or reasoning, shown), UNKNOWN (no source; the test that settles it is named).

Units used throughout: **MAC** = one multiply plus one add. AMD's "FLOPS" and "TOPS" count a MAC as 2 operations.
"Per CU per clock" means per compute unit per shader clock. A WGP (work-group processor) is 2 CUs and 4 SIMDs.

## Summary

1. **Matrix (Part I).** On gfx11 every 16x16x16 WMMA (4096 MAC) takes **32 cycles** per SIMD for F16, BF16 and IU8,
   and **16 cycles** for IU4 (CALC; ACO's cost table agrees). That is 128 MAC/SIMD/clk = **256 MAC/CU/clk** for
   F16, BF16 and IU8, and 512 MAC/CU/clk for IU4. GO3's table gives the same as 512 "FLOPS/clock/CU" for the
   RX 7900 XTX. A WMMA cannot run alongside other vector instructions (CALC: "Can co-execute with VALU: False").
2. **Peak, RX 7900 XT:** MAC/s = 84 CU x 256 x f. At AMD's boost clock (f = 2.4 GHz) that is **51.6 T MAC/s =
   103 TFLOPS F16 = 103 TOPS INT8**, exactly AMD's spec page. At the 2.0 GHz "game frequency" it is 43.0 T MAC/s.
   An RX 9070 XT's FP8 WMMA does 1024 MAC/CU/clk: 64 x 1024 x 2.97 GHz = 194.6 T MAC/s (389 TFLOPS, AMD's figure).
   That is **3.8x** the 7900 XT's F16 or INT8 matrix peak at boost clocks, and 4x per CU per clock.
3. **Shader (Part II).** The dot-product instructions reach the **same documented peak as WMMA** for INT8
   (`v_dot4_i32_iu8`: 32 lanes x 4 MAC per SIMD per clock = 256 MAC/CU/clk) and IU4. F16 WMMA is 2x
   `v_dot2_f32_f16` issued one at a time, and **1x when the dot2 is dual-issued**, either paired as
   `v_dual_dot2acc_f32_f16` in wave32 or run as wave64. ISA3 itself says WMMA instructions "internally use the DOT
   instructions". So on RDNA 3, WMMA saves registers and data movement. It does not add arithmetic rate.
4. **"Dual issue"** covers two things: (a) the VOPD encoding, wave32 only, with a fixed list of 13/18 opcodes. The
   list has FP32 FMA/MUL/ADD, MOV, CNDMASK, MIN/MAX and `DOT2ACC_F32_F16/BF16`, but **no packed FP16 math, no
   int8 dot product, no conversions**. (b) Running a wave64 instruction over one cycle instead of two, for a
   similar list (ACO's `is_dual_issue_capable`). `v_dot4_i32_iu8`, conversions and lane permutes are in neither.
5. **Precision statements (ISA3 7.9):** "WMMA supports only round-to-nearest-even rounding for float types", "WMMA
   does not generate any ALU exceptions", and the instructions "internally use the DOT instructions". Nothing is
   said about intermediate precision, accumulation order or denormals. LLVM marks gfx11 WMMA `ReadsModeReg = 0`,
   so the compiler assumes the round/denormal MODE bits do not apply. ISA4 instead says its WMMA adds "using fused
   multiply add". No document claims a single rounding for RDNA 3 WMMA. The measured inexact mixed-sign sums
   contradict no document.
6. **Lane exchange:** gfx11 **does have** `v_permlane16_b32` and `v_permlanex16_b32` (VOP3, since gfx10; ISA3
   opcodes 603/604). gfx12 only adds the `_var` forms. `v_permlanex16_b32` with identity selects swaps lanes L and
   L^16 in one vector instruction. `ds_swizzle_b32` documents the same swap as "SWAPX16" through the LDS path.
   `v_permlane64_b32` swaps the halves of a wave64.
7. **Occupancy:** 1536 VGPRs per SIMD in wave32 (768 wave64 registers), allocated in blocks of 24 (wave32) or 12
   (wave64), at most 16 waves per SIMD. Waves per SIMD = min(16, floor(1536 / roundUp(VGPRs, 24))) in wave32:
   128 VGPRs -> 10 waves, 256 -> 5. LDS: 128 KB per WGP, at most 64 KB per work-group.

## Part I: matrix throughput (v_wmma_*)

### I.1 What each document states about WMMA timing

| statement | source | label |
|---|---|---|
| "A single WMMA instruction coordinates 32 clocks of optimal work scheduling." | GO3 | READ |
| "These instructions work over multiple cycles to compute the result matrix and internally use the DOT instructions." | ISA3 section 7.9, p.73 | READ |
| `'cycles': 32` for f16/bf16/iu8 entries, `16` for iu4; `'coexec': False, 'coexec_delay': -1` for all rdna3 entries | CALC `matrix_calculator.py` rdna3 dict (from line 2301) | READ |
| Ops per WGP per cycle = (2*M*N*K / cycles) x 4 (4 SIMDs per WGP) | CALC `matrix_calculator.py:5240-5245` | READ |
| gfx11 WMMA cost: `cost = v_wmma_i32_16x16x16_iu4 ? 16 : 32`, comment "int8 and (b)f16 have the same performance"; latency `4 + cost`; VALU resource busy for `cost` cycles. gfx12: f16/bf16 16, others 8 | ACO `aco_statistics.cpp:198-217` | READ |
| WMMA is not a "vector" class in ACO's wave64 two-pass model, so wave64 WMMA is costed as one 32-cycle pass | ACO `aco_statistics.cpp:397-415, 426-429` | READ |
| LLVM's gfx11 scheduling model has **no WMMA entry**. The gfx11 WMMA pseudos are scheduled as `Write32Bit` (latency 5, like an ordinary vector op) | LLVM `VOP3PInstructions.td:1720-1730`, `SISchedule.td:405-441` | READ |
| RX 9070 XT measured 7.7 (FP8) and 15.5 (F16) cycles per WMMA per SIMD, against CALC's documented 8 and 16 | `clean/k/kernels/k_block8.h:7-8` | MEASURED-IN-REPO |

The repo's RDNA 4 measurement matches CALC to within 4 %. That is the best evidence that CALC's cycle numbers
describe real hardware, not just a model. INFERRED.

### I.2 Per-instruction table (gfx11, from CALC output, cross-checked with GO3 and ACO)

CALC run: `matrix_calculator.py --architecture rdna3 --instruction <name> --wavefront {32,64} --detail-instruction`.

| instruction | MAC per instr. | cycles (w32 and w64) | MAC/SIMD/clk | MAC/CU/clk | ops/CU/clk (AMD's unit) | A, B VGPRs (w32 / w64) | C, D VGPRs w32 | C, D VGPRs w64 |
|---|---|---|---|---|---|---|---|---|
| v_wmma_f32_16x16x16_f16 | 4096 | 32 | 128 | 256 | 512 | 8 / 8 | 8 (F32) | 4 |
| v_wmma_f32_16x16x16_bf16 | 4096 | 32 | 128 | 256 | 512 | 8 / 8 | 8 | 4 |
| v_wmma_f16_16x16x16_f16 | 4096 | 32 | 128 | 256 | 512 | 8 / 8 | 8 (unpacked, OPSEL half) | 4 |
| v_wmma_bf16_16x16x16_bf16 | 4096 | 32 | 128 | 256 | 512 | 8 / 8 | 8 (unpacked) | 4 |
| v_wmma_i32_16x16x16_iu8 | 4096 | 32 | 128 | 256 | 512 | 4 / 4 | 8 (I32) | 4 |
| v_wmma_i32_16x16x16_iu4 | 4096 | 16 | 256 | 512 | 1024 | 2 / 2 | 8 | 4 |

READ (CALC; per-CU = CALC's per-WGP figure / 2). GO3's table for the RX 7900 XTX gives the same per-CU values
(FP16 512, BF16 512, IU8 512, IU4 1024 "FLOPS/clock/CU"). GO4 repeats FP16 512, BF16 512, I8 512 for the 7900 XTX.

Wave64: CALC has a single `cycles` field per instruction, so it prints 32 for both wave sizes, and ACO also costs a
wave64 WMMA as one 32-cycle pass. A wave64 WMMA does the same 16x16x16 product (GO3: "the number of threads that
will participate in the 16x16x16 GEMM operation"). So the documented expectation is **the same 32 cycles per
instruction in wave64, and half the MAC per lane-slot**. No AMD text states the wave64 cycle count directly:
UNKNOWN until measured (see the measurement list).

Results per cycle per CU, in matrix elements: one F16 WMMA writes 256 results after 32 cycles, so 2 SIMDs x 256 /
32 = **16 result elements per CU per clock** (8 for IU4 per 16 cycles: also 2 x 256 / 16 = 32). INFERRED.

### I.3 Restrictions, hazards and NOPs (gfx11)

| rule | source | label |
|---|---|---|
| A and B "lanes 0-15 data are replicated into lanes 16-31 (for wave64: also into lanes 32-47 and 48-63)" "In order to achieve this performance" | ISA3 7.9 p.73; GO3 | READ |
| SRC0 (A) and SRC1 (B) "WMMA: must be a VGPR"; inline constants "can only be used for C-matrix" (F16/BF16 inline replicated to both halves) | ISA3 7.5 p.66, 7.9 p.73 | READ |
| EXEC is forced to all ones for the instruction (`saved_exec = EXEC; EXEC = 64'B(-1); ...; EXEC = saved_exec`) | ISA3 16.10 p.358-359 | READ |
| "Back-to-back dependent WMMA instructions require one V_NOP (or independent VALU op) between them if the first instruction's matrix D is the same or overlaps with the second instruction's matrices A or B. Matrix A/B can overlap C as long as C is distinct from D. The typical case is that C and D are the same." | ISA3 7.9 p.73 | READ |
| LLVM inserts one `V_NOP_e32` when a WMMA's src0/src1 overlaps the previous WMMA's vdst with no VALU in between | LLVM `GCNHazardRecognizer.cpp:2658-2706` (`fixWMMAHazards`) | READ |
| Mesa ACO inserts `v_nop` under the same condition, and clears the tracking on any other VALU | ACO `aco_insert_NOPs.cpp:1761-1778` | READ |
| The "wmma-coexecution-hazards" feature ("Hazards when WMMA co-executes with other WMMA/VALU instruction") is set only for gfx125x / gfx13 feature sets, not for gfx11 | LLVM `GCNFeatures.td:1323-1325` and its uses at lines 2621-2734 | READ |
| No co-execution of WMMA with VALU on gfx11 (or gfx12) | CALC (`coexec: False`) | READ |
| NEG[1:0] select signed/unsigned for IU8/IU4 ("NEG[2] and NEG_HI[2:0] must be zero"); for F16/BF16 NEG/NEG_HI negate A/B halves, {NEG_HI[2], NEG[2]} = {ABS, NEG} on C | ISA3 7.9 p.73 | READ |
| OPSEL[2] chooses the low or high 16 bits for C and D of the 16-bit-output forms; OPSEL[1:0] unused | ISA3 7.5 p.66; GO3 | READ |
| VGPR alignment: 4 bytes (no even/quad alignment rule) | CALC "GPR alignment requirement: 4 bytes" | READ |
| WMMA cannot use DPP | ISA3 7.7 table 30, p.70 | READ |
| Wave sizes: both wave32 and wave64 have WMMA forms (`_w32`, `_w64` builtins) | GO3 | READ |

INFERRED from the rows above: a chain of WMMAs that only reuses the accumulator (C = D) needs no NOP. Feeding a
result straight back as A or B costs one NOP, but the format conversion and lane exchange that such a step needs
anyway (Part III) already fill that slot.

### I.4 Every precision, rounding and denormal sentence found for WMMA

ISA3 (RDNA 3), quoted in full:
- "WMMA supports only round-to-nearest-even rounding for float types." (7.9, p.73)
- "WMMA does not generate any ALU exceptions." (7.9, p.73)
- "These instructions work over multiple cycles to compute the result matrix and internally use the DOT
  instructions." (7.9, p.73)
- Per-instruction pseudocode is only `eval "D0.f32(16x16) = S0.f16(16x16) * S1.f16(16x16) + S2.f32(16x16)"`
  (16.10, p.358). It does not specify order, intermediate width, or rounding points.
- The DOT instruction the sentence points at is specified as sequential F32 adds: `tmp = 32'F(S0[15:0].f16) *
  32'F(S1[15:0].f16); tmp += 32'F(S0[31:16].f16) * 32'F(S1[31:16].f16); tmp += S2.f; D0.f = tmp` (V_DOT2_F32_F16,
  p.354). The same section says only: "DOT2_F16_F16 and DOT2_BF16_BF16 support round-to-nearest-even rounding.
  DOT2_F16_F16 supports denorms, and DOT2_BF16_BF16 disables all denorms." (7.2.4, p.61). Nothing about
  V_DOT2_F32_F16's intermediate rounding.
- MODE register: "Denorm mode affects float ops in: VALU, LDS, and VMEM atomics." WMMA is not named (3.4.2, p.22). "16-bit floats
  support denormals, infinity and NaN." (7.2.4, p.61)

Other sources:
- LLVM declares the gfx11 WMMA pseudos with `mayRaiseFPException = 0, ReadsModeReg = 0`
  (`VOP3PInstructions.td:1721`). READ. INFERRED: LLVM assumes WMMA ignores FP_ROUND and FP_DENORM, which matches
  "only round-to-nearest-even" and "no ALU exceptions".
- CALC labels the operands "FP16 (IEEE binary16 floating point)" and "FP32 (IEEE binary32 floating point)". That
  describes the storage format, not the arithmetic. READ.
- GO3 and GO4 contain no sentence about precision, rounding, denormals or IEEE compliance. READ (searched the full
  article text).
- ISA4 (RDNA 4) drops the RNE and "internally use the DOT instructions" sentences and says instead "add the 16x16
  matrix in the third input using fused multiply add ... the row-column dot products are distributed across the
  vector ALU" (V_WMMA_F32_16X16X16_FP8_FP8 description). READ.
- No source found says "may not be IEEE compliant" about WMMA on either generation. READ (searched ISA3, ISA4, GO3,
  GO4, CALC README).

INFERRED: "internally use the DOT instructions" together with sequential DOT pseudocode allows a 16-term dot
product computed as a chain of shorter dot products, each rounded to F32 (or to some internal width). That would
give exactly the measured symptom, slightly inexact sums when terms of opposite sign cancel. The documents do not
say how K = 16 is split, so the split is UNKNOWN. Settle it with a probe that puts one large positive and one
large negative product at chosen k positions and moves them around. The positions where exactness breaks show the
grouping (lane 3 numerics T2 uses the same idea).

### I.5 Documented peak, RX 7900 XT, and RX 9070 XT FP8 for comparison

Official specs (SPEC-XT): 84 compute units, 5376 stream processors, "AI Accelerators 168", boost "Up to 2400 MHz",
game frequency 2000 MHz. The page lists "Peak Half Precision (FP16 Matrix) 103 TFLOPs", "Peak 8-bit Precision
(INT8 Matrix) 103 TOPs", "Peak 4-bit Precision (INT4 Matrix) 206 TOPs", "Peak Single Precision (FP32 Vector) 51.6
TFLOPs" and "Peak Half Precision (FP16 Vector) 51.6 TFLOPs". AMD gives no base clock and no footnote with the
formula. READ.

Formula (INFERRED, reproduces every AMD figure to 3 digits):

```
MAC/s (7900 XT, F16/BF16/IU8 WMMA) = N_CU x 256 MAC/CU/clk x f = 84 x 256 x f
  f = 2.4 GHz (boost):  51.6e12 MAC/s = 103.2 TFLOPS / TOPS   (AMD: 103)
  f = 2.0 GHz (game):   43.0e12 MAC/s =  86.0 TFLOPS / TOPS
MAC/s (7900 XT, IU4 WMMA) = 84 x 512 x f  -> 103.2e12 MAC/s at 2.4 GHz = 206 TOPS (AMD: 206)
WMMA instructions/s (whole GPU) = 168 SIMDs x f / 32  -> 12.6e9 /s at 2.4 GHz
```

"AI Accelerators 168" = 84 CU x 2 = one per SIMD. INFERRED from the count, which matches the SIMD count.

RX 9070 XT (SPEC-9070): 64 CUs, boost "Up to 2970 MHz", game 2400 MHz; FP8 matrix 389 TFLOPs (779 with structured
sparsity), INT8 389 TOPs, FP16 matrix 195 TFLOPs, FP16/FP32 vector 48.7 TFLOPs. READ. CALC gives gfx12
`v_wmma_f32_16x16x16_fp8_fp8` 8 cycles, 4096 FLOPs/WGP/cycle = 1024 MAC/CU/clk. READ.

```
MAC/s (9070 XT, FP8 WMMA) = 64 x 1024 x f -> 194.6e12 MAC/s at 2.97 GHz (389 TFLOPS, AMD: 389); 157.3e12 at 2.4 GHz
MAC/s (9070 XT, F16 WMMA) = 64 x  512 x f ->  97.3e12 MAC/s at 2.97 GHz (195 TFLOPS, AMD: 195)
```

Ratios (INFERRED): per CU per clock, 9070 XT FP8 / 7900 XT F16 = 1024 / 256 = **4.0**. Whole card at boost clocks
= 194.6 / 51.6 = **3.77**; at game clocks = 157.3 / 43.0 = 3.66. For an FP8 network moved to F16 (or IU8) WMMA on
the 7900 XT, the documented matrix ceiling is about 3.7-3.8x lower than the 9070 XT's, before the extra lane
exchanges of Part III.

## Part II: shader throughput (ordinary vector instructions)

### II.1 Issue model

| fact | source | label |
|---|---|---|
| 4 SIMDs per WGP, 2 per CU; SIMD datapath 32 wide; wave32 native, wave64 supported | GOocc ("SIMD datapaths in AMD RDNA-based GPUs are 32-wide"; "6 SEs x 8 WGPs x 4 SIMDs = 192 SIMDs" for the 7900 XTX) | READ |
| Wave64 executes as two passes ("both passes of a wave64 use the state of the wave prior to instruction execution") | ISA3 2.1, p.9 | READ |
| VOPD: "a single shader instruction to encode two separate VALU operations that are executed in parallel ... legal only for wave32 ... skipped for wave64" | ISA3 7.6, p.67-68 | READ |
| VOPD X opcodes: FMAC_F32, FMAAK_F32, FMAMK_F32, MUL_F32, ADD_F32, SUB_F32, SUBREV_F32, MUL_DX9_ZERO_F32, MOV_B32, CNDMASK_B32, MAX_F32, MIN_F32, DOT2ACC_F32_F16, DOT2ACC_F32_BF16. Y adds ADD_NC_U32, LSHLREV_B32, AND_B32 | ISA3 tables 91-92, p.168-169 | READ |
| VOPD constraints: each half at most 2 VGPR sources; SRC0X/SRC0Y in different banks (4 banks = VGPR index mod 4), VSRC1X/VSRC1Y in different banks; one even and one odd destination; at most one literal; no DPP; "These are hard rules - the instruction does not function if these rules are broken" | ISA3 7.6, p.67 | READ |
| Ops that wave64 runs in **one** pass (single-cycle wave64): `v_fma_f32/fmac/fmaak/fmamk`, `v_mul/add/sub/subrev_f32`, the F16 non-packed forms (`v_fma_f16`, `v_mul_f16`, `v_add_f16`, ...), `v_mov_b32`, `v_cndmask_b32`, `v_min/max_f32/f16`, 16-bit int add/sub/mad/mul_lo/logic/shift, `v_dot2_f32_f16`, `v_dot2c_f32_f16`, `v_dot2_f32_bf16`, `v_dot2_f16_f16`, `v_dot2_bf16_bf16`, `v_fma_mix*` (conditions) | ACO `aco_statistics.cpp:72-158` (`is_dual_issue_capable`) | READ |
| All other vector ops in wave64 cost two passes in ACO's model | ACO `aco_statistics.cpp:426-429` | READ |
| ACO costs (GFX10+): `valu32`, `valu_convert32`, `valu_fma`: latency 5, 1 VALU cycle; `valu_quarter_rate32` (e.g. `v_mul_lo_u32`): 4 cycles; transcendental: 1 VALU cycle + 4 cycles on the separate transcendental unit, latency 10 | ACO `aco_statistics.cpp:171-196`; `aco_opcodes.py:1345` | READ |
| LLVM GFX11SpeedModel: `Write32Bit` / `WriteFloatFMA` / `WriteFloatCvt` latency 5, `WriteQuarterRate32` 8, `WriteTrans32` 10 (separate `HWTransVALU`), `WriteLDS` 20, `WriteVMEM` 320 ("latency values are 1 / (operations / cycle). Add 1 stall cycle for VGPR read") | LLVM `SISchedule.td:405-441` | READ (a compiler model, not hardware documentation) |
| "DPP instructions incur an extra cycle of delay to execute." | ISA3 7.7, p.69 | READ |
| C&C: "RDNA 3 ... is not able to dual issue such packed instructions"; "there is only one VOPD instruction that can dual issue packed FP16 instructions along with another that can work with packed BF16 numbers"; "In wave64 mode, the SIMD will naturally try to start executing a 64-wide wavefront over a single cycle, provided the instruction can be dual issued" | C&C | READ (third party) |

So "dual issue" covers (a) VOPD pairs in wave32 and (b) single-pass wave64 for the ACO list. It does **not** cover
packed FP16 math (`v_pk_fma_f16`, `v_pk_mul_f16`), `v_dot4_i32_iu8`, `v_dot8_i32_iu4`, conversions, lane permutes,
`v_perm_b32` or integer 32-bit multiply. INFERRED from the three lists, which agree.

### II.2 Cost of the instructions a network layer needs (gfx11)

"Issue cycles" = cycles the SIMD's VALU is busy for one wave32 instruction. Wave64 doubles that unless the op is
dual-issue capable. MAC/lane counts come from the ISA3 pseudocode.

| instruction | MAC per lane | wave32 issue cycles | VOPD pair? | wave64 single pass? | MAC/SIMD/clk best case | MAC/CU/clk best case | source |
|---|---|---|---|---|---|---|---|
| v_fma_f32 / v_fmac_f32 | 1 | 1 | yes (FMAC, FMAAK, FMAMK) | yes | 64 (dual) / 32 (single) | 128 / 64 | ISA3, ACO |
| v_mul_f32 / v_add_f32 | (0.5 / 0.5) | 1 | yes | yes | 64 ops | 128 ops | ISA3, ACO |
| v_pk_fma_f16 | 2 | 1 | no | no | 64 | 128 | ISA3 7.5, ACO |
| v_fma_mix_f32 (F16 x F16 + F32) | 1 | 1 | no | conditional | 32-64 | 64-128 | ISA3, ACO |
| v_dot2_f32_f16 (VOP3P) / v_dot2acc_f32_f16 (VOP2) | 2 | 1 | DOT2ACC only | yes (both) | 128 (dual) / 64 (single) | 256 / 128 | ISA3 p.354, tables 91-92; ACO |
| v_dot2_f32_bf16 | 2 | 1 | DOT2ACC_F32_BF16 | yes | 128 / 64 | 256 / 128 | same |
| v_dot4_i32_iu8, v_dot4_u32_u8 | 4 | 1 | no | no | 128 | 256 | ISA3 p.355; ACO class valu32 |
| v_dot8_i32_iu4, v_dot8_u32_u4 | 8 | 1 | no | no | 256 | 512 | ISA3 p.355-356 |
| v_cvt_f16_f32, v_cvt_f32_f16, v_cvt_pk_rtz_f16_f32, v_cvt_f32_i32, v_cvt_i32_f32, v_cvt_pk_i16_i32, v_cvt_pk_u8_f32 | - | 1 | no | no | 32 results | 64 results | ACO class valu32 (`aco_opcodes.py:1028-1042, 1333, 1411`); LLVM `WriteFloatCvt` 5 |
| v_perm_b32 (byte select) | - | 1 | no | no | 32 | 64 | ACO valu32 |
| v_mul_lo_u32 | - | 4 | no | no | 8 | 16 | ACO `ValuQuarterRate32` |
| v_exp_f32, v_rcp_f32, v_rsq_f32 (transcendental) | - | 1 VALU + 4 trans | no | no | 8 | 16 | ACO; LLVM `WriteTrans32` |
| v_permlane16_b32, v_permlanex16_b32 (VOP3) | - | 1 (ACO model) | no | no | - | - | ISA3 p.423-424; ACO valu32 |

READ for the opcode facts and the ACO/LLVM model values. **No AMD document states issue cycles per instruction
for RDNA 3**: the cycle columns are the compilers' cost models (ACO, LLVM). That is source-derived, not documented.
The MAC/SIMD/clk columns are INFERRED from 32 lanes x MAC per lane x pairs per cycle.

Conversion notes (READ, ISA3): `v_cvt_pk_rtz_f16_f32` rounds toward zero (the GO4 article uses it,
`__builtin_amdgcn_cvt_pkrtz`, to pack F32 pairs into B operands). `v_cvt_f16_f32` follows the MODE round mode.
There is no packed round-to-nearest F32 -> F16 instruction on gfx11 (lane 4, section 5).

### II.3 Cross-lane exchange on gfx11 (correcting the brief's premise)

| mechanism | scope | cost as documented | source |
|---|---|---|---|
| `v_permlane16_b32 vD, vS, sLo, sHi` | arbitrary gather within each 16-lane row; selects in 2 SGPRs (4 bits per lane) | 1 VALU op (ACO valu32). Must not directly follow `v_cmpx` ("any other VALU opcode may be inserted") | ISA3 p.423, 7.2.8 p.62 |
| `v_permlanex16_b32` | gather from the **other** row of each 32-lane pair. With selects 0x76543210 / 0xfedcba98 it gives lane L <- lane L^16 | 1 VALU op (ACO valu32) | ISA3 p.423-424 |
| `v_permlane64_b32` | swaps the two 32-lane halves of a wave64; "Performs no operation in wave32 mode" | 1 VALU op | ISA3 16.8 p.292 (VOP1 103); ACO `aco_opcodes.py:1118` (gfx11=0x67) |
| DPP16 (`row_share`, `row_xmask`, shifts, broadcasts) | within 16 lanes only | +1 cycle ("DPP instructions incur an extra cycle of delay") | ISA3 7.7 p.69 |
| DPP8 | arbitrary within 8 lanes | +1 cycle | ISA3 7.7 |
| `ds_swizzle_b32` | xor/or/and masks within 32 lanes; ISA lists "SWAPX16 : xor_mask = 0x10" | LDS path, no LDS memory used; latency counted by LGKMcnt | ISA3 16.15 p.513 |
| `ds_bpermute_b32` / `ds_permute_b32` | arbitrary within 32 lanes (wave64: each half separately) | LDS path, "do not use any memory storage" | ISA3 12.5.2 p.128 |
| `v_readlane_b32` / `v_writelane_b32` | one lane to/from an SGPR | 1 op per lane moved | ISA3 |

READ. LLVM and ACO agree that `v_permlane16_b32` / `v_permlanex16_b32` exist since gfx10 (ACO `aco_opcodes.py:1420-1421`,
`op(gfx10=0x377, gfx11=0x25b)` / `op(gfx10=0x378, gfx11=0x25c)`). gfx12 adds `v_permlane16_var_b32` /
`v_permlanex16_var_b32` (ACO `aco_insert_NOPs.cpp:1436-1440` lists them alongside). The brief's statement that
permlane16 / permlanex16 are gfx12-only is wrong. READ.

WrongGPU's HLSL helper `wg_xor16` is `WaveReadLaneAt(v, WaveGetLaneIndex() ^ 16)`
(`clean/dx12/hlsl/wg_platform.hlsli:103-107`), and its comment names HIP's `ds_bpermute`. Which instruction a
D3D12 or Vulkan compiler emits for it on gfx11 (`ds_bpermute_b32` or `v_permlanex16_b32`) is UNKNOWN; read it
in an RGA or RADV ISA dump.

### II.4 Documented dot-product peak (no matrix instruction), RX 7900 XT, and the ratio to Part I

INFERRED, from II.2 and the AMD clock figures:

| path | MAC/CU/clk | MAC/s at 2.4 GHz (84 CU) | at 2.0 GHz | AMD figure | WMMA / this path |
|---|---|---|---|---|---|
| WMMA F16 / BF16 / IU8 (Part I) | 256 | 51.6 T | 43.0 T | 103 TFLOPS / TOPS | 1 |
| v_dot2_f32_f16, dual-issued (VOPD dot2acc pair, or wave64) | 256 | 51.6 T | 43.0 T | none | **1.0** |
| v_dot2_f32_f16, single issue (wave32, unpaired) | 128 | 25.8 T | 21.5 T | none | 2.0 |
| v_pk_fma_f16 | 128 | 25.8 T | 21.5 T | "FP16 Vector 51.6 TFLOPs" | 2.0 |
| v_dot4_i32_iu8 (wave32 or wave64) | 256 | 51.6 T | 43.0 T | none | **1.0** |
| v_fma_f32, dual-issued | 128 | 25.8 T | 21.5 T | "FP32 Vector 51.6 TFLOPs" | 2.0 |
| v_fma_f32, single issue | 64 | 12.9 T | 10.8 T | none | 4.0 |
| WMMA IU4 vs v_dot8_i32_iu4 | 512 vs 512 | 103.2 T | 86.0 T | 206 TOPS | 1.0 |

Consistency checks: AMD's "FP32 Vector 51.6" equals 84 x 128 MAC x 2 x 2.4 GHz, so it assumes dual-issued FMA.
AMD's "FP16 Vector 51.6" equals packed FMA without dual issue, which matches the VOPD list (no `v_pk_*`). GO3's
RDNA 2 column (IU8 512, FP16 256 per CU per clock) shows RDNA 2's dot4 already reached RDNA 3's WMMA IU8 rate per
CU. INFERRED.

The documented ratio of matrix to dot-product throughput is therefore **1.0 for INT8 and IU4, and 1.0-2.0 for F16**,
depending on whether the dot2 instructions dual-issue. Any WMMA speed-up on RDNA 3 has to come from fewer
instructions, fewer VGPR reads and fewer data loads per MAC, not from a higher arithmetic peak. INFERRED.

### II.5 Memory side: LDS, VGPRs, occupancy, bandwidth

| fact | value | source | label |
|---|---|---|---|
| LDS per WGP | 128 KB, "configured with 64 banks, each with 512 entries of 4 bytes" | ISA3 1.2.2.1 p.7 | READ |
| LDS per work-group | at most 64 KB, allocated in 1024-byte blocks | ISA3 1.2.2.1, 3.3.4 p.19 | READ |
| LDS halves | two 64 KB blocks, one per CU. CU mode: the wave's allocation stays on its own CU's side. WGP mode: can be anywhere | ISA3 3.3.4 p.19 | READ |
| LDS access time | "as little as one cycle (for wave32, or 2 cycles for wave64), or take as many 64 cycles, depending upon the number of bank conflicts" | ISA3 12.5 p.126 | READ |
| LDS peak bandwidth | one conflict-free wave32 dword access per cycle = 32 x 4 B = 128 B/clk; whether per CU or per WGP is not stated | INFERRED from the line above | UNKNOWN which unit |
| LDS latency | not documented. LLVM models 20 cycles, ACO models issue cost only | LLVM `SISchedule.td` | READ (model) |
| LDS size, LLVM | "gfx10 / 11 / 12: 64 KiB addressable, 128 KiB physical block" | LLVM `AMDGPUBaseInfo.cpp:1143-1149` | READ |
| VGPRs per wave | up to 256 (V0-V255) | ISA3 3.1 p.13, 3.3.2.1 p.17 | READ |
| VGPRs per SIMD, gfx1100 | 1536 (wave32 units); 768 in wave64 units | GOocc ("for the AMD Radeon RX 7900 XTX GPU, there are 1536 VGPRs available per SIMD"); LLVM `AMDGPUTargetParser.cpp:446-455`; `Feature1536VGPRs` set for gfx1100 (`GCNFeatures.td`, FeatureISAVersion11_0_0) | READ |
| VGPR file size | 1536 x 32 lanes x 4 B = **192 KB per SIMD**, 768 KB per WGP | INFERRED | INFERRED |
| Allocation block | "Devices which have 1536 VGPRs per SIMD allocate in blocks of 24 for wave32 and 12 for wave64" | ISA3 3.3.2.1 p.17; LLVM `AMDGPUTargetParser.cpp:430-438` | READ |
| Wave slots | "RDNA 2 and RDNA 3 have 16 slots per SIMD" | GOocc | READ |
| Waves per SIMD (wave32) | min(16, max(1, floor(1536 / roundUp(V, 24)))) | LLVM `AMDGPUBaseInfo.cpp:1367-1374` | READ |
| Waves per SIMD (wave64) | min(16, floor(768 / roundUp(V, 12))) | same function with wave64 totals | INFERRED (no source states the wave64 slot cap separately) |
| DRAM | 800 GB/s, 320-bit GDDR6, 20 GB; 80 MB Infinity Cache, "Effective Memory Bandwidth Up to 2900 GB/s" | SPEC-XT | READ |
| DRAM per CU per clock | 800e9 / (84 x 2.4e9) = 3.97 B/CU/clk (DECK gives 4.2 for the 7900 XTX) | INFERRED; DECK | INFERRED |

Occupancy by VGPR count, wave32, gfx1100 (INFERRED from the formula):

| VGPRs used | allocated | waves/SIMD |
|---|---|---|
| <= 96 | 96 | 16 |
| 128 | 144 | 10 |
| 168 | 168 | 9 |
| 192 | 192 | 8 |
| 216 | 216 | 7 |
| 256 | 264 | 5 |

WMMA register cost per tile on gfx11 wave32 (INFERRED from I.2): F16 A 8 + B 8 + C/D 8 = 24 VGPRs. The same tile
on gfx12 as FP8 is 2 + 2 + 8 = 12. A kernel holding k independent accumulators and reusing one A against them
needs 8 + 8k + 8k (B and C per accumulator) on gfx11. That is where the gfx11 occupancy pressure comes from.

## Part III: using these for neural networks on RDNA 3 (what the documents say)

1. GO3 recommends FP16/BF16 for "training online or offline" and "8-bit and 4-bit integer data types suitable for
   inference". READ.
2. GO3's speed-up claims: FP16 2x RDNA 2 per CU per clock (512 vs 256), IU8 and IU4 unchanged per CU per clock
   (512 = 512, 1024 = 1024), and "MISA ... Resnet50 for a performance uplift of roughly 2x from RDNA 2". TH reports
   AMD's launch claim of "2.7x" matrix throughput, which TH attributes to more CUs, clock and the 2x per CU. READ.
3. Moving a result back into an operand: GO4 says that on RDNA 3 "C and D matrices are split into even and odd for
   lower 16 lanes and upper 16 lanes. Therefore, in order to convert D matrix to B matrix ... data needs to be
   shuffled among lanes in RDNA 3. This is not needed for RDNA 4." READ. GO3 gives no cost and no code for it.
4. Cost of that shuffle, wave32 F32 D -> F16 B (INFERRED from CALC's layouts). Each lane holds 8 of the 16 rows
   it needs, the even rows in lanes 0-15 and the odd rows in lanes 16-31. It needs 8 `v_permlanex16_b32` (one per
   F32 VGPR) plus 8 `v_cvt_pk_rtz_f16_f32`, or 4 conversions + 4 exchanges + about 8 `v_perm_b32` to recombine
   halves. The pack order differs between the half-waves (own row is the low half in lanes 0-15, the high half in
   16-31), which adds a select or an EXEC-masked second pass. Total about 16-24 single-cycle vector ops per 16x16
   tile, against 32 cycles for the WMMA it feeds. On gfx12 the same step is 4 conversions and no exchange.
   Wave64 adds `v_permlane64_b32` steps (each lane holds 4 of 16 rows). None of this is documented as a cost.
5. GO4's own MLP example only works on RDNA 4 ("we do not need any data exchange among lanes, since matrix D and B
   have the same layout"). GO3 has no MLP or chaining example. READ.
6. AMD's spec pages put the RDNA 3 F16 and INT8 matrix peaks at the same 103 T(FL)OPS, so INT8 brings no
   arithmetic gain over F16 on WMMA. It halves A/B registers and load bytes (4 vs 8 VGPRs). READ / INFERRED.

Where the documentation is silent:
- Wave64 WMMA cycle count (CALC has one field; ISA3, GO3 silent). ACO also costs one 32-cycle pass.
- WMMA result latency (time until D is readable by a VALU op). ACO models 4 + 32 = 36 cycles; no AMD text.
- How the "internal DOT instructions" split K = 16, the intermediate precision, and whether F16 denormal inputs
  and outputs are kept or flushed (only "only round-to-nearest-even" and "no ALU exceptions" are stated).
- Whether a WMMA on one wave blocks VALU issue from **other** waves on the same SIMD for all 32 cycles (CALC's
  "co-execute: False" implies yes; no prose says so).
- Issue cycles of any RDNA 3 vector instruction: only compiler models (ACO, LLVM) give them; ISA3 gives none.
- LDS bandwidth per clock and LDS latency.
- Cycle cost of `v_permlanex16_b32` versus `ds_swizzle_b32` / `ds_bpermute_b32` for a half-wave swap.
- Base clock of the RX 7900 XT and the clock the "Peak" figures assume (inferred 2.4 GHz from the arithmetic).

Where two sources disagree:
- **ROCPROF vs ISA3 / C&C / SPEC-XT on packed FP16 dual issue.** ROCPROF (written for RDNA 3.5) says
  "packed FP16 VOPD dual-issue: 512 FLOPs/CU/cycle". ISA3's VOPD tables contain no packed-math opcode,
  C&C says packed instructions cannot dual issue, and AMD's spec sheet gives FP16 Vector = FP32 Vector = 51.6
  TFLOPS (256 FLOPs/CU/clk). 512 is only reachable through `DOT2ACC_F32_F16` pairs, which are dot products, not
  packed FMA.
- **GO3 vs ISA3 on "IU".** GO3: "IU8 and IU4 refers to the unsigned 8-bit integer datatype and unsigned 4-bit integer
  datatype". ISA3: "IU4 and IU8 mean that the operand is either signed or unsigned ... as indicate by the NEG bits".
  The ISA is the authority.
- **LLVM's scheduling model vs CALC/ACO on WMMA cost.** LLVM schedules gfx11 WMMA as a 5-cycle ordinary op;
  CALC and ACO use 32 cycles. This only affects how well HIP/LLVM code hides WMMA time, not the hardware rate.
- **ISA3 vs ISA4 on the arithmetic.** ISA3: RNE only, "internally use the DOT instructions". ISA4: "using fused multiply
  add". The generations may differ, but no document states RDNA 3's intermediate rounding.
- GO3/GO4 label integer rates "FLOPS". AMD spec pages use "TOPs". Same numbers, different words.

## Documented expectation (check measurements against this)

All values per SIMD unless stated, wave32 unless stated, RX 7900 XT = 84 CU / 168 SIMD.
f = measured shader clock: convert by MAC/s = MAC/CU/clk x 84 x f.

| # | quantity | documented value | source | confidence |
|---|---|---|---|---|
| 1 | v_wmma_f32_16x16x16_f16 cycles per instruction (back-to-back, independent) | 32 | CALC, ACO, GO3 "32 clocks" | high |
| 2 | v_wmma_f32_16x16x16_bf16, v_wmma_f16_16x16x16_f16, v_wmma_bf16_16x16x16_bf16 | 32 each | CALC, ACO | high |
| 3 | v_wmma_i32_16x16x16_iu8 | 32 | CALC, ACO ("int8 and (b)f16 have the same performance") | high |
| 4 | v_wmma_i32_16x16x16_iu4 | 16 | CALC, ACO | high |
| 5 | wave64 WMMA cycles | 32 (not stated; one CALC field, ACO single pass) | CALC, ACO | low |
| 6 | WMMA MAC per CU per clock (F16/BF16/IU8; IU4) | 256; 512 | CALC, GO3, GO4 | high |
| 7 | WMMA peak, whole card | 84 x 256 x f MAC/s = 51.6 T at 2.4 GHz, 43.0 T at 2.0 GHz | SPEC-XT (103 TFLOPS / TOPS), CALC | high |
| 8 | VALU issue while a WMMA runs on the same SIMD | none (no co-execution) | CALC | medium |
| 9 | NOP between WMMA whose D feeds next WMMA's A/B | 1 V_NOP or independent VALU; C = D chaining needs none | ISA3 7.9, LLVM, ACO | high |
| 10 | WMMA rounding | round-to-nearest-even only; MODE bits not read; no exceptions | ISA3 7.9; LLVM `ReadsModeReg = 0` | high (as stated) / unknown intermediate |
| 11 | WMMA single rounding per output (fused 16-term sum) | not claimed for RDNA 3; "internally use the DOT instructions" | ISA3 7.9 | n/a (silent) |
| 12 | v_fma_f32 wave32 | 1 instr/clk; 2/clk as VOPD pair | ISA3 7.6, ACO | high |
| 13 | v_fma_f32 wave64 | 1 instr (64 lanes)/clk (single pass) | ACO, C&C | medium |
| 14 | v_dot2_f32_f16 wave32 unpaired | 1 instr/clk = 64 MAC/SIMD/clk | ACO, ISA3 | medium |
| 15 | v_dual_dot2acc_f32_f16 pair | 2 ops/clk = 128 MAC/SIMD/clk | ISA3 tables 91-92 | medium |
| 16 | v_dot2_f32_f16 wave64 | 1 instr/clk = 128 MAC/SIMD/clk | ACO | medium |
| 17 | v_dot4_i32_iu8 wave32 / wave64 | 1 instr/clk / 1 per 2 clk = 128 MAC/SIMD/clk | ACO, ISA3 | medium |
| 18 | v_pk_fma_f16 | 1 instr/clk = 64 MAC/SIMD/clk, never dual | ISA3 VOPD list, SPEC-XT FP16 vector 51.6 | high |
| 19 | conversions (v_cvt_f16_f32, v_cvt_pk_rtz_f16_f32, ...) | 1 instr/clk wave32, 2 clk wave64 | ACO valu32 | medium |
| 20 | transcendental (v_exp_f32, v_rcp_f32) | 1 per 4 clk on the transcendental unit | ACO, LLVM | medium |
| 21 | v_mul_lo_u32 | 1 per 4 clk | ACO | medium |
| 22 | v_permlanex16_b32 (lane L <-> L^16) | 1 vector op; cycle cost not documented | ISA3, ACO | low (cost) / high (exists) |
| 23 | dot-path peak / WMMA peak | INT8 1.0; F16 0.5 (unpaired) to 1.0 (dual) | derived from 6, 14-17 | medium |
| 24 | 9070 XT FP8 WMMA / 7900 XT F16 WMMA | 4.0 per CU-clk; 3.77 at boost clocks (194.6 vs 51.6 T MAC/s) | CALC, SPEC-9070, SPEC-XT | high |
| 25 | VGPRs per SIMD; allocation block | 1536 (wave32 units); 24 (wave32) / 12 (wave64) | ISA3, LLVM, GOocc | high |
| 26 | wave slots per SIMD | 16 | GOocc | high |
| 27 | waves/SIMD at 128 / 192 / 256 VGPRs (wave32) | 10 / 8 / 5 | LLVM formula | high |
| 28 | LDS per WGP / per work-group | 128 KB / 64 KB | ISA3 | high |
| 29 | LDS conflict-free access | 1 clk wave32, 2 clk wave64 per instruction; up to 64 with conflicts | ISA3 12.5 | medium |
| 30 | DRAM bandwidth | 800 GB/s (3.97 B/CU/clk at 2.4 GHz) | SPEC-XT | high |

## Open questions

1. How does the hardware split K = 16 inside a gfx11 WMMA ("internally use the DOT instructions"), and how wide
   are the intermediate sums? Is it one F32 rounding per DOT step, and in which order?
2. Does a wave64 WMMA take 32 or 64 cycles? Only one value exists in any source.
3. Is WMMA result latency about 36 cycles (ACO's model), and does an accumulate-only chain (C = D) run back to back
   at 32 cycles per instruction, as ISA3's hazard text implies?
4. Does a WMMA on one wave stall VALU issue from other waves on the same SIMD for its whole 32 cycles?
5. Does the D3D12 driver or RADV compile WrongGPU's `wg_xor16` to `v_permlanex16_b32` or to `ds_bpermute_b32`, and
   what does each cost?
6. Does the compiler actually emit `v_dual_dot2acc_f32_f16` pairs or wave64 single-pass dot2 for a dot-product
   fallback? Only then does the F16 dot path reach WMMA's peak.
7. Which shader clock does the card hold under a sustained WMMA load (AMD's 103 TFLOPS assumes about 2.4 GHz)?

## What to measure on an RX 7900 XT

1. **Clock first:** record the shader clock during each test (`SHADER_CYCLES` / `s_getreg` against wall time, or the
   driver's sensor) and report results per CU per clock, so they compare with this table at any f.
2. **WMMA rate:** a loop of independent `v_wmma_f32_16x16x16_f16` (4+ accumulators, C = D, no NOPs needed) in wave32.
   Expect 32 cycles per instruction per SIMD = 256 MAC/CU/clk. Repeat for `_f16_16x16x16_f16`, `_bf16`, `_iu8`
   (expect 32) and `_iu4` (expect 16).
3. **Wave64 WMMA:** the same loop with `_w64` forms. Expect 32 (documented single figure). 64 means half the
   per-lane rate.
4. **Latency and the D -> A/B hazard:** one dependent chain where D feeds the next B, with and without one
   `v_nop`. Measure cycles per link: the ACO model predicts about 36.
5. **Co-execution:** two waves per SIMD, one running WMMA and one running `v_fma_f32`. Check whether the VALU wave's
   rate drops to zero while the WMMA runs.
6. **Dot paths:** loops of `v_dot2_f32_f16` (wave32 unpaired), `v_dual_dot2acc_f32_f16` (VOPD, from hand-written ISA or
   a compiler that emits it), `v_dot2_f32_f16` in wave64, and `v_dot4_i32_iu8` in wave32 and wave64. Expect 64,
   128, 128, 128, 128 MAC/SIMD/clk.
7. **Lane exchange:** cycles per `v_permlanex16_b32`, per `ds_swizzle_b32` SWAPX16, and per `ds_bpermute_b32` for
   the L^16 swap. Then the full F32 D -> F16 B conversion of item 4 in Part III, against the 16-24 op estimate.
8. **K grouping:** WMMA with exactly two non-zero products of opposite sign and large magnitude plus a small third
   term, swept over k positions. Exactness breaking at certain k pairs reveals the internal grouping (open question 1).
9. **Occupancy:** a kernel padded to 128, 192 and 256 VGPRs. Read the launched waves per SIMD in RGP or the
   profiler: expect 10, 8 and 5.
