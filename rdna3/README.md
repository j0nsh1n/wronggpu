# RDNA 3 analysis for WrongGPU (issue #28)

A written analysis of what an RDNA 3 (gfx11) path for WrongGPU would need, with measurements from a Radeon RX 7900 XT
under Linux (Mesa RADV). Made on 2026-11-14 against WrongGPU main at 994f304. Start with
[RDNA3-ANALYSIS.md](RDNA3-ANALYSIS.md); its first section is a one-page summary.

| Path | What |
|---|---|
| `RDNA3-ANALYSIS.md` | the document |
| `findings/` | nine research notes it is built from, with full citations |
| `probe-vk/` | matrix-instruction arithmetic probe (native Vulkan, `GL_KHR_cooperative_matrix`): sources, `RESULTS.md`, disassembly |
| `probe-post/` | what the instruction's small errors do after rounding to E4M3 or FP16 (`POST.md`) |
| `probe-vk-valu/` | shader and matrix timing (`RESULTS-VALU.md`), sources, disassembly, raw clock readings |
| `int8-study/` | CPU study of INT8 forms of Model M (`STUDY.md`) and a script that runs on any converted model |

## Probe data for a replay on RDNA 4

The inputs and the RX 7900 XT's outputs (`*.npz`, about 0.7 GB) are downloads on the release
[rdna3-probe-data-1](https://github.com/j0nsh1n/wronggpu/releases/tag/rdna3-probe-data-1). Put them in
`probe-vk/results/`. Their layout is in `RDNA3-ANALYSIS.md` section 10 and `probe-vk/RESULTS.md` section 5. The inputs
also regenerate from `probe-vk/gen.py` (fixed seeds).

## What is not here

- No model data and nothing derived from DLSS files. The INT8 study's per-tensor tables are left out for that reason.
- Nothing was run on Windows or on an RDNA 4 card. No game was run.

## How it was made

Reading and measuring were done with Claude models (Anthropic) under the direction of the repository owner of this
fork, and checked against the cited source lines and the probe outputs. Every claim in the document carries a label:
measured, read in a named source, inferred, or unknown until measured. Treat each "inferred" as a claim to test.

Licence: MIT, as WrongGPU. The probe programs and scripts were written for this analysis.
