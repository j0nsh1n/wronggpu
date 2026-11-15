# Lane 2: GPU gates and everything outside the kernels that assumes RDNA 4

Source: upstream main at 994f304 (read-only). All paths are relative to the repo root. Labels: READ = read in code or
docs; INFERRED = my reasoning (shown); UNKNOWN = only a test on hardware can tell; MEASURED-IN-REPO = the repo
records a measurement. No GPU program, build or test was run for this lane.

Reference card used below: RX 7900 XT, PCI 1002:744C, DXGI name "AMD Radeon RX 7900 XT", HIP arch gfx1100.
(The PCI id is the maintainer's brief; nothing in the repo lists it.)

## 0. Summary

- There is no single "supported GPU" switch. The same question ("is this an RDNA 4 card?") is answered in **six
  separate places**, written six times, by three different methods: DXGI name text, a PCI id list, and the HIP
  architecture string. READ.
- Every one of them says no to a 7900 XT, and **every one of them fails closed**. Nothing guesses. On a 7900 XT the
  launcher blocks the install, the runtime refuses the device, and the game keeps its own upscaler or denoiser. READ.
- The **runtime does have a capability probe** (not just an assumption): `native_driver_probe` in the Model M runtime
  and `rr_rt_selftest` in the RR runtime. Both run real matrix instructions through the driver channel at create and
  compare the result. They only run after the name gate, so on RDNA 3 they are currently unreachable. READ.
- The probes are written for gfx12 and FP8. The Model M probe uses FP16 and FP8 in one kernel, so on a card with no
  FP8 matrix instruction it cannot pass as written. Its FP16 half is a usable starting point for a gfx11 probe. INFERRED.
- Nothing selects kernels or models by GPU. The natural seams are listed in section 6.
- CI has no GPU job. No test anywhere runs on a second GPU family, and none is set up to. READ.

## 1. Where "is this a supported GPU?" is decided

Columns: what is checked, what happens on failure, what the 7900 XT hits today.

### 1.1 Installer / launcher (clean/installer)

| file:line | What is checked | Failure behaviour | 7900 XT today |
|---|---|---|---|
| clean/installer/src/wg_core/checks.cpp:30 `IsRdna4Name` | DXGI description contains `RX 90`, `PRO R97` or `PRO R96` (text match, not a number range) | returns false | "AMD Radeon RX 7900 XT": false. (`RX 90` would also match `RX 9000` or a future `RX 9xx`; READ) |
| checks.cpp:67-83 `LinuxRdna4Name`, id list at :74 | Only under Wine (`UnderWine()`, :110): reads `/sys/class/drm/card*/device/{vendor,device}` through `Z:`; vendor 0x1002 and device in {0x7550, 0x7551, 0x7590, 0x7591} | returns empty name | 744C is not in the list, so no |
| checks.cpp:93-116 `DetectGpu` | enumerates DXGI adapters, skips software; takes the first adapter, then any AMD (vendor 0x1002) adapter, preferring one whose name passes `IsRdna4Name` (:106-107) | `GpuInfo.rdna4 = false`, `amd = true` | `amd = true, rdna4 = false` |
| checks.cpp:124 `kSupportedArch = {"gfx1201","gfx1200"}` and :192-206 `FindArchName` | HIP architecture string found by scanning the device-properties buffer for "gfx" + digits (works across HIP versions) | `HipProbe` (:209-250) skips the device; exit 14 (no RDNA 4 device) or 16 (other gfx12 only) | gfx1100 is not `gfx12*`, so exit 14 |
| checks.cpp:244 | `arch.rfind("gfx12",0)==0` marks "some RDNA 4 family device seen" | exit 16 vs 14 only changes the message | not hit |
| checks.cpp:536-538 (the GPU row) | `!sys.gpu.rdna4` | row `L_GPU` = FAIL "WrongGPU needs an AMD Radeon RX 9000-series graphics card. (found <name>)"; `L_HIP` = N/A | FAIL with that text and "(found AMD Radeon RX 7900 XT)" |
| checks.h:20 `blocking()` | FAIL or N/A on any row blocks install, except GPU and HIP rows when `/FORCE` (:557-558 adds " - check skipped (/FORCE)") | install blocked | blocked, unless the user passes `/FORCE` (documented as "tests, development", clean/installer/src/app/main.cpp:113) |
| checks.cpp:545-548 and clean/installer/src/app/cli.cpp:72 | `ctx.hipOptional` (the launcher): when the card passed the name check, a broken HIP runtime is only a warning | warning, not a block | not reached |
| checks.cpp:550-554 | Installer path (not launcher): name passed but HIP found no gfx1200/gfx1201: "Installation is blocked." | block | not reached |
| clean/installer/src/app/ui_checks.cpp:38-50 | First-run page: `gpu.rdna4` selects the wording; `rx9060` (name contains "9060") adds "Supported, not yet tested ... takes about twice as long" | line severity 2 (problem) and "Direct3D 12: Not available on this card" | shows the card name plus ". WrongGPU needs a Radeon RX 9000-series card." |
| clean/installer/src/app/launcher_ui.cpp:1101 | toolbar status: `rdna4` -> "· Direct3D 12"; else `amd` -> "· HIP"; else "· Not supported" | label only | shows "AMD Radeon RX 7900 XT · HIP". This is wrong for RDNA 3: HIP does not run there either (see 1.3). READ |
| clean/installer/src/app/ui_render.cpp:58 | test-fixture renderer: `rdna4 = name contains "RX 90"` | fixture only | n/a |
| clean/installer/src/wg_core/rrprep.cpp:270 | RR prepare helper exit code 2 from the RR runtime (adapter not supported) | "no Radeon RX 9000 series graphics card (Ray Reconstruction needs one)", RR not prepared | this text, via 1.4 |
| clean/installer/src/wg_core/logs.cpp:~205-222 | support bundle lists every DXGI adapter with vendor_id, device_id, dedicated VRAM, UMD driver | none (logging only) | would record 4098 / 17484 (0x744C) and VRAM. Useful for RDNA 3 reports |
| clean/installer/src/app/ui_game.cpp:1356, clean/installer/src/wg_core/profile.cpp:19 | wording only: "Auto uses Direct3D 12 on RX 9000-series cards" | none | wording |

Driver-version checks: the installer reads each adapter's UMD driver version (checks.cpp:96-103) but only to display and
log it and to key the RR "prepared" record (rrprep.cpp, `PrepLine.driver`). There is **no minimum-driver check anywhere**
in the repo (grep for driver-version comparisons found none). READ.

### 1.2 Runtime (Model M, Model K), D3D12 path

| file:line | What is checked | Failure behaviour | 7900 XT today |
|---|---|---|---|
| clean/runtime/rdna4.h:16-19 `rdna4_adapter()` | vendor == 0x1002, not a software adapter, and description contains "RX 9070" or "RX 9060" (no PCI device id used) | false | false. Note: "PRO R97/R96" workstation cards pass the installer's check but **fail** this one, so the installer and the runtime disagree for those boards. READ |
| clean/runtime/interop/executor_select.h:105-107 (`choose_executor`, auto) | adapter unknown or `!rdna4_adapter` | executor = `hip`, one INFO line "the adapter X is not an RDNA 4 Radeon RX 9070 / RX 9060, where the D3D12 executor runs" | picks HIP (see next row) |
| clean/runtime/interop/backend.cpp:~615-650, :649 `configure_hip(...)` | HIP executor is created | `return fail(WG_ERROR, err)`: the context is not created | context creation fails (see 1.3) |
| clean/runtime/interop/executor_select.h `executor=dx12` forced | skips the name rule, runs the D3D12 executor on any adapter; the device then refuses it (next rows) | warning, HIP executor takes the context (Model M); Model K: context ends (backend.cpp:672) | falls through to the device gate |
| clean/runtime/device_d3d12.cpp:1367-1368 (game's device, "borrowed") | `rdna4_adapter` on the DXGI adapter of the supplied device | `fail("native D3D12 borrowed device requires the supported Radeon RX 9070/9060 adapter")` | refused here |
| clean/runtime/device_d3d12.cpp:1440 and :1444 (own device, enumeration) | same predicate; first matching adapter is selected (an AMD iGPU is deliberately not selected, comment :1439) | `fail("native D3D12: no supported Radeon RX 9070/9060 adapter")` | refused here |
| clean/runtime/device_d3d12.cpp:1369-1371, :1452-1454 | `D3D12_FEATURE_SHADER_MODEL` >= 6.6 | fail | passes on RDNA 3 with a current driver (INFERRED; UNKNOWN until run) |
| clean/runtime/device_d3d12.cpp:~205-215 (init) | `DoublePrecisionFloatShaderOps` (D3D12 folds the extended-double ops into it); kernels flagged `fp64:"ext"` are removed from the port table when missing | log line, no port; a context that needs it is refused | UNKNOWN: whether the RDNA 3 D3D12 driver reports it |
| clean/runtime/device.cpp:398 and :406 (HIP device) | HIP arch must equal `gfx1200` or `gfx1201` exactly | `fail("no matching HIP device with packaged gfx1200/gfx1201 kernels: ...")` | gfx1100: refused. Also the code objects in the build are only gfx1201/gfx1200 (clean/runner/build.py:149 `ARCH`, :481 rejects any arch outside those two) |
| clean/runtime/device.cpp:693 `alt_arch_` | "never mix code objects across architectures": code objects are looked up as `<arch>.<name>` | fail "missing gfx.. code object" | n/a |
| clean/tests/test_device_targets.cpp:42-44 | **asserts** that `gfx1100`, `gfx1299` and `unknown` are refused by `make_hip_device` | test would fail the day gfx1100 is supported | this test must change with any RDNA 3 HIP support. READ |

### 1.3 What a 7900 XT does inside a game today, through the FSR front end (clean/fsr)

READ unless noted.
1. The front end DLL itself has **no GPU check**: grep of clean/fsr/src for vendor, RDNA, gfx, RX 9 finds only comments
   and the DXGI/HIP strings below. The decision is entirely the runtime's.
2. The runtime picks the HIP executor (1.2), `make_hip_device` loads `amdhip64_7.dll` and refuses gfx1100
   (device.cpp:406), `configure_hip` fails, the context create returns an error (backend.cpp:649).
3. The front end then follows `fallback_to_amd` (default true, clean/fsr/src/config.h:35): the context goes to AMD's own
   FSR for that game. With `fallback_to_amd = 0` the create fails with `ERROR_RUNTIME_ERROR`
   (clean/fsr/src/frontend.cpp:819-820). So the game is never left without an upscaler. If the HIP DLL is missing
   the front end logs the same path (clean/fsr/src/win_util.cpp:539-564).
4. The OptiScaler plugin path uses the same front end, so the same fallback rules apply. I did not read the plugin's
   own host code for GPU decisions beyond this; the clean/fsr tree has none. (INFERRED from the grep.)

### 1.4 Ray Reconstruction (clean/rr_f)

| file:line | What is checked | Failure behaviour | 7900 XT today |
|---|---|---|---|
| clean/rr_f/runtime/src/rr_runtime.cpp:185-187 `rdna4_adapter` | a **copy** of the runtime rule (vendor 0x1002, not software, name has "RX 9070"/"RX 9060") | false | false |
| rr_runtime.cpp:214-216 `rdna4_device_id` + :220-231 `rdna4_adapter_desc` | if the DXGI text fails, asks the kernel driver (D3DKMT `PHYSICALADAPTERDEVICEIDS` by LUID, :189-212) for the real PCI ids and accepts 0x7550/0x7551/0x7590/0x7591. Purpose: survive OptiScaler spoofing the DXGI vendor/name | false, no note | false (744C not in the list) |
| rr_runtime.cpp:1344-1353 (`set_device_locked`) | `rdna4_adapter_desc` on the game's adapter | log "adapter ... is not an RDNA 4 board: RR not available"; `WG_E_ADAPTER_NOT_SUPPORTED`; `device_failed = true` | refused |
| rr_runtime.cpp:1152, :3784, :4285 | adapter list/status and prepare-helper adapter pick, all through `rdna4_adapter_desc` | adapter not chosen / status "no" | not chosen |
| rr_runtime.cpp:1367-1375 | `DoublePrecisionFloatShaderOps` required ("exposure / output stage need them") | `WG_E_ADAPTER_NOT_SUPPORTED` | not reached; UNKNOWN on RDNA 3 |
| clean/rr_f/proxy/src/proxy_core.cpp:410-440, :836-852 | proxy reports "unsupported" (NGX `is-supported` false) when the runtime says no, when RR is switched off, or when the runtime's `adapter_supported(luid)` is false | game uses its own denoiser. Log text "not an RDNA 4 GPU, or a LUID size other than 8" (:435-437) | game keeps its own denoiser |
| clean/rr_f/head/dx12/rr_host.cpp:96 and clean/rr_f/gpu/rr_host.cpp:92 | test hosts pick the first vendor-0x1002 hardware adapter | test only | would pick the 7900 XT and run RDNA 4 kernels on it; test hosts have no gate. READ |

### 1.5 Other places

| file:line | What | Note |
|---|---|---|
| clean/linux/README.md:29, clean/linux/run_wine_tests.py:25 | "Needs an AMD RDNA 4 GPU with RADV" | the Linux suite's own assumption; `ags-layout` checks the gfx12 register semantics only (run_wine_tests.py:176) |
| clean/converter (grep rdna/gfx/RX 9) | nothing | the converter is GPU-independent. READ |
| .github/workflows/ci.yml:2 "No GPU job" | all jobs are CPU, `windows-2022` or `ubuntu-24.04`; the launcher job uses a **mock HIP DLL** reporting `gfx1201` (clean/installer/tests/mock_hip.c:11) | no hardware of any kind in CI |
| .github/ISSUE_TEMPLATE/game-compatibility-report.yml:99 | placeholder "RX 9070 XT, Adrenalin version..." | wording |
| clean/runner/packing.py:17-18, clean/runtime/runtime.cpp:617-621 | a `gfx11` accumulator row table (`row = 2v + h`) already exists "for probes", and the runtime accepts a non-gfx12 `d_row` on the HIP path. `native_direct_io` refuses it (runtime.cpp:620-621) | this is existing groundwork for gfx11 layouts, but only a table. See lane 1 for what the HLSL kernels do with it |

## 2. Start-up sequence on a GPU, and where a card without FP8 matrix ops first fails

### 2.1 Model M in a game (D3D12 executor; the default on RDNA 4). All READ unless marked.

1. **Executor choice**: `choose_executor` (clean/runtime/interop/executor_select.h:105) with the adapter read from
   DXGI (backend.cpp:612). RDNA 4 name -> `dx12`. Else HIP.
2. **Device**: the game's own D3D12 device and queue are borrowed (`make_d3d12_device_on`,
   clean/runtime/interop/executor_d3d12.cpp:126, :152, native_candidate.cpp:181). Not a device of WrongGPU's own.
3. **Adapter gate**: `rdna4_adapter` (device_d3d12.cpp:1367). *A 7900 XT stops here today.*
4. Shader model 6.6 check (:1369-1371).
5. Manifest read (`read_manifest`, device_d3d12.cpp:1237+): `dx12/manifest.json`, one entry per kernel with name, file,
   `co`, block, wave, args, SHA-256. Kernels flagged fp64 are dropped when the device lacks the feature (:210-215).
6. `create_objects` (:1595-1643): queue, fence, allocators, the arena buffer (default 2560 MiB, :191, or measured, see
   4.4), and a **256-byte `channel_` buffer** with a raw UAV, bound by a descriptor table range at **register space
   0x7FFF0ADE** (:1627). That is the driver-channel UAV. The root signature has 4 parameters (UAV, UAV, root constants,
   descriptor table with the channel UAV and the GELU-T5 SRV).
7. `profile_init`, `reserve_events(64)`.
8. **Driver-channel probe** (`driver_probe`, :1385-1409, called at :218 **only when the device is borrowed**, i.e. in a
   game): finds the `native_driver_probe` kernel (dx12/kernels/native_game_helpers.hlsl:20), `create_pso` for it on the
   first launch (:647 -> :1645: reads DXIL, checks SHA-256, `CreateComputePipelineState`), uploads 4.5 KiB of operands
   (clean/runtime/wmma_probe.h), dispatches **one wave of 32 lanes**, reads back 2 KiB and compares every float
   (`first_mismatch`). The kernel runs one **FP16** `wg_wmma_f16` and one **FP8** `wg_wmma_fp8`
   (native_game_helpers.hlsl:25-26, both before any output store). Test hook `WG_TEST_DRIVER_PROBE_FAIL=1` forces a
   mismatch (device_d3d12.cpp:1400).
   Mismatch -> `fail("... driver-channel probe mismatch at float N")` -> the executor create fails.
9. GELU tables uploaded (`gelu_lut_upload`, `gelu_t5_upload`).
10. The executor creates the runtime twice: first only to measure the arena need, then again with an exactly sized
    arena (executor_d3d12.cpp:17-18, :118-152). Then `warm_up` (device_d3d12.cpp:443-451) creates **every** PSO up
    front; any `CreateComputePipelineState` failure ends the create. Create failures (Model M) fall back to the HIP
    executor with one WARN line (backend.cpp:673-679); for Model K the context ends (backend.cpp:672).
11. First frame: `launch` -> `Dispatch` on the game's command list (device_d3d12.cpp:640-660). Device removal is
    watched (:974, :1022 "device removed").

So the probe is the **first thing that executes matrix instructions**, before any network work. In the current code
nothing reaches it on RDNA 3 because step 3 refuses first.

### 2.2 Where a card without FP8 matrix instructions would first fail (if step 3 were opened)

The kernels never contain a WMMA instruction in the DXIL. They contain an `InterlockedCompareExchange` on a UAV in
register space 0x7FFF0ADE whose "address" is an encoded instruction word (clean/dx12/hlsl/wg_ags.hlsli:1-80). The
driver's shader compiler turns that into the real `v_wmma_*` instruction. FP8 x FP8 -> FP32 is format immediate 0x7
(`WG_AGS_FP8_F32`). So what happens on RDNA 3 depends on what AMD's RDNA 3 driver does with that code word. Possible
outcomes, in the order they would be met:

| Stage | Possible outcome on gfx11 | Label |
|---|---|---|
| PSO compile of `native_driver_probe` | the driver rejects the encoded op -> `CreateComputePipelineState` fails -> clear error at create | UNKNOWN. A test is needed |
| PSO compile OK, driver does **not** treat the space as special | it is an ordinary UAV; the atomic goes to an out-of-range address of a 256-byte buffer; D3D12 defines out-of-range raw-buffer atomics as no-ops returning 0. Every probe output stays 0 (or the upload value) -> "probe mismatch at float 0". Clean refusal | INFERRED (D3D12 bounds rule is standard; not tested here) |
| PSO compile OK, driver emits a gfx11 WMMA for the FP16 format (0x5) | RDNA 3 WMMA fragments are not laid out like gfx12 (the repo itself knows two accumulator row tables, `row = v+8h` for gfx12 and `row = 2v+h` for gfx11, clean/runner/packing.py:17-18; the probe's own unit test lists "accumulator rows 2v + h" as a layout that **must fail**, clean/runtime/tests/test_wmma_probe.cpp:218-222). The FP16 results would not match | INFERRED for the mismatch if the driver accepts the code at all; UNKNOWN what the driver does with the A/B lane layout |
| FP8 op (0x7) | RDNA 3 has no FP8 matrix instruction. Driver may fail compile, emit garbage, or emulate. All three show as a mismatch or a PSO failure | UNKNOWN |
| Device removal / TDR / crash | a bad code word reaching the hardware could hang the queue. `finish_queue` and `GetDeviceRemovedReason` (device_d3d12.cpp:974, :1022) would report it, but a game-side hang is possible | UNKNOWN. Cannot be excluded without a run |
| Wrong output, no probe | not possible for the Model M path **with** the probe: any wrong layout fails it (the probe is built to catch every lane/element/half permutation, wmma_probe.h:12-22, tests prove it). It is possible on any path **without** a probe (RR, see 2.3) | READ |

Best reading: refusal at create with a mismatch or PSO error, not a silent wrong picture, for Model M. A crash or TDR
is the one outcome the code cannot rule out. That risk is for whoever runs the first test, not for players, as long as
the name gate stays closed.

### 2.3 Ray Reconstruction start-up (differences)

1. Adapter gate with spoof-proof PCI id fallback (rr_runtime.cpp:1344, 2 above).
2. Compute queue on the game's device, `DoublePrecisionFloatShaderOps` check (:1367-1375).
3. `make_root` (root signature with the AGS space), kernel table lookup, then PSOs for **two** kernels only:
   `K_SELFTEST` and `K_PASS` (:1381-1397). All other PSOs compile on a background thread in three passes (:1480-1520)
   after the self-test, so a compile error there shows up later as `compile_failed`, not at create.
4. **Self-test** `selftest()` (:1258-1297, kernel `rr_rt_selftest`, clean/rr_f/runtime/kernels/rr_runtime.hlsl:140-148):
   one wave, FP8 WMMA of `1.0 x 1.0 + 0.5` (every byte 0x38), expects every lane and element == 16.5, then the driver's
   F32->F16 conversion expects 0x4C204C20. Mismatch -> log "driver-channel self-test mismatch (fp8 WMMA / f16
   conversion)... RR not available", `WG_E_ADAPTER_NOT_SUPPORTED` (:1432-1437).
5. Weakness (INFERRED from the kernel): all operands are the **same value in every lane**, so this self-test cannot see
   a wrong fragment layout; it would pass on any device whose FP8 multiply works and whose layout differs. The Model M
   probe is the stricter of the two. A gfx11 port of RR would need the stricter probe.

## 3. Is there a run-time check that the GPU computes the right thing?

| Mechanism | Runs when | What it proves | Reusable on a second family? |
|---|---|---|---|
| `native_driver_probe` + `wmma_probe.h` | every Model M / K create on a game device (device_d3d12.cpp:218) | exact FP16 and FP8 16x16x16 multiply, exact fragment layouts, distinct expected value for every one of 256 results; costs one 4.5 KiB upload and one wave | The mechanism is generic. The **layout table** and the FP8 half are gfx12. Needs: a gfx11 operand/expected-value variant (the header's comment says fragments are "gfx12 wave32", wmma_probe.h:9-13) and an FP16-only mode. The CPU side (`first_mismatch`, expected values) is host-only code, testable on any machine |
| `rr_rt_selftest` (rr_runtime.cpp:1258) | every RR device set-up | the FP8 multiply and the F32->F16 conversion give 16.5 / 0x4C20 | weak (uniform operands, see 2.3 point 5) |
| ELF/kernel SHA-256 check | each PSO create (device_d3d12.cpp:1650-1652) | the DXIL file is the one in the manifest | independent of GPU |
| `clean/dx12/compare.py` | test only, by hand | **byte-for-byte** HIP vs D3D12 on synthetic frames: every named runtime buffer after every frame, and that the listed kernels really ran on D3D12 (compare.py:2-12) | The HIP side is the reference, written for gfx12 code objects. For RDNA 3 the reference would have to be a CPU reference or the emulator (clean/kernels/emu, `-DWG_EMU` host build, wg_platform.h:5-6) |
| `clean/dx12/parity_gate.py` | test only | **speed**, not correctness: native/HIP wall-time ratio, with GPU-activity qualification. Exit 0/1/2 (parity_gate.py:2-6) | needs a second GPU with a HIP baseline; there is none on gfx11 |
| `clean/dx12/tests/*` (native_*_check.py, tuned_*_check.py, matched_stage_*) | test only, on a developer machine | stage-by-stage equality HIP vs D3D12 and the tuned variants | same HIP-as-reference limit |
| `clean/equiv` + clean/EQUIVALENCE-PROTOCOL.md | spec team's machine only | our model vs the original (MATCHES / CLOSE / DIFFERENT) | outside this repo's reach (needs the original) |
| `clean/linux/run_wine_tests.py ags-layout` | developer, Linux | channel register semantics vs a CPU model, 7 cases, gfx12 specific (:176, :262) | would need a gfx11 CPU model |
| CI | every PR | CPU only, mocks (mock HIP says `gfx1201`) | none |

To run the existing proof on a second GPU family you would need, at minimum: (1) a build of the reference path for
that family, since "HIP vs D3D12 equal" assumes HIP kernels exist for it (they do not for gfx11: the platform layer
calls `__builtin_amdgcn_wmma_*_gfx12`, clean/kernels/wg_platform.h:48-50; see lane 1); (2) a probe with that family's
layouts; (3) a GPU-bearing runner. Today there is no GPU CI at all (.github/workflows/ci.yml:2); the maintainer runs the
GPU tests (CONTRIBUTING.md, BUILDING.md section 7: "Run them on an RX 9000-series card; `WG_GPU_LOCK` can serialise
them").

## 4. Per-GPU tuning data that is RDNA 4 specific

### 4.1 Build-time target
- clean/runner/build.py:149 `ARCH = WG_GPU_ARCH` default `gfx1201`; :152 `EXTRA_ARCHS` (gfx1200 added by
  `--extra-archs`); :481 refuses any other target. Docs say the same: BUILDING.md:16, :63-65.
- clean/dx12/build_dx12.py:289-311: RGA is always run with `-c gfx1201` (ISA and the VGPR check).
- clean/kernels/wg_platform.h:23 `WG_WAVE 32`, :48-50 gfx12 builtins.

### 4.2 clean/dx12/vgpr_limits.json
- 20 entries (the 5 `l_swin_block_c32_*` Model L ones at 96, all others at 192). The `_doc` field derives them from
  RDNA 4 numbers: "VGPR file 1536 per SIMD lane, granule 8, a 256-lane wave32 workgroup = 4 waves on each of a CU's 2
  SIMDs, so 2 workgroups per CU need <= 192, 3 need <= 128" (RGA gfx1201). A kernel over its bound fails the build
  (build_dx12.py:326-343). READ.
- RDNA 3 differs in register-file size by die and in dual-issue/occupancy rules. I did not verify the numbers, so the
  1536 / 192 / 128 limits are UNKNOWN for gfx1100. The test is an RGA run with `-c gfx1100` on every Swin kernel.
  A second file or a per-arch key would be needed; the limits file has no arch key today.

### 4.3 clean/dx12/kernel_sets.json
- 428 lines, three keys: `model_m`, `model_l`, `model_k` (test_kernel_sets.py:4-10, :24-26). It lists
  (kernel name, code object) pairs per **model**, not per GPU. Nothing in it is RDNA 4 specific by content, but it has
  no place for a second architecture either (the `.co` names are architecture-neutral; the arch is picked from the
  folder or `gfx....` prefix at load, device.cpp:519-526).

### 4.4 Constants tied to one card
| file:line | Constant | Basis |
|---|---|---|
| clean/runtime/runtime.cpp:174-176 | `kTfSlots = 128`, `kTfTailPairs = 36`: "about one round of resident workgroups: kTfSlots = 4 per WGP of an RX 9070 XT" | RX 9070 XT (INFERRED 32 WGPs x 4). Tail-shortening only; a smaller or larger GPU gets a worse but correct plan. 7900 XT (84 CU) and 7600 would want their own number |
| clean/runtime/runtime.cpp:1039-1041 | post kernels default to wave64: "bit-identical, faster on gfx1201"; `WG_POST_WAVE=32` falls back | measured on gfx1201 only; UNKNOWN on gfx11 |
| clean/runtime/runtime.cpp (tf4_supported, `kTf4Cap = 448`, comment: D3D12 32 KB LDS vs 41472 bytes) | LDS ceilings | D3D12's own 32 KB limit, so GPU-independent in principle |
| clean/PERF.md:10, :16, :75 | "645 GB/s", "372 TFLOPS", occupancy numbers | RX 9070 XT's figures. Evidence for choices, not read by code |
| clean/dx12/README.md:55-70 "Driver exceptions" | compiler behaviours seen on the **gfx1201 D3D12 shader compiler** (regrouping, FMA fusion, dropped stores, loop unrolling) | all measured on one compiler. Another family's compiler needs its own pass over every port |
| clean/k/README.md, clean/FORMATS.md | many "on the RX 9070 XT" validation claims; hardware reciprocal `rcp`/`rsq` choices "on gfx1201" (clean/k/README.md:396, :440, :454) | arch-specific numeric behaviour; matters for bit-exactness |

### 4.5 Memory budgets (VRAM)
- Model M D3D12 arena: default **2560 MiB**, settable 16..4095 MiB with `WG_D3D12_ARENA_MB` (device_d3d12.cpp:191-192).
  In a game the executor measures the real need first and then sizes the arena "of exactly that size"; if the network
  does not fit 2560 MiB it re-measures in a 4095 MiB arena (`kMeasureMaxBytes`, executor_d3d12.cpp:55, :118-152). So
  the arena is **model-sized, not card-sized**. READ.
- **No code reads the card's VRAM** to decide anything. Grep for `DedicatedVideoMemory` / `QueryVideoMemoryInfo` finds
  it only in logs.cpp:218 (support bundle) and in tests (clean/fsr/tests, rr_f, interop tests, e.g.
  clean/rr_f/runtime/tests/rr_test_host.cpp:1035 which watches growth). READ.
- Cards: 9070 XT is 16 GB, 9060 XT 16 or 8 GB; the code comment at clean/installer/src/app/ui_checks.cpp:41-43 says 8 GB
  9060 cards are untested. For a 7900 XT (20 GB) the budget is not the limit. For RX 7600 (8 GB) it is an open
  question: the arena is up to 2.5-4 GiB plus the game, RR's own work area and the model. UNKNOWN; measure
  `arena_high_water` and RR's work area, then compare with 8 GB.
- HIP path: about 300 MiB of HIP's own memory per instance (MEASURED-IN-REPO on an RX 9070 XT, clean/runtime/device.cpp:57-60).
- RR: `WG_RR_*` arena modes and a "+3 GiB stop" in a test (rr_test_host.cpp:1035) show RR's budget is also model-sized.

### 4.6 Recorded baselines (all one card)
- clean/l/records/cp2077-model-l-timing-2026-11-14.json (adapter "RX 9070 XT"), clean/l/SPEED.md:20 (RX 9070 XT, driver
  32.0.32015.7029), clean/fg/records/net1-tierb-2026-11-14.json and net1-fused-2026-11-14.json (gfx1201), README.md:65
  ("1.17 ms Model M, 2.25 ms Model L, 1080p Quality, RX 9070 XT"), clean/installer/RELEASE-NOTES.md:98.
  All are MEASURED-IN-REPO on the RX 9070 XT. There are no records for any other card, and no per-GPU timing table
  that code reads. The speed-gate files (parity_gate.py) compare two paths on the **same** card.
- The records contain a gfx1201 WMMA probe result, `clean/fg/records/wmma-probe-2026-11-14.json`, from the HIP side.
  It is the only on-disk precedent for recording a probe per architecture.

## 5. User-facing strings and docs that need a second hardware family

Counts from `grep -E "RX ?9...|RDNA ?4|gfx12|9000-series"` (my count; approximate, includes comments).

**Strings the player or a log reader can see (code):**
- Launcher / installer UI and checks, 7 files:
  - clean/installer/src/wg_core/checks.cpp: 12 lines (GPU row texts :537-538, :550-554; the comments)
  - clean/installer/src/app/ui_checks.cpp (3: first-run card line, "Direct3D 12: Not available on this card")
  - clean/installer/src/app/wizard.cpp (4: welcome sub-titles :116, :118; "Radeon RX 9000" :655; requirement :660)
  - clean/installer/src/app/ui_words.cpp:87 (the player-facing translation of the check text; matches the English
    string "RX 9000-series" so **it would silently stop translating if the check text changed**. READ)
  - clean/installer/src/app/ui_game.cpp:1356 ("Auto uses Direct3D 12 on RX 9000-series cards...")
  - clean/installer/src/wg_core/profile.cpp:19 (same sentence in the profile text)
  - clean/installer/src/wg_core/rrprep.cpp:270 (+ rrprep.h 2)
  - clean/installer/src/app/launcher_ui.cpp:1101 (status line "· Direct3D 12 / · HIP / · Not supported")
- Runtime log and error lines: executor_select.h (6), device_d3d12.cpp (3), device.cpp:406, rr_runtime.cpp (13, incl.
  :1348-1349, :1371, :1433-1434), proxy_core.cpp:435-437, backend.cpp (7).
- clean/fsr/src/config.cpp:554 and config.h:92: the default settings file comment mentions "RX 9000".
- Roughly **30 code files** mention RDNA 4 / RX 9000 by name; about **60 lines** are text a user, a log reader or a
  support request would see.

**Docs (files with at least one mention):** README.md (7), COMPATIBILITY.md (3), BUILDING.md (7), RELEASING.md (1),
ROADMAP.md (2, including the section "More hardware: RDNA 3", ROADMAP.md:24-28), CONTRIBUTING.md (2), clean/README.md (7),
clean/PERF.md (2), clean/installer/README.md (8), clean/installer/RELEASE-NOTES.md (6), clean/linux/README.md (3),
clean/dx12/README.md (1), clean/rr_f/README.md (2) plus rr_f/runtime/README.md, rr_f/gpu/README.md (3),
clean/k/README.md (14), clean/k/FORMATS.md (2), clean/FORMATS.md (6), clean/fg/README.md, clean/l/SPEED.md,
docs/TESTING-OTHER-GAMES.md, docs/CLEAN-ROOM.md (2), THIRD-PARTY-NOTICES (2), the issue template. About 27 files.
Most are measurements ("on the RX 9070 XT") that are true as records and would stay.
The ones that change meaning with a second family: README.md:3, :6, :23, :44, :109; COMPATIBILITY.md:9, :36, :56;
BUILDING.md:16, :63-65, :203; RELEASING.md:26 ("GPU tests pass on a Radeon RX 9000-series card"); the installer README
lines about `kSupportedArch` (clean/installer/README.md:187-188 still says gfx1200 is "reported as not supported yet",
which contradicts checks.cpp:124 where gfx1200 is supported: stale text, READ).
README.md:23 currently states "older Radeon cards are not supported".

Tests that hard-code the RDNA 4 list (would need rows added): clean/runtime/interop/tests/test_executor_select.cpp
(about 25 lines, 40-48 and onward), clean/tests/test_device_targets.cpp:42-44, clean/tests/test_build_targets.py (both
archs hard-coded), clean/installer/tests/wg_core_tests.cpp and run_installer_tests.py (mock arch gfx1201;
`mock_hip.c`), clean/installer/tests/ui_fixture.ini, clean/dx12/tests/native_probe_test.cpp:21-56 (uses
`rdna4_adapter`, prints "SKIP no Radeon RX 9070 / 9060 adapter"), clean/runtime/tests/test_wmma_probe.cpp:218-222.

## 6. Model / data selection by GPU

READ: **nothing in the repo selects a kernel set, model or tuning variant by GPU.** What exists is selection by model
(M / L / K), by route (hybrid backend 2, native backend 3, HIP), by option, and by build variant (`__orig`,
`__pack12`, `w64`). The GPU is only a yes/no gate. Natural seams for an RDNA 3 set, from smallest to largest:

1. **The predicate**: `wgrt::rdna4_adapter` (clean/runtime/rdna4.h:16-19). One function used by
   device_d3d12.cpp:1367 and :1440, executor_select.h:105, tests/native_probe_test.cpp:48, and *copied* in
   rr_runtime.cpp:185 and by name in the installer (checks.cpp:30, :74, ui_render.cpp:58). Replacing the bool with a
   small "GPU family" value (rdna4 / rdna3 / unsupported) is the single point to widen. The copies must follow
   (6 sites). The installer, runtime and RR runtime currently disagree on workstation boards (PRO R97/R96), so unify
   as part of this.
2. **The kernel manifest**: `kernels_` is a `multimap` keyed by kernel name with a `co` tag and `variant_matches` /
   `variant_loaded` (device_d3d12.cpp:~621-640, `read_manifest` :1237+; "one port per (name, code object)"). The
   machinery for "several ports of one name, the runtime picks one" already exists (post_w64.co vs
   post_w64__orig.co). A per-family manifest folder, e.g. `dx12/gfx11/manifest.json`, chosen by the family value from
   (1), is the natural place. The runtime reads `build_dir + "/dx12/manifest.json"` at init (device_d3d12.cpp:197).
3. **Kernel choice in the runtime**: `choose_kernels` (clean/runtime/runtime.cpp:936-1040) picks code objects and
   variants (wave 32/64, p12, filter kernel, tuned/untuned) from options and `c.*` fields. A family check here would
   select the no-FP8 variants. `WgRtConfig` has `d_row[16]` (wg_runtime.h:39) and `post_wave` already, so config
   fields for layout exist.
4. **Packaging**: `kernel_sets.json` and `build_dx12.py kernel_set()` (the lists), `vgpr_limits.json` (add an arch
   key), `build.py` `ARCH`/`EXTRA_ARCHS` and `write_embedded` (the embedded table is `"<arch>.<name>"` already,
   test_build_targets.py:20-25).
5. **Model data**: the converter writes Model M weights as FP8 E4M3 plus packed tables (see lane 1/3). A model variant
   for a no-FP8 card (FP16 weights) would be a different packed format, and nothing selects one today
   (clean/runtime/model_select.h only chooses between models M/K/L by option).

## 7. What a first RDNA 3 experiment could do without touching the gates

Only reading advice; I ran nothing. The smallest honest experiment is: build `native_probe_test.cpp` with the name rule
replaced, run `native_driver_probe` alone on the 7900 XT, and record the outcome (PSO result, 512 floats, device state).
That one run answers the first row of 2.2 and decides whether FP16-via-channel exists on RDNA 3 at all.

## Open questions

1. Does the RDNA 3 D3D12 driver recognise register space 0x7FFF0ADE and the encoded `InterlockedCompareExchange`
   opcodes at all? (If not, the whole channel approach is closed on RDNA 3 and the question moves to plain HLSL
   cooperative/wave-matrix features.)
2. If it does, what does it map FP16 (format 0x5) to, and with which fragment layout? Same registers as gfx12, or the
   gfx11 duplicated-lane layout?
3. Does RDNA 3's D3D12 driver report `DoublePrecisionFloatShaderOps`? RR and Model K's exact input stage need it
   (rr_runtime.cpp:1367, device_d3d12.cpp:210-215).
4. Is the PCI device id of the 7900 XT 0x744C? The repo has no RDNA 3 id; it also has no RDNA 4 id outside the RR
   runtime and installer (the Model M runtime matches on the name only). Which identification method should win
   (name, id, HIP arch)?
5. Should the installer's `IsRdna4Name` ("RX 90") and the runtime's "RX 9070/9060" match be unified before adding a
   family? Today workstation RDNA 4 boards pass one and fail the other.
6. VRAM: what is the real peak of the Model M arena plus RR on an 8 GB RX 7600? The code never checks VRAM.
7. Who runs the GPU proof? The repo has no GPU CI and the maintainer runs the GPU tests (CONTRIBUTING.md).
8. clean/installer/README.md:187-188 says gfx1200 is "not supported yet" while checks.cpp:124 supports it: which is true?
   (Almost certainly stale text; I only read it.)

## What to measure on an RX 7900 XT

1. Adapter facts: DXGI description string, vendor/device id (support bundle `logs.cpp:~218` records them), driver
   version, `DedicatedVideoMemory`, HIP `gcnArchName` (expect gfx1100), HIP device name.
2. `D3D12_FEATURE_SHADER_MODEL` highest model (needs >= 6.6), `D3D12_FEATURE_D3D12_OPTIONS.DoublePrecisionFloatShaderOps`,
   wave size range (`WaveLaneCountMin/Max`), `D3D12_FEATURE_D3D12_OPTIONS1`, and whether native 16-bit types are exposed.
3. `native_driver_probe` alone (gate bypassed in a test build): PSO create result and time; the 512 floats; whether
   FP16 results match a gfx12 layout, a gfx11 layout, or neither; whether the FP8 half errors, returns zeros or garbage;
   whether anything resets the device. Repeat with the FP8 call removed.
4. The same with `rr_rt_selftest` to see if the weak uniform-operand test passes or fails.
5. RGA (`-s dx12 -c gfx1100`) on every Swin kernel: USED_VGPRs against the 192/128/96 bounds in `vgpr_limits.json`,
   to find which bounds hold for gfx11. Also check for scratch spills and the 32 KB LDS limit.
6. wave32 vs wave64 post kernels and the `kTfSlots` effect (the card has 42 WGPs, 84 CUs): one timing sweep.
7. Peak VRAM of the arena (`arena_high_water`) and of RR's work area at 1440p Quality, to size 8 GB and 20 GB cards.
8. Whether the AMD Adrenalin driver exposes `amdhip64_7.dll` on this card, and whether `HipProbe` reports gfx1100 as
   `exit 14` (it should), so the launcher's message is correct before any RDNA 3 work lands.
9. The launcher on a clean Windows install: first-run page, toolbar status ("· HIP" wording), `/FORCE` behaviour,
   to confirm the sections 1.1 rows.
