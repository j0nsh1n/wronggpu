"""Turn raw/time_*.txt into the tables of RESULTS-VALU.md (printed as Markdown; numbers also in logs/analysis.json).

Per kernel: fixed = 2 med t(N) - med t(2N) (launch, prologue, epilogue); per-rep slope_i = (t_i(2N) - fixed) / 2N for
the 10 reps at 2N (ns per loop iteration of the whole dispatch). The median slope equals (med t(2N) - med t(N)) / N;
min-max is the rep spread of the 2N run. SIMD-cycles per wave-iteration = slope x f / (waves per SIMD), with f fixed by CALIB
(128 VOP3 v_fma_f32 per wave-iteration, one per SIMD per cycle). Dispatch: 672 workgroups x 8 waves = 5376 waves = 32
per SIMD (84 CUs x 2 SIMDs).
"""
import glob
import json
import re
import statistics as st
import sys

sys.path.insert(0, ".")
from disasm_loop import summarize  # noqa: E402

WAVES, SIMDS = 672 * 8, 168
WPS = WAVES / SIMDS


def reps(path):
    return [float(m.group(1)) for m in re.finditer(r"^rep \d+ gpu_ns (\d+)", open(path).read(), re.M)]


def kernel(k):
    files = {int(re.search(r"_(\d+)\.txt$", f).group(1)): f for f in glob.glob(f"raw/time_{k}_672_*.txt")}
    ns = sorted(files)
    n, n2 = ns[-2], ns[-1]
    assert n2 == 2 * n, (k, ns)
    a, b = reps(files[n]), reps(files[n2])
    big = reps(f"raw/time_{k}_1344_{n}.txt")
    fixed = 2 * st.median(a) - st.median(b)
    sl = [(y - fixed) / n2 for y in b]
    stats = open(f"disasm/{k}.stats").read()
    occ = int(re.search(r"Subgroups per SIMD: (\d+)", stats).group(1))
    vgpr = int(re.search(r"stat VGPRs: (\d+)", stats).group(1))
    lds = int(re.search(r"LDS size: (\d+)", stats).group(1))
    _, cls, ops = summarize(f"disasm/{k}.txt")
    return dict(n=n, fixed=fixed, slope=st.median(sl), slope_min=min(sl), slope_max=max(sl), t2_med=st.median(b),
                t2_spread=(max(b) - min(b)) / st.median(b), lin_iters=st.median(b) / st.median(a),
                lin_wgs=st.median(big) / st.median(a), occ=occ, vgpr=vgpr, lds=lds,
                slots=cls["valu"] + cls["vopd"] + cls["wmma"], lds_ops=cls["lds"], vmem=cls["vmem"])


names = sorted({re.match(r"raw/time_(.+)_672_\d+\.txt", f).group(1) for f in glob.glob("raw/time_*_672_*.txt")})
R = {k: kernel(k) for k in names}
F = 128 * WPS / R["CALIB"]["slope"]  # GHz: cycles per ns
for k, r in R.items():
    r["cyc"] = r["slope"] * F / WPS
    r["cyc_min"], r["cyc_max"] = r["slope_min"] * F / WPS, r["slope_max"] * F / WPS
empty = R["EMPTY"]["cyc"]

clk = []
for f in glob.glob("raw/clk_*.txt"):
    for l in open(f):
        m = re.match(r"clk (\d+) busy (\d+)", l)
        if m and int(m.group(2)) >= 90:
            clk.append(int(m.group(1)) / 1e9)
clk.sort()
out = {"f_eff_GHz": F, "empty_cyc": empty, "sysfs_busy_GHz": [clk[0], st.median(clk), clk[-1], len(clk)], "k": R}
json.dump(out, open("logs/analysis.json", "w"), indent=1)

print(f"f_eff (CALIB) = {F:.3f} GHz; sysfs sclk while busy>=90%: median {st.median(clk):.3f}, range {clk[0]:.3f}-{clk[-1]:.3f} GHz ({len(clk)} samples)")
print(f"empty loop: {empty:.2f} SIMD-cycles per wave-iteration\n")
print("| kernel | N | body slots | occ | t(2N) med ms | spread | 2N/N | 2xWG/N | cycles/wave-iter med [min-max] |")
print("|---|---|---|---|---|---|---|---|---|")
for k, r in R.items():
    print(f"| {k} | {r['n']} | {r['slots']} | {r['occ']} | {r['t2_med']/1e6:.3f} | {100*r['t2_spread']:.1f}% | {r['lin_iters']:.3f} | "
          f"{r['lin_wgs']:.3f} | {r['cyc']:.1f} [{r['cyc_min']:.1f}-{r['cyc_max']:.1f}] |")

def row(k, per, base=None, unit_values=None):
    r = R[k]
    c, lo, hi = r["cyc"] / per, r["cyc_min"] / per, r["cyc_max"] / per
    return c, lo, hi


print("\n### A: cycles per 4096 products per SIMD (one wave-iteration), whole-card ns per 4096 products")
print("| kernel | cycles med [min-max] | minus empty loop | whole-card ns / 4096 |")
print("|---|---|---|---|")
for k in ["CALIB", "A_FMA32", "A_F16MIX", "A_F16DOT16", "A_F16PK", "A_I8DOT", "A_I8DOTSAT", "A_U8DOT", "A_I32MAD"]:
    r = R[k]
    print(f"| {k} | {r['cyc']:.1f} [{r['cyc_min']:.1f}-{r['cyc_max']:.1f}] | {r['cyc']-empty:.1f} | {r['slope']/WAVES:.4f} |")
for k in ["C_F16_W4", "C_I8_W4"]:
    r = R[k]
    print(f"| {k} (per WMMA) | {r['cyc']/4:.1f} [{r['cyc_min']/4:.1f}-{r['cyc_max']/4:.1f}] | {(r['cyc']-empty)/4:.1f} | {r['slope']/WAVES/4:.4f} |")

print("\n### B: per step = 8 values per lane (256 per wave); glue = kernel - B_BASE where the harness matches")
print("| kernel | cycles/step med [min-max] | glue cycles/step | slots/value | glue slots/value | whole-card ps/value |")
print("|---|---|---|---|---|---|")
bb = R["B_BASE"]
same = {"B_Q8_HW", "B_Q8_SAT", "B_F2H", "B_F2H_SW", "B_RQ_F32", "B_RQ_MAGIC", "B_GELU_LDS"}
for k in ["B_BASE", "B_E4H_BITS", "B_E4H_EXACT", "B_E4H_LDS", "B_Q8_HW", "B_Q8_SAT", "B_F2H", "B_F2H_SW", "B_RQ_F32",
          "B_RQ_MAGIC", "B_GELU_LDS", "B_XOR16", "B_BPERM", "B_W_GLOBAL", "B_W_LDS"]:
    r = R[k]
    g = f"{(r['cyc']-bb['cyc'])/4:.1f}" if k in same else "-"
    gs = f"{(r['slots']-bb['slots'])/32:.2f}" if k in same else "-"
    print(f"| {k} | {r['cyc']/4:.1f} [{r['cyc_min']/4:.1f}-{r['cyc_max']/4:.1f}] | {g} | {r['slots']/32:.2f} | {gs} | "
          f"{1000*r['slope']/(WAVES*32*32):.3f} |")

print("\n### C: overlap, cycles per wave-iteration")
print("| kernel | measured | sum of parts | max of parts |")
print("|---|---|---|---|")
for w, v, mix in [("C_F16_W4", "C_F16_V64", "C_F16_W4V64"), ("C_F16_W4", "C_F16_V128", "C_F16_W4V128"),
                  ("C_F16_W4", "C_F16_V256", "C_F16_W4V256"), ("C_I8_W4", "C_F16_V128", "C_I8_W4V128")]:
    a, b, m = R[w]["cyc"], R[v]["cyc"], R[mix]
    print(f"| {mix} | {m['cyc']:.1f} [{m['cyc_min']:.1f}-{m['cyc_max']:.1f}] | {a+b:.1f} | {max(a,b):.1f} |")
