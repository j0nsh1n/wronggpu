"""Turn results.json into markdown tables (fragments.md). Run: python3 -I render.py"""
import json

R = json.load(open("results.json"))
g = lambda *k: R.get("|".join(map(str, k)), [0] * 9)
K = {"e4_random": [0], "p1b": list(range(8)), "adv": list(range(8))}
PMF = lambda c, n: f"{c / n * 1e6:,.0f}" if n else "-"
out = []


def vec(cls, kinds, dtn, pair):
    n = sum(g(cls, k, dtn, "n")[0] for k in kinds)
    v = [sum(g(cls, k, dtn, pair)[i] for k in kinds) for i in range(9)]
    return n, v


def row(name, cls, kinds, dtn, pair):
    n, v = vec(cls, kinds, dtn, pair)
    if n == 0:
        return f"| {name} | 0 (all overflow FP16) | | | | | |"
    return (f"| {name} | {n:,} | {v[0] / n * 100:.1f} % | {v[1]:,} ({PMF(v[1], n)}) | {v[2]:,} / {v[3]:,} / {v[4]:,} | "
            f"{v[5]:,} ({PMF(v[5], n)}) | {v[6]:,} / {v[7]:,} / {v[8]:,} |")


HDR = ("| data | outputs | bits differ | q8 differ (per M) | q8: 1 code / 2 / 3+ | RN16 differ (per M) | RN16: 1 step / 2 / 3+ |\n"
       "|---|---|---|---|---|---|---|")


def block(title, dtn, pair, adv_sub=False):
    out.append(f"\n#### {title}\n\n{HDR}")
    out.append(row("e4_random", "e4_random", K["e4_random"], dtn, pair))
    out.append(row("p1b, all modes", "p1b", K["p1b"], dtn, pair))
    for k in range(8):
        out.append(row(f"p1b mode {k}", "p1b", [k], dtn, pair))
    out.append(row("adv, all", "adv", K["adv"], dtn, pair))
    if adv_sub:
        for k in range(8):
            out.append(row(f"adv sub-kind {k}", "adv", [k], dtn, pair))


block("A. 7900 XT F32-accumulate vs exact sum (RN32 first)", "f32", "gpu~exact", adv_sub=True)
block("B. 7900 XT F32-accumulate vs M-g8 (RDNA 4 fit)", "f32", "gpu~alt")
block("C. Reference: exact vs M-g8 (no 7900 XT involved)", "f32", "exact~alt")
block("D. 7900 XT F16-accumulate (D_f16) vs RN16 of exact sum", "f16", "gpu~exact")
block("E. 7900 XT F16-accumulate vs M-g2", "f16", "gpu~alt")
out.append("\nExcluded (FP16 overflow to inf, non-finite): " + ", ".join(
    f"{c} {sum(g(c, k, 'f16', 'n')[1] for k in K[c]):,}" for c in K))

# bins
def binrows(cls, kinds, title):
    out.append(f"\n#### {title}\n")
    out.append("| depth | outputs | share | GPU!=exact bits | mean err (units 2^(E-24)) | q8 flips (per M) | exact is a q8 tie | flips among ties | flips elsewhere | RN16 flips (per M) | RN16 flips among ties / elsewhere |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    names = ["<=0", "1", "2", "3", "4-5", "6-8", "9-12", "13-16", ">16", "exact 0", "all"]
    tot = sum(g(cls, k, "bin", 10)[0] for k in kinds if "|".join(map(str, (cls, k, "bin", 10))) in R)
    for b in range(11):
        v = [0] * 9
        for k in kinds:
            key = "|".join(map(str, (cls, k, "bin", b)))
            if key in R:
                v = [x + y for x, y in zip(v, R[key])]
        n = v[0]
        if not n:
            continue
        t8, f8t, t16, f16t = v[4], v[5], v[6], v[7]
        out.append(f"| {names[b]} | {n:,} | {n / tot * 100:.1f} % | {v[1] / n * 100:.1f} % | {v[8] / 1000 / n:.2f} | "
                   f"{v[2]:,} ({PMF(v[2], n)}) | {t8:,} ({t8 / n * 100:.1f} %) | {f8t:,} ({PMF(f8t, t8)}) | {v[2] - f8t:,} ({PMF(v[2] - f8t, n - t8)}) | "
                   f"{v[3]:,} ({PMF(v[3], n)}) | {f16t:,} / {v[3] - f16t:,} |")


binrows("e4_random", K["e4_random"], "e4_random")
binrows("p1b", K["p1b"], "p1b, all modes")
for k in (3, 4, 7):
    binrows("p1b", [k], f"p1b mode {k}")
binrows("adv", K["adv"], "adv, all")
open("fragments.md", "w").write("\n".join(out) + "\n")
