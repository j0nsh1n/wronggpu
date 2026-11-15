"""Tables from results/e2e-M.json (aggregate numbers only).   python report.py results"""
import json
import math
import sys
from collections import defaultdict

d = json.load(open(sys.argv[1] + "/e2e-M.json"))
cases = list(d["cases"])
runs = list(d["cases"][cases[0]]["runs"])
FINAL = ["dec0.h", "dec0.A", "dec0.B", "dec0.C", "dec0.latent", "post.display"]
print(f"### Model M end to end (render {d['render'][0]}x{d['render'][1]}, probe cases: {', '.join(cases)})\n")
print("Self-check (re-composed network == repo reference, every stage, every case):",
      all(all(v.values()) for v in (d["cases"][c]["self_check"] for c in cases)), "\n")
print("| Run | display PSNR dB (min / mean over cases) | display max abs | network-output SNR dB (mean) |"
      " final outputs (chained), worst verdict | display equal share | stages DIFFERENT isolated (of 19) |"
      " isolated equal share, mean over stages |")
print("|---|---|---|---|---|---|---|---|")
for r in runs:
    ps = [d["cases"][c]["runs"][r]["image"]["psnr_db"] for c in cases]
    mx = max(d["cases"][c]["runs"][r]["image"]["max_abs"] for c in cases)
    snr = [d["cases"][c]["runs"][r]["net_snr_db"] for c in cases]
    order = {"MATCHES": 0, "CLOSE": 1, "DIFFERENT": 2, "ERROR": 3}
    worst = max((d["cases"][c]["runs"][r][k]["verdict"] for c in cases for k in FINAL), key=order.get)
    eq = sum(d["cases"][c]["runs"][r]["post.display"]["share_equal"] for c in cases) / len(cases)
    nd = max(sum(v == "DIFFERENT" for v in d["cases"][c]["runs"][r]["isolated"].values()) for c in cases)
    ie = [sum(d["cases"][c]["runs"][r]["isolated_equal"].values()) / len(d["cases"][c]["runs"][r]["isolated_equal"])
          for c in cases]
    f = lambda v: "inf" if math.isinf(v) else f"{v:.1f}"  # noqa: E731
    print(f"| {r} | {f(min(ps))} / {f(sum(ps) / len(ps)) if not any(map(math.isinf, ps)) else 'inf'} | {mx:.3g} | "
          f"{f(sum(snr) / len(snr)) if min(snr) < 1000 else 'exact'} | {worst} | {100 * eq:.1f} % | {nd} | "
          f"{100 * sum(ie) / len(ie):.1f} % |")

print("\nPer stage, chained, share of values equal (mean over cases):\n")
stages = [k for k in d["cases"][cases[0]]["runs"][runs[0]] if isinstance(d["cases"][cases[0]]["runs"][runs[0]][k], dict)
          and "share_equal" in d["cases"][cases[0]]["runs"][runs[0]][k]]
print("| Run | " + " | ".join(stages) + " |")
print("|---|" + "---|" * len(stages))
for r in runs:
    print(f"| {r} | " + " | ".join(
        f"{100 * sum(d['cases'][c]['runs'][r][k]['share_equal'] for c in cases) / len(cases):.0f}" for k in stages) + " |")

# activations by site kind
kind = defaultdict(lambda: {"n": 0, "tile": [0] * 17, "tok": [0] * 17, "tspan": -1, "err": defaultdict(lambda: [0, 0, 0, 0]),
                            "wmma": [0, 0, 0], "ovf": [0, 0], "maxbits": 0, "nonneg": True, "stages": 0})
for site, s in d["activations"].items():
    k = site.split(".", 1)[1]
    k = {"W": site.split(".")[0] + ".W"}.get(k, k)
    a = kind[k]
    a["stages"] += 1
    a["n"] += s["n"]
    a["tile"] = [x + y for x, y in zip(a["tile"], s["tile_spans"])]
    a["tok"] = [x + y for x, y in zip(a["tok"], s["token_spans"])]
    a["tspan"] = max(a["tspan"], s["tensor_span"])
    a["nonneg"] = a["nonneg"] and s["nonneg"]
    for q, v in s["err"].items():
        a["err"][q] = [x + y for x, y in zip(a["err"][q], v)]
    a["wmma"] = [a["wmma"][0] + s["wmma"]["base"], a["wmma"][1] + s["wmma"]["tile"], a["wmma"][2] + s["wmma"]["row"]]
    a["ovf"] = [a["ovf"][0] + s["ovf"][0], a["ovf"][1] + s["ovf"][1]]
    a["maxbits"] = max(a["maxbits"], s["maxbits"])
print("\n### Activations feeding a multiply (all cases, all stages per site kind)\n")
print("| Site (operand) | values | 16x16 tiles span <=3 / <=10 | tokens (full K) span <=3 / <=10 | max tensor span |"
      " rel RMS: per tensor / per token / tile 1 plane / tile 2 planes | nonzero -> 0: tensor / token / tile1 |")
print("|---|---|---|---|---|---|---|")
tot = [0, 0, 0]
for k, a in kind.items():
    tl, tk = a["tile"], a["tok"]
    sh = lambda h, m: 100 * sum(h[:m + 1]) / max(1, sum(h))  # noqa: E731
    e = a["err"]
    rr = lambda q: (e[q][0] / e[q][1]) ** 0.5 if e[q][1] else 0  # noqa: E731
    zz = lambda q: 100 * e[q][2] / max(1, e[q][3])  # noqa: E731
    print(f"| {k}{' (>=0)' if a['nonneg'] else ''} | {a['n']} | {sh(tl, 3):.0f} / {sh(tl, 10):.0f} % | "
          f"{sh(tk, 3):.0f} / {sh(tk, 10):.0f} % | {a['tspan']} | "
          + " / ".join(f"{rr(q):.1e}" for q in ("tensor", "token", "tile1", "tile2")) + " | "
          + " / ".join(f"{zz(q):.1f}" for q in ("tensor", "token", "tile1")) + " % |")
print("\n### Exact adaptive digit planes (step iv): WMMAs per original WMMA\n")
print("| Site | per-16x16-block scales (needs per-K-tile rescale) | per-row scales over the whole K (free) |"
      " outputs whose bound sum|a'||w'| >= 2^31 (row scales) | max bound bits |")
print("|---|---|---|---|---|")
for k, a in kind.items():
    b, t, r = a["wmma"]
    if b:
        tot = [tot[0] + b, tot[1] + t, tot[2] + r]
        print(f"| {k} | {t / b:.2f} | {r / b:.2f} | {100 * a['ovf'][0] / max(1, a['ovf'][1]):.2f} % | {a['maxbits']} |")
print(f"| **all (weighted by WMMA count; S sampled 1/64 of windows)** | **{tot[1] / tot[0]:.2f}** | "
      f"**{tot[2] / tot[0]:.2f}** | | |")
