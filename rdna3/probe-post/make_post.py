"""Assemble POST.md from results.json (tables) plus fixed text. Run: python3 -I make_post.py"""
import json
R = json.load(open("results.json"))
g = lambda *k: R.get("|".join(map(str, k)), [0] * 9)
K = {"e4_random": [0], "p1b": list(range(8)), "adv": list(range(8))}
pm = lambda c, n: f"{c / n * 1e6:,.0f}" if n else "-"


def v(cls, kinds, dtn, pair):
    return sum(g(cls, k, dtn, "n")[0] for k in kinds), [sum(g(cls, k, dtn, pair)[i] for k in kinds) for i in range(9)]


def row(name, cls, kinds, dtn, pair):
    n, x = v(cls, kinds, dtn, pair)
    if not n:
        return f"| {name} | 0 | | | | |"
    return (f"| {name} | {n:,} | {x[1]:,} ({pm(x[1], n)}) | {x[2]:,} / {x[3] + x[4]:,} | {x[5]:,} ({pm(x[5], n)}) | "
            f"{x[6]:,} / {x[7] + x[8]:,} |")


H = "| data | outputs | q8 differs: count (per M) | 1 code / 2+ | RN16 differs: count (per M) | 1 step / 2+ |\n|---|---|---|---|---|---|"
L = []
w = L.append


def tbl(rows):
    w(H)
    L.extend(rows)
    w("")


def bins(cls, kinds, b):
    key = lambda kk: g(cls, kk, "bin", b)
    x = [sum(key(kk)[i] for kk in kinds) for i in range(9)]
    return x


names = ["<=0", "1", "2", "3", "4-5", "6-8", "9-12", "13-16", ">16", "exact 0", "all"]
w("# RDNA 3 F16 WMMA: how often do its FP32 errors change a quantized value?")
w("")
w("CPU-only analysis of the saved RX 7900 XT probe outputs (`probe-vk/results/*.npz`). MEASURED = counted here from those files;")
w("INFERRED = reasoning. Script: `post_quant.py` (compute, 3 min on one core), `render.py` / `make_post.py` (tables), `selfcheck.py`.")
w("")
w("## Headline (MEASURED)")
n, a = v("p1b", K["p1b"], "f32", "gpu~exact")
_, b = v("p1b", K["p1b"], "f32", "gpu~alt")
ne, ae = v("e4_random", [0], "f32", "gpu~exact")
w(f"- F32-accumulate, p1b data: **{pm(a[1], n)} per million outputs** change their E4M3 code vs the exact sum ({a[1]:,} of {n:,}; "
  f"{a[2]:,} of them by one code) and **{pm(a[5], n)} per million** change their FP16 value. Across the 8 data modes: 4 to 1,030 (q8), 259 to 3,345 (RN16).")
w(f"- Against M-g8 (RDNA 4 fit) the rates are the same: {pm(b[1], n)} (q8), {pm(b[5], n)} (RN16). M-g8 and the exact sum themselves differ by only 9 (q8) / 66 (RN16) per million, so this card vs RDNA 4 is about the same question.")
w(f"- Uniform-random E4M3 (e4_random): {pm(ae[1], ne)} per million (q8; {ae[1]:,} of {ne:,}), {pm(ae[5], ne)} (RN16). q8 changes are one code in 99 % of cases (p1b 941 of 945); RN16 changes are one step in 95 %.")
w("- F16-accumulate instruction: 1.75e5 per million of p1b outputs differ from RN16(exact) (4.2e5 for e4_random), but most of that is the instruction rounding after every pair of products (M-g2), not the artefact; vs M-g2 it is 5.0e4 (p1b).")
w("- The rate is set by how many results sit near a rounding boundary, so it depends on cancellation depth and on exact ties (section 2).")
w("")
w("Method: D_exact = RN32(C + exact sum) from `exact.py` (float64 two-sum, equal to the Python-int backend on a 16,384-output check); D_model likewise (`models.py`).")
w("q8 = `fp8.q8` (RNE, saturate 448); RN16 = float16 RNE. A *flip* = quantized values compare unequal (+0 and -0 count as equal; they differ in code only on 54 of 16,384 sampled adv outputs).")
w("Size = distance in E4M3 codes / FP16 steps. Non-finite inputs and FP16 overflow excluded. Not measured: a bias add or activation before the quantizer.")
w("")
w("## 1. Flip rates, F32-accumulate (`v_wmma_f32_16x16x16_f16`)")
w("")
w("#### A. 7900 XT vs exact sum (modes: 0,1,2,5 sparse full-range; 3 dense full-range; 4 large exponents; 6 sparse small; 7 dense small with wide C)")
w("")
rows = [row("e4_random", "e4_random", [0], "f32", "gpu~exact"), row("**p1b all modes**", "p1b", K["p1b"], "f32", "gpu~exact")]
rows += [row(f"p1b mode {k}", "p1b", [k], "f32", "gpu~exact") for k in range(8)]
rows += [row("**adv all**", "adv", K["adv"], "f32", "gpu~exact")]
rows += [row(f"adv sub-kind {k}", "adv", [k], "f32", "gpu~exact") for k in (0, 1, 5, 7)]
tbl(rows)
w("adv sub-kinds 2, 3, 4, 6 (ties, big C, near overflow): 0 q8 flips; sub-kind 0 (C cancels the big product) flips 22 % of outputs. adv is a stress set, not real data.")
w("")
w("#### B. 7900 XT vs M-g8 (stand-in for RX 9070 XT) and C. exact vs M-g8 (no 7900 XT)")
w("")
tbl([row("B e4_random", "e4_random", [0], "f32", "gpu~alt"), row("B p1b all", "p1b", K["p1b"], "f32", "gpu~alt"), row("B adv all", "adv", K["adv"], "f32", "gpu~alt"),
     row("C e4_random", "e4_random", [0], "f32", "exact~alt"), row("C p1b all", "p1b", K["p1b"], "f32", "exact~alt"), row("C adv all", "adv", K["adv"], "f32", "exact~alt")])
w("Per-mode versions of B and C are in `fragments.md`. For p1b the 7900 XT sits no closer to M-g8 than to the exact sum.")
w("")
w("## 2. Why that rate: cancellation depth and ties (MEASURED)")
w("")
w("Depth d = exponent of the largest of the 17 terms minus exponent of the exact result. Error is about u x 2^(E-24) with u = 1.0-1.4 (mean) once terms have mixed signs.")
w("")
w("| depth | e4 share | e4 q8/M | e4 RN16/M | p1b share | p1b q8/M | p1b RN16/M | adv share | adv q8/M | adv RN16/M |\n|---|---|---|---|---|---|---|---|---|---|")
tots = {c: bins(c, K[c], 10)[0] for c in K}
for b in range(10):
    cells = []
    for c in ("e4_random", "p1b", "adv"):
        x = bins(c, K[c], b)
        cells += [f"{x[0] / tots[c] * 100:.1f} %", pm(x[2], x[0]), pm(x[3], x[0])] if x[0] else ["0", "-", "-"]
    w(f"| {names[b]} | " + " | ".join(cells) + " |")
w("| all | 100 % | 2 | 212 | 100 % | 451 | 1,472 | 100 % | 33,988 | 164,046 |")
w("")
w("Two mechanisms, both visible in the table:")
w("1. **Away from ties** (random position between two grid values): flip probability = error / grid spacing = u x 2^(d-21) for E4M3 (3 mantissa bits) and u x 2^(d-14) for FP16 (10 bits).")
e = [bins("e4_random", [0], b) for b in range(11)]
chk = []
for b, d in ((5, 7), (6, 10.5), (7, 14.5)):
    x = e[b]
    chk.append(f"d={d}: formula {1.19 * 2 ** (d - 21) * 1e6:,.0f} vs measured non-tie {pm(x[2] - x[5], x[0] - x[4])} per M")
w("   Check on e4_random (q8, u = 1.19): " + "; ".join(chk) + ". Little cancellation (d <= 3) gives 0-8 per M; 10 binades of cancellation gives about 1,000 per M.")
w("2. **Exact ties**: E4M3 products have few mantissa bits, so with sparse or low-precision data the exact sum often lies exactly halfway between two q8 values. A tie flips only when the GPU error is nonzero and points the wrong way: 6 % of p1b ties, 25 % of e4_random, 28 % of adv (more at larger depth)")
p = bins("p1b", K["p1b"], 10)
w(f"   (p1b: {p[4]:,} of {p[0]:,} outputs are q8 ties, {p[5] / p[4] * 100:.1f} % of them flip; {p[5]:,} of the {p[2]:,} q8 flips are at ties). Random E4M3 has almost no ties ({pm(e[10][4], e[10][0])} per M), so its rate is tiny.")
w("")
w("Predicting for other data: rate_q8 = sum over depth of share(d) x u x 2^(d-21), plus tie_fraction x tie_flip_rate (6-28 % above), and the same with 2^(d-14) for RN16. RN16 ties are also common (p1b: 2,525 of 3,087 flips).")
w("p1b has 85 % of outputs at d <= 0, which is why its rate is dominated by ties; adv has 14 % at d > 16, which is why it flips 3-4 % of outputs.")
w("")
w("## 3. F16-accumulate (`v_wmma_f16_16x16x16_f16`, C = RN16(C)) (MEASURED)")
w("")
d_rows = lambda pair, tag: [row(f"{tag} e4_random", "e4_random", [0], "f16", pair), row(f"{tag} p1b all", "p1b", K["p1b"], "f16", pair),
                            row(f"{tag} p1b mode 0", "p1b", [0], "f16", pair), row(f"{tag} p1b mode 7", "p1b", [7], "f16", pair), row(f"{tag} adv all", "adv", K["adv"], "f16", pair)]
tbl(d_rows("gpu~exact", "vs RN16(exact)") + d_rows("gpu~alt", "vs M-g2"))
w("Outputs with an FP16 overflow in C or the result are excluded (e4_random 8.1 M, p1b 0.49 M, adv 0.50 M), so counts are on fewer outputs. In this table the first (q8) columns apply q8 to the FP16 result.")
w("Against RN16(exact) the card is far off because it rounds to FP16 after every pair of products; the part M-g2 does not explain is 6.1 % of e4_random outputs and 5.0 % of p1b.")
w("")
w("## 4. Scale-up (INFERRED, back of the envelope)")
w("")
w("- Differing codes per stage = N x r x 1e-6, with r = flips per million outputs. Per frame = sum over the stages: N_total x r x 1e-6.")
w("- Worked, N = 1e6 values per layer, p1b rates: q8 451 per M gives **about 450 codes per stage** (4 to 1,030 depending on the data mode); RN16 1,472 gives about 1,500 FP16 values. A 20-layer frame: about 9,000 q8 codes (range 80 to 21,000).")
w("- Rate 451 per M is 1 in 2,200 outputs; random E4M3 would be 2 per M (about 2 per stage). Real weights and activations are neither, so measure a real layer before relying on either number.")
w("- If a ReLU or a clamp zeroes about half the outputs, the count roughly halves (not measured). A bias add moves the sum off its tie, which would lower the tie share (not measured).")
w("- Cascade: a differing code is 1 step (6-12 % of the value for E4M3) in the next layer's input. It touches every output that reads it (fan-out F, for example K x K x channels), and each of those can flip with probability about weight x step / grid spacing.")
w("  If F x that probability is above 1 per stage the difference spreads; below 1 it dies out. Not measured: replay a real network on CPU with injected one-code errors to find which.")
w("- 1 in 2,200 is a different claim from \"bit identical\": the frame can still differ in a few thousand codes yet stay visually close, which a CLOSE verdict would allow and a byte-exact check would not.")
w("")
w("## Limits")
w("- Rates are for these probe data sets (uniform E4M3, P1b mimic, adversarial), 16x16x16 tiles, no bias/activation. The p1b rate varies 250x across modes, so treat 451 per M as a central guess, not a bound.")
w("- Exact reference is RN32 first, then the quantizer (the pipeline's order). Rounding the exact sum straight to E4M3 would change only the rare double-rounding cases (not counted).")
w("- Only 7900 XT outputs were available; the RX 9070 XT numbers use M-g8 as a proxy, which fit 2 of 51 unexplained samples.")
open("POST.md", "w").write("\n".join(L) + "\n")
