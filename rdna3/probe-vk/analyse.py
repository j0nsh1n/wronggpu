"""Analysis of the coopmat probe outputs (out/*.bin) against exact arithmetic and candidate models.

E4M3 classes use the float64 backend (exact, see exact.py); the full-range FP16 class uses the Python-int backend on
its first F16_FULL_N problems (it is slow)."""
import functools
import numpy as np

print = functools.partial(print, flush=True)

from exact import F64, INT
from models import MODELS

F16_FULL_N = 256
CHUNK = 2048


def load(c):
    return np.load(f"inputs/{c}.npz")


def dout(c, suffix, dt, rep=0):
    return np.fromfile(f"out/{c}.{suffix}.{rep}.bin", dt).reshape(-1, 16, 16)


def ordered(x):
    """Monotone integer image of float bits (for ULP distances)."""
    if x.dtype == np.float32:
        i = x.view(np.int32).astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFF), i)
    i = x.view(np.int16).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFF), i)


def same_bits(a, b):
    return (a.view(np.uint32 if a.dtype == np.float32 else np.uint16) == b.view(np.uint32 if b.dtype == np.float32 else np.uint16)) | (np.isnan(a) & np.isnan(b))


BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 16), (17, 1 << 62)]


def hist(d):
    d = np.abs(d)
    return [int(((d >= lo) & (d <= hi)).sum()) for lo, hi in BUCKETS]


def products(A, B):
    return A[:, :, None, :] * B.transpose(0, 2, 1)[:, None, :, :]      # [p, m, n, k]


def fit(c, D, dt, n=None, backend=F64, ctype=np.float32, models=MODELS):
    z = load(c)
    N = len(D) if n is None else n
    res = {name: [0, [0] * len(BUCKETS), 0] for name in models}
    for s in range(0, N, CHUNK):
        e = min(N, s + CHUNK)
        A = z["A"][s:e].astype(np.float64)
        B = z["B"][s:e].astype(np.float64)
        C = z["C"][s:e].astype(ctype).astype(np.float64)
        Dh = D[s:e]
        finite = np.isfinite(A).all((1, 2))[:, None, None] & np.isfinite(B).all((1, 2))[:, None, None] & np.isfinite(C)
        P = products(A, B)
        if backend is INT:
            P = INT.const(P)
            Cx = C
        else:
            Cx = C
        for name, f in models.items():
            M = f(P, Cx, backend, dt).astype(dt)
            ok = same_bits(M, Dh)
            fin = finite & np.isfinite(M)
            d = np.where(fin, ordered(Dh) - ordered(M), 0)
            res[name][0] += int((~ok & finite).sum())
            res[name][2] += int(finite.sum())
            h = hist(np.where(ok, 0, np.where(d == 0, 1, d))[finite])
            res[name][1] = [a + b for a, b in zip(res[name][1], h)]
    return res


def print_fit(title, res):
    print(f"\n### {title}")
    print("| model | mismatches | of | ulp 0 | 1 | 2 | 3-4 | 5-16 | >16 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, (bad, h, tot) in sorted(res.items(), key=lambda kv: kv[1][0]):
        print(f"| {name} | {bad} | {tot} | " + " | ".join(str(x) for x in h) + " |")


def determinism(c, suffix, dt):
    try:
        Ds = [dout(c, suffix, dt, r) for r in range(3)]
    except FileNotFoundError:
        return None
    return all(np.array_equal(Ds[0].view(np.uint8), D.view(np.uint8)) for D in Ds[1:])


def err_units(c, D, n):
    """|D - exact| in units of 2^(E-24), E = exponent of the largest |term| (products and C)."""
    z = load(c)
    A = z["A"][:n].astype(np.float64)
    B = z["B"][:n].astype(np.float64)
    C = z["C"][:n].astype(np.float64)
    P = products(A, B)
    T = np.concatenate([P, C[..., None]], -1)
    mx = np.abs(T).max(-1)
    E = np.frexp(np.where(mx > 0, mx, 1))[1] - 1
    ex = P.sum(-1) + C
    fin = np.isfinite(ex)
    u = (D[:n].astype(np.float64) - ex) / np.exp2(E - 24.0)
    return u[fin], ex, P, C


def main():
    print("## Determinism (3 runs, bitwise)")
    for c in ("e4_random", "p1b", "adv", "specials", "f16_full", "kperm", "pos_shift", "pos_perm"):
        print(f"- {c}: f32-acc identical = {determinism(c, 'f32', np.float32)}, f16-acc identical = {determinism(c, 'f16', np.float16)}")
    for c in ("int_s8.wrap", "int_s8.sat", "int_u8.wrap"):
        base, suf = c.split(".")
        print(f"- {c}: identical = {determinism(base, suf, np.int32)}")

    print("\n## Position dependence")
    D0 = dout("e4_random", "f32", np.float32)[:16384]
    Ds = dout("pos_shift", "f32", np.float32)
    print(f"- shifted by one problem slot: outputs differing = {int((~same_bits(np.roll(Ds, -1, 0), D0)).sum())} of {D0.size}")
    zp = load("pos_perm")
    mp, npm = zp["mperm"], zp["nperm"]
    Dp = dout("pos_perm", "f32", np.float32)
    print(f"- rows of A / columns of B permuted (other lanes and registers): outputs differing = "
          f"{int((~same_bits(Dp, D0[:, mp][:, :, npm])).sum())} of {D0.size}")

    print("\n## K-order dependence (kperm: 2048 problems x 8 K permutations of A columns / B rows)")
    zk = load("kperm")
    Dk = dout("kperm", "f32", np.float32).reshape(8, -1, 16, 16)
    Dk16 = dout("kperm", "f16", np.float16).reshape(8, -1, 16, 16)
    for j in range(1, 8):
        print(f"- perm {j} {zk['perms'][j].tolist()}: f32-acc outputs differing from identity order = "
              f"{int((~same_bits(Dk[j], Dk[0])).sum())}, f16-acc = {int((~same_bits(Dk16[j], Dk16[0])).sum())} of {Dk[0].size}")

    print("\n## Error size, f32 accumulate (|D - exact| in units of 2^(E-24), E = exponent of the largest term)")
    for c, n in (("e4_random", 16384), ("p1b", 8192), ("adv", 8192), ("f16_full", 4096)):
        u, ex, P, C = err_units(c, dout(c, "f32", np.float32), n)
        neg = ((P < 0).any(-1) | (C < 0))
        pos = ((P > 0).any(-1) | (C > 0))
        mixed = (neg & pos)[np.isfinite(ex)]
        print(f"- {c}: max |err| = {np.abs(u).max():.3f} units; outputs with err != 0: all terms same sign "
              f"{int((u[~mixed] != 0).sum())} of {int((~mixed).sum())}, mixed signs {int((u[mixed] != 0).sum())} of {int(mixed.sum())}")

    two = {k: MODELS[k] for k in ("M-exact", "M-g8")}
    print_fit("e4_random (all 131072 problems): v_wmma_f32_16x16x16_f16, two headline models",
              fit("e4_random", dout("e4_random", "f32", np.float32), np.float32, None, F64, models=two))
    for c, n, be in (("e4_random", 16384, F64), ("p1b", None, F64), ("adv", None, F64), ("kperm", 4096, F64),
                     ("f16_full", F16_FULL_N, INT)):
        print_fit(f"{c}: v_wmma_f32_16x16x16_f16 (f32 accumulate)" + (f", first {n} problems" if n else ""),
                  fit(c, dout(c, "f32", np.float32), np.float32, n, be))
    sub = {k: MODELS[k] for k in ("M-exact", "M-seq", "M-g8", "M-g4", "M-g2", "M-tree", "M-sum16-then-C")}
    for c, n, be in (("e4_random", 16384, F64), ("p1b", None, F64), ("adv", None, F64), ("f16_full", F16_FULL_N, INT)):
        print_fit(f"{c}: v_wmma_f16_16x16x16_f16 (f16 accumulate, C = RN16(C))" + (f", first {n} problems" if n else ""),
                  fit(c, dout(c, "f16", np.float16), np.float16, n, be, ctype=np.float16, models=sub))

    print("\n## Signed zero (adv sub-kind 7: C = +-0, products +-0 or nonzero)")
    z = load("adv")
    D = dout("adv", "f32", np.float32)
    k7 = z["kind"] == 7
    A, B, C = z["A"][k7].astype(np.float64), z["B"][k7].astype(np.float64), z["C"][k7].astype(np.float64)
    A16, B16 = z["A"][k7], z["B"][k7]
    P = products(A, B)
    nb = np.signbit(A16)[:, :, None, :] ^ np.signbit(B16).transpose(0, 2, 1)[:, None, :, :]
    allzero = (P == 0).all(-1)
    allneg = allzero & nb.all(-1) & np.signbit(C)
    Dk7 = D[k7]
    for label, sel in (("every term -0 (C = -0, all products -0)", allneg),
                       ("all terms zero, some +0", allzero & ~allneg),):
        d = Dk7[sel]
        print(f"- {label}: {int(sel.sum())} outputs -> +0: {int(((d == 0) & ~np.signbit(d)).sum())}, -0: "
              f"{int(((d == 0) & np.signbit(d)).sum())}, nonzero: {int((d != 0).sum())}")
    zs = ~allzero & (P.sum(-1) + C == 0)
    d = Dk7[zs]
    print(f"- nonzero products cancelling to an exact 0: {int(zs.sum())} outputs -> +0: {int(((d == 0) & ~np.signbit(d)).sum())}, "
          f"-0: {int(((d == 0) & np.signbit(d)).sum())}, nonzero: {int((d != 0).sum())}"
          + (f" (values seen: {sorted(set(np.unique(d[d != 0]).tolist()))[:6]})" if (d != 0).any() else ""))
    for name in ("f16",):
        D16 = dout("adv", "f16", np.float16)[k7]
        d = D16[allneg]
        print(f"- f16 accumulate, every term -0: +0 {int(((d == 0) & ~np.signbit(d)).sum())}, -0 {int(((d == 0) & np.signbit(d)).sum())}")

    print("\n## Specials (C = +-inf, NaN, NaN operands)")
    z = load("specials")
    D = dout("specials", "f32", np.float32)
    kind = z["kind"]
    A, B, C = z["A"].astype(np.float64), z["B"].astype(np.float64), z["C"].astype(np.float64)
    with np.errstate(invalid="ignore"):
        ref = np.einsum("pmk,pkn->pmn", A, B) + C      # IEEE-style expectation (NaN where any NaN, inf from C)
    for k, label in ((0, "C = +-inf (30% of outputs)"), (1, "C = NaN (30%)"), (2, "one NaN code in A"), (3, "C = +-inf everywhere")):
        s = kind == k
        r, d = ref[s], D[s]
        print(f"- {label}: outputs where expected NaN: {int(np.isnan(r).sum())}, GPU NaN there: {int(np.isnan(d[np.isnan(r)]).sum())}; "
              f"expected +-inf: {int(np.isinf(r).sum())}, GPU same inf: {int((d[np.isinf(r)] == r[np.isinf(r)]).sum())}; "
              f"GPU NaN elsewhere: {int(np.isnan(d[~np.isnan(r)]).sum())}")

    print("\n## FP16 subnormal operands (f16_full sub-kinds 2 and 3)")
    z = load("f16_full")
    D = dout("f16_full", "f32", np.float32)
    for k, label in ((2, "A subnormal, B normal"), (3, "A, B in the lowest binades (subnormals), C = 0")):
        s = z["kind"] == k
        A, B, C = z["A"][s].astype(np.float64), z["B"][s].astype(np.float64), z["C"][s].astype(np.float64)
        ex = np.einsum("pmk,pkn->pmn", A, B) + C
        Ac = np.where(np.abs(z["A"][s]) < 2.0 ** -14, 0, A)
        Bc = np.where(np.abs(z["B"][s]) < 2.0 ** -14, 0, B)
        flushed = np.einsum("pmk,pkn->pmn", Ac, Bc) + C
        d = D[s].astype(np.float64)
        rel = np.abs(d - ex) / np.maximum(np.abs(ex), 1e-300)
        print(f"- {label}: outputs {d.size}; |D - exact| / |exact| median {np.median(rel):.2e}, max {rel.max():.2e}; "
              f"D equal to the flush-to-zero result where that differs from exact: "
              f"{int(((d == flushed.astype(np.float32)) & (flushed != ex)).sum())} of {int((flushed != ex).sum())}")

    print("\n## Integer (iu8) instruction")
    for c, suf, signed in (("int_s8", "wrap", True), ("int_s8", "sat", True), ("int_u8", "wrap", False)):
        z = load(c)
        A, B, C = z["A"].astype(np.int64), z["B"].astype(np.int64), z["C"].astype(np.int64)
        ex = np.einsum("pmk,pkn->pmn", A, B) + C
        D = dout(c, suf, np.int32 if signed else np.uint32).astype(np.int64)
        if suf == "wrap":
            m = (ex + (2 ** 31 if signed else 0)) % 2 ** 32 - (2 ** 31 if signed else 0)
        else:
            m = np.clip(ex, -2 ** 31, 2 ** 31 - 1)
        print(f"- {c} {suf}: mismatches vs exact ({'wrapping' if suf == 'wrap' else 'saturating'}) = {int((D != m).sum())} of {D.size}; "
              f"outputs that overflow int32: {int(((ex < (-2 ** 31 if signed else 0)) | (ex >= (2 ** 31 if signed else 2 ** 32))).sum())}")


if __name__ == "__main__":
    main()
