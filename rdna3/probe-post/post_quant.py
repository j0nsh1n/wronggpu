"""Quantization-flip analysis of the RDNA 3 probe outputs (CPU only, read-only on probe-vk/results).

Question: after D (RX 7900 XT v_wmma_f32_16x16x16_f16, or the F16-accumulate variant) is rounded to E4M3 (q8) or to
FP16 (RN16), how often does the code differ from the one made from the exact sum (RN32 of the exact value first, then
the quantizer), or from the RDNA 4 fit M-g8 / the F16 fit M-g2?
Run: python3 -I post_quant.py   (from this directory; writes results.json)."""
import json
import os
import sys
import numpy as np

PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probe-vk")
sys.path.insert(0, PROBE)
sys.path.insert(1, os.environ.get("WRONGGPU_REFERENCE", "clean/reference"))  # the WrongGPU checkout's clean/reference (fp8.py)
from exact import F64          # noqa: E402
from models import MODELS      # noqa: E402
from fp8 import q8, _POS_VALUES  # noqa: E402

CHUNK = 1024
NB = 11   # cancellation bins, see BIN_NAMES
BIN_NAMES = ["<=0", "1", "2", "3", "4-5", "6-8", "9-12", "13-16", ">16", "exact 0", "all"]


def binidx(depth, zero):
    b = np.select([depth <= 0, depth == 1, depth == 2, depth == 3, depth <= 5, depth <= 8, depth <= 12, depth <= 16],
                  [0, 1, 2, 3, 4, 5, 6, 7], 8)
    return np.where(zero, 9, b)


def ord16(x):
    i = x.astype(np.float16).view(np.int16).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFF), i)


def code8(v):
    return np.sign(v).astype(np.int64) * np.searchsorted(_POS_VALUES, np.abs(v))


def is_tie(x, mant, minexp):
    """True where x lies exactly on the midpoint of two neighbouring grid values (grid spacing 2^(e-mant))."""
    a = np.abs(x)
    _, e = np.frexp(a)
    sp = np.ldexp(1.0, np.maximum(e - 1, minexp) - mant)
    t = a / sp
    return (t - np.floor(t)) == 0.5


def cmp_stats(g, r, qa, qb, ra, rb):
    """Counts for one pair of results. qa/qb = q8 values, ra/rb = RN16 values (both float64)."""
    d8 = np.abs(code8(qa) - code8(qb))
    d8 = np.where(qa == qb, 0, np.maximum(d8, 1))
    d16 = np.abs(ord16(ra) - ord16(rb))
    d16 = np.where(ra == rb, 0, np.maximum(d16, 1))
    return d8, d16


class Acc:
    def __init__(self):
        self.d = {}

    def add(self, key, vec):
        v = np.asarray(vec, dtype=np.int64)
        self.d[key] = self.d.get(key, 0) + v


def step_counts(d):
    return [int((d >= 1).sum()), int((d == 1).sum()), int((d == 2).sum()), int((d >= 3).sum())]


def run(cls, acc, bins):
    z = np.load(f"{PROBE}/results/{cls}.npz")
    N = len(z["kind"])
    for s in range(0, N, CHUNK):
        e = min(N, s + CHUNK)
        A = z["A"][s:e].astype(np.float64)
        B = z["B"][s:e].astype(np.float64)
        kind = z["kind"][s:e]
        P = A[:, :, None, :] * B.transpose(0, 2, 1)[:, None, :, :]
        for acc_t, Cin, D, dtn in ((np.float32, z["C"][s:e], z["D_f32"][s:e], "f32"),
                                   (np.float16, z["C16"][s:e], z["D_f16"][s:e], "f16")):
            C = Cin.astype(np.float64)
            dt = np.float32 if dtn == "f32" else np.float16
            Dg = D.astype(np.float64)
            ex = MODELS["M-exact"](P, C, F64, dt)
            alt = MODELS["M-g8" if dtn == "f32" else "M-g2"](P, C, F64, dt)
            ok = np.isfinite(C) & np.isfinite(Dg) & np.isfinite(ex) & np.isfinite(alt)
            ok &= np.isfinite(A).all((1, 2))[:, None, None] & np.isfinite(B).all((1, 2))[:, None, None]
            qx, qa_, qg = q8(ex), q8(alt), q8(Dg)
            rx, ra_, rg = ex.astype(np.float16).astype(np.float64), alt.astype(np.float16).astype(np.float64), \
                Dg.astype(np.float16).astype(np.float64)
            pairs = {"gpu~exact": (Dg, ex, qg, qx, rg, rx), "gpu~alt": (Dg, alt, qg, qa_, rg, ra_),
                     "exact~alt": (ex, alt, qx, qa_, rx, ra_)}
            for kv in np.unique(kind):
                m = (kind == kv)[:, None, None] & ok
                if not m.any():
                    continue
                acc.add((cls, int(kv), dtn, "n"), [int(m.sum()), int((~ok & (kind == kv)[:, None, None]).sum())])
                for pn, (a, b, qa, qb, ra, rb) in pairs.items():
                    d8, d16 = cmp_stats(a, b, qa, qb, ra, rb)
                    mism = int((a.astype(dt).view(np.uint32 if dt == np.float32 else np.uint16)[m]
                                != b.astype(dt).view(np.uint32 if dt == np.float32 else np.uint16)[m]).sum())
                    acc.add((cls, int(kv), dtn, pn), [mism] + step_counts(d8[m]) + step_counts(d16[m]))
                if dtn != "f32":
                    continue
                # cancellation bins (gpu vs exact only)
                T = np.concatenate([np.abs(P), np.abs(C)[..., None]], -1)
                mx = T.max(-1)
                E = np.frexp(np.where(mx > 0, mx, 1.0))[1] - 1
                er = np.frexp(np.where(ex != 0, np.abs(ex), 1.0))[1] - 1
                bi = binidx(E - er, ex == 0)
                d8, d16 = cmp_stats(Dg, ex, qg, qx, rg, rx)
                t8, t16 = is_tie(ex, 3, -6), is_tie(ex, 10, -14)
                uerr = np.abs(Dg - ex) / np.exp2(E - 24.0)
                for bname in range(NB):
                    mb = m & ((bi == bname) if bname < 10 else True)
                    if not mb.any():
                        continue
                    nz = (Dg != ex)
                    acc.add((cls, int(kv), "bin", bname),
                            [int(mb.sum()), int((mb & nz).sum()), int((mb & (d8 > 0)).sum()), int((mb & (d16 > 0)).sum()),
                             int((mb & t8).sum()), int((mb & t8 & (d8 > 0)).sum()),
                             int((mb & t16).sum()), int((mb & t16 & (d16 > 0)).sum()),
                             int(np.round(uerr[mb & np.isfinite(uerr)].sum() * 1000))])
        print(cls, e, "/", N, flush=True)


def main():
    acc = Acc()
    bins = None
    for cls in ("e4_random", "p1b", "adv"):
        run(cls, acc, bins)
    out = {"|".join(map(str, k)): v.tolist() for k, v in acc.d.items()}
    with open("results.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
