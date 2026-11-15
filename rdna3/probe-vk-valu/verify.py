"""Inputs and CPU references for the three verification kernels. The references are written from the definitions
(nearest representable value, ties to even, saturation rules), not from the shader's bit tricks.
    python3 verify.py gen    -> raw/verify_q8.in, raw/verify_rq.in
    python3 verify.py check  -> compares raw/verify_*.out with the references, prints mismatch counts
"""
import sys
import numpy as np

N = 1 << 22  # 4,194,304 inputs per kernel
R = "raw/"


def e4m3_values():
    c = np.arange(0x7f)  # positive finite codes 0..0x7e
    e, m = c >> 3, c & 7
    return np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7.0))


def ref_e4m3(x32, saturate):
    """FP8 E4M3FN code, round to nearest even. hw: |x| > 464 or inf -> 0x7f|sign. sat: |x| >= 448 -> 0x7e|sign.
    NaN -> 0xff."""
    x = x32.astype(np.float64)
    s = np.where(np.signbit(x32), 0x80, 0).astype(np.uint32)
    a = np.abs(x)
    v = e4m3_values()
    lo = np.clip(np.searchsorted(v, a, side="right") - 1, 0, 0x7e)
    hi = np.minimum(lo + 1, 0x7e)
    dlo, dhi = a - v[lo], v[hi] - a
    code = np.where(dhi < dlo, hi, np.where(dlo < dhi, lo, np.where(lo % 2 == 0, lo, hi))).astype(np.uint32)
    if saturate:
        code = np.where(a >= 448.0, 0x7e, code)
    else:
        code = np.where(a > 464.0, 0x7f, np.where(a > 448.0, 0x7e, code))
    return np.where(np.isnan(x), 0xff, s | code).astype(np.uint32)


def gen():
    r = np.random.default_rng(99)
    q = N // 4
    bits = r.integers(0, 2 ** 32, q, dtype=np.uint32)                                  # every bit pattern
    rng = (r.integers(0, 2 ** 23, q, dtype=np.uint32) | (r.integers(100, 141, q, dtype=np.uint32) << 23)
           | (r.integers(0, 2, q, dtype=np.uint32) << 31))                               # 2^-27 .. 2^13
    v = e4m3_values()
    mids = (v[:-1] + v[1:]) / 2.0                                                        # exact ties and neighbours
    t = np.concatenate([mids, -mids, [448.0, 464.0, 480.0, -464.0, 2.0 ** -10]]).astype(np.float32)
    tb = np.tile(t.view(np.uint32), q // len(t) + 1)[:q]
    tb = tb + r.integers(-1, 2, q).astype(np.uint32)                                     # tie, one ulp below, above
    sp = np.array([0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00001, 0x7f800001, 1, 0x80000001,
                   0x43e00000, 0x43e80000, 0x43e80001, 0x3c800000, 0x3c7fffff], np.uint32)
    special = np.concatenate([sp, r.integers(0, 2 ** 32, q - len(sp), dtype=np.uint32)])
    np.concatenate([bits, rng, tb, special]).astype(np.uint32).tofile(R + "verify_q8.in")
    ints = np.concatenate([r.integers(-30000, 30001, N // 2), r.integers(-2 ** 31, 2 ** 31, N // 2)]).astype(np.int32)
    ints[:6] = [0, 1, -1, 2 ** 31 - 1, -2 ** 31, 11916]
    ints.tofile(R + "verify_rq.in")


def check():
    ok = True
    x = np.fromfile(R + "verify_q8.in", np.uint32)
    o = np.fromfile(R + "verify_q8.out", np.uint32)
    xf = x.view(np.float32)
    for name, got, sat in (("e4m3_hw", o & 0xff, False), ("e4m3_sat", (o >> 8) & 0xff, True)):
        ref = ref_e4m3(xf, sat)
        bad = np.nonzero(got != ref)[0]
        ok &= len(bad) == 0
        print(f"{name}: {len(bad)} mismatches of {len(x)}" + (f"; first x=0x{x[bad[0]]:08x} gpu=0x{got[bad[0]]:02x} cpu=0x{ref[bad[0]]:02x}" if len(bad) else ""))
    o = np.fromfile(R + "verify_e4h.out", np.uint32)[:256]
    c = np.arange(256)
    val = np.where((c & 0x7f) >> 3 == 0, (c & 7) / 8.0 * 2.0 ** -6, (1 + (c & 7) / 8.0) * 2.0 ** (((c & 0x7f) >> 3) - 7.0))
    val = np.where(c & 0x80, -val, val)
    for name, got, scale in (("e4m3->f16 bit form (x 2^-8)", o & 0xffff, 2.0 ** -8), ("e4m3->f16 exact", o >> 16, 1.0)):
        ref = (val * scale).astype(np.float16).view(np.uint16).astype(np.uint32)
        fin = (c & 0x7f) != 0x7f
        bad = np.nonzero((got != ref) & fin)[0]
        ok &= len(bad) == 0
        print(f"{name}: {len(bad)} mismatches of {fin.sum()} finite codes (codes 0x7f/0xff, NaN in E4M3FN, give "
              f"0x{got[0x7f]:04x}/0x{got[0xff]:04x}: finite, not NaN)")
    a = np.fromfile(R + "verify_rq.in", np.int32)
    o = np.fromfile(R + "verify_rq.out", np.uint32)
    exact = np.clip(np.rint(a.astype(np.float32).astype(np.float64) * 0.0107421875), -128, 127).astype(np.int64) & 0xff
    for name, got in (("requant f32 (cvt, mul, rndne, cvt, clamp)", o & 0xff), ("requant magic (fma + med3)", (o >> 8) & 0xff)):
        bad = np.nonzero(got != exact)[0]
        ok &= len(bad) == 0
        print(f"{name}: {len(bad)} mismatches of {len(a)}" + (f"; first a={a[bad[0]]} gpu={got[bad[0]]} cpu={exact[bad[0]]}" if len(bad) else ""))
    print("VERIFY", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen()
    else:
        sys.exit(0 if check() else 1)
