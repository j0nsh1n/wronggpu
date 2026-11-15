"""Input generator for the coopmat probe. Fixed seeds; writes inputs/<class>.npz and raw .bin files for the harness.

Every problem p is one 16x16x16 multiply-accumulate D = A @ B + C:
  A[p] : 16x16 row-major (row m, column k), B[p] : 16x16 row-major (row k, column n), C[p] : 16x16 row-major.
E4M3 classes also store the codes (A8, B8); their float16 values are the exact widening of the codes."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "inputs")


def e4m3_table():
    t = np.empty(256, np.float64)
    for c in range(256):
        s = -1.0 if c & 0x80 else 1.0
        e, m = (c >> 3) & 0xF, c & 7
        if e == 0xF and m == 7:
            t[c] = np.nan
        elif e == 0:
            t[c] = s * m * 2.0 ** -9
        else:
            t[c] = s * (1 + m / 8) * 2.0 ** (e - 7)
    return t


E4 = e4m3_table()
FINITE = np.array([c for c in range(256) if not np.isnan(E4[c])], np.uint8)
assert len(FINITE) == 254 and np.all(E4[FINITE].astype(np.float16).astype(np.float64) == E4[FINITE])


def code_of(v):
    """E4M3 code of an exactly representable value (array)."""
    v = np.asarray(v, np.float64)
    pos = E4[:0x7F]
    idx = np.searchsorted(pos, np.abs(v))
    assert np.all(pos[np.minimum(idx, 0x7E)] == np.abs(v)), "value not E4M3"
    return (idx + np.where(np.signbit(v), 0x80, 0)).astype(np.uint8)


def rand_c(rng, shape):
    """P1b-style FP32 C: zero, full-mantissa f32 with exponent -24..21, f16-like, or x1024."""
    kind = rng.integers(0, 4, shape)
    ex = rng.integers(-24, 22, shape)
    mant = rng.random(shape) + 1.0
    sg = np.where(rng.random(shape) < 0.5, -1.0, 1.0)
    full = (sg * mant * np.exp2(ex)).astype(np.float32)
    with np.errstate(over="ignore"):
        f16v = np.where(np.abs(full) < 65504, full.astype(np.float16).astype(np.float32), full)  # f16-like only where finite
    return np.where(kind == 0, np.float32(0), np.where(kind == 1, full, np.where(kind == 2, f16v, full * np.float32(1024)))).astype(np.float32)


def rand_f32_bits(rng, shape, emin, emax):
    e = rng.integers(emin, emax + 1, shape).astype(np.uint32)
    b = (e << 23) | rng.integers(0, 1 << 23, shape).astype(np.uint32) | (rng.integers(0, 2, shape).astype(np.uint32) << 31)
    return b.view(np.float32)


# ---- E4M3 operand classes -------------------------------------------------------------------------------------------

def e4_random(rng, n):
    A8 = rng.choice(FINITE, (n, 16, 16))
    B8 = rng.choice(FINITE, (n, 16, 16))
    return A8, B8, rand_c(rng, (n, 16, 16)), np.zeros(n, np.int8)


def p1b(rng, n):
    """P1b's 8 data modes (clean/rr_f/gpu/probe_p1b.py), on 16x16 operands; mode = problem index % 8."""
    modes = (np.arange(n) % 8).astype(np.int8)
    A8 = np.zeros((n, 16, 16), np.uint8)
    B8 = np.zeros((n, 16, 16), np.uint8)
    for i in range(n):
        m, s = modes[i], (16, 16)
        if m in (0, 1, 2, 5):
            dens = {0: 0.25, 1: 0.35, 2: 0.5, 5: 0.3}[m]
            A8[i] = np.where(rng.random(s) < dens, rng.choice(FINITE, s), 0)
            B8[i] = np.where(rng.random(s) < dens, rng.choice(FINITE, s), 0)
        elif m == 3:
            A8[i], B8[i] = rng.choice(FINITE, s), rng.choice(FINITE, s)
        elif m == 4:
            e, mm, sg = rng.integers(12, 15, (2,) + s), rng.integers(0, 8, (2,) + s), rng.integers(0, 2, (2,) + s)
            A8[i], B8[i] = sg[0] * 128 + e[0] * 8 + mm[0], sg[1] * 128 + e[1] * 8 + mm[1]
        else:
            dens = 0.3 if m == 6 else 1.0
            e, mm, sg = rng.integers(1, 10, (2,) + s), rng.integers(0, 8, (2,) + s), rng.integers(0, 2, (2,) + s)
            A8[i] = np.where(rng.random(s) < dens, sg[0] * 128 + e[0] * 8 + mm[0], 0)
            B8[i] = np.where(rng.random(s) < dens, sg[1] * 128 + e[1] * 8 + mm[1], 0)
    C = rand_c(rng, (n, 16, 16))
    C[modes == 7] *= np.float32(2.0 ** 10)   # "wide C": push C far above the products in one mode
    return A8, B8, C, modes


def adv(rng, n):
    """Adversarial E4M3 cases, sub-kind = problem index % 8:
    0 cancel_acc : C = -A[m,0]*B[0,n] exactly, the other 15 products tiny -> result = tiny exact sum
    1 cancel_pair: products cancel in pairs (k, k+8) and (k, k+1), C small
    2 ties_one   : C has ulp u; one product = u/2 (a tie), a sticky +-2^-18 product or none, positions random
    3 ties_split : the u/2 tie split in 2-4 parts spread over K 0..7 and 8..15, sticky term in either half
    4 big_c      : |C| in 2^20..2^40 full mantissa, all 16 products tiny with random signs
    5 spread     : products spanning 2^-18 .. 2^17 in every output, random C
    6 near_ovf   : C near FP32 max, products push past it or cancel
    7 sgn_zero   : C = +-0, products +-0 (mixed signs), exact-zero pairs"""
    kinds = (np.arange(n) % 8).astype(np.int8)
    A = np.zeros((n, 16, 16), np.float64)
    B = np.zeros((n, 16, 16), np.float64)
    C = np.zeros((n, 16, 16), np.float64)
    tiny_codes = np.array([c for c in FINITE if 0 < abs(E4[c]) <= 2.0 ** -4], np.uint8)
    for i in range(n):
        k = kinds[i]
        if k == 0:
            A[i] = E4[rng.choice(tiny_codes, (16, 16))]
            B[i] = E4[rng.choice(tiny_codes, (16, 16))]
            big = np.array([c for c in FINITE if abs(E4[c]) >= 8], np.uint8)
            A[i, :, 0] = E4[rng.choice(big, 16)]
            B[i, 0, :] = E4[rng.choice(big, 16)]
            C[i] = -np.outer(A[i, :, 0], B[i, 0, :])
        elif k == 1:
            x = E4[rng.choice(FINITE, (16, 8))]
            y = E4[rng.choice(FINITE, (8, 16))]
            A[i, :, :8], A[i, :, 8:] = x, x
            B[i, :8, :], B[i, 8:, :] = y, -y
            j = rng.integers(0, 8)                       # one leftover product pair breaks the cancellation
            B[i, 8 + j, :] = E4[rng.choice(tiny_codes, 16)]
            C[i] = E4[rng.choice(tiny_codes, (16, 16))] * E4[rng.choice(tiny_codes, (16, 16))]
        elif k in (2, 3):
            # C row m in binade E_m (E_m in Ebase..Ebase+12): ulp 2^(E-23), half ulp 2^(E-24) = A[m,k] * B[k,n] with
            # B[k,n] = +-b0, b0 = 2^(Ebase-17), so A[m,k] = 2^(E-Ebase-7) in 2^-7 .. 2^5 (quarters down to 2^-9)
            Ebase = rng.integers(9, 18)
            E = Ebase + rng.integers(0, 13, 16)
            frac = rng.integers(0, 1 << 23, (16, 16))
            sg = rng.choice([-1.0, 1.0], (16, 16))
            C[i] = sg * (2.0 ** 23 + frac) * np.exp2(E[:, None] - 23.0)
            b0 = 2.0 ** (Ebase - 17)
            a_half = np.exp2(E - Ebase - 7.0)
            col_sg = rng.choice([-1.0, 1.0], 16) * b0   # the tie pushes toward or away from zero per column
            perm = rng.permutation(16)
            if k == 2:
                A[i, :, perm[0]] = a_half
                B[i, perm[0], :] = col_sg
                ks = perm[1]
            else:
                parts = int(rng.integers(2, 5))
                share = {2: [2, 2], 3: [2, 4, 4], 4: [4, 4, 4, 4]}[parts]
                for q in range(parts):
                    A[i, :, perm[q]] = a_half / share[q]
                    B[i, perm[q], :] = col_sg
                ks = perm[parts]
            sticky = rng.choice([2.0 ** -18, -2.0 ** -18, 0.0], 16)
            A[i, :, ks] = 2.0 ** -9
            B[i, ks, :] = sticky / 2.0 ** -9
        elif k == 4:
            A[i] = E4[rng.choice(tiny_codes, (16, 16))] * rng.choice([1.0, 16.0], (16, 16))
            B[i] = E4[rng.choice(tiny_codes, (16, 16))]
            C[i] = rand_f32_bits(rng, (16, 16), 127 + 20, 127 + 40).astype(np.float64)
        elif k == 5:
            ea = rng.integers(-6, 9, (16, 16))
            eb = rng.integers(-6, 9, (16, 16))
            A[i] = np.exp2(ea) * rng.choice([1.0, 1.125, 1.5, 1.875], (16, 16)) * rng.choice([-1.0, 1.0], (16, 16))
            B[i] = np.exp2(eb) * rng.choice([1.0, 1.25, 1.75], (16, 16)) * rng.choice([-1.0, 1.0], (16, 16))
            np.clip(A[i], -448, 448, out=A[i])
            np.clip(B[i], -448, 448, out=B[i])
            C[i] = rand_c(rng, (16, 16))
        elif k == 6:
            A[i] = E4[rng.choice(FINITE, (16, 16))]
            B[i] = E4[rng.choice(FINITE, (16, 16))]
            C[i] = rng.choice([-1.0, 1.0], (16, 16)) * np.float64(np.finfo(np.float32).max) * rng.choice([1.0, 0.99999994, 0.9999999], (16, 16))
            C[i] = C[i].astype(np.float32)
        else:
            z = rng.choice([0.0, -0.0], (16, 16))
            A[i] = np.where(rng.random((16, 16)) < 0.5, z, E4[rng.choice(FINITE, (16, 16))])
            B[i] = np.where(rng.random((16, 16)) < 0.5, rng.choice([0.0, -0.0], (16, 16)), E4[rng.choice(FINITE, (16, 16))])
            # make some columns all-zero products: B column n zero (signed) in every k
            zc = rng.random(16) < 0.5
            B[i][:, zc] = rng.choice([0.0, -0.0], (16, int(zc.sum())))
            C[i] = rng.choice([0.0, -0.0], (16, 16))
    for X in (A, B):
        X[np.abs(X) > 448] = np.sign(X[np.abs(X) > 448]) * 448
    return code_of(A), code_of(B), C.astype(np.float32), kinds


def specials(rng, n):
    """C = +-inf / NaN, NaN E4M3 operands (code 0x7F / 0xFF), on otherwise random E4M3 data. Sub-kind = index % 4:
    0 C = +-inf, 1 C = NaN, 2 one NaN code in A per problem, 3 C = +inf and -inf mixed with large finite products."""
    A8, B8, C, _ = e4_random(rng, n)
    kinds = (np.arange(n) % 4).astype(np.int8)
    for i in range(n):
        k = kinds[i]
        if k == 0:
            msk = rng.random((16, 16)) < 0.3
            C[i][msk] = rng.choice([np.inf, -np.inf], int(msk.sum()))
        elif k == 1:
            msk = rng.random((16, 16)) < 0.3
            C[i][msk] = np.nan
        elif k == 2:
            A8[i, rng.integers(0, 16), rng.integers(0, 16)] = rng.choice([0x7F, 0xFF])
        else:
            C[i] = rng.choice([np.inf, -np.inf], (16, 16))
    return A8, B8, C, kinds


# ---- full-range F16 operand classes ---------------------------------------------------------------------------------

def f16_bits(rng, shape, emax=30, emin=0):
    e = rng.integers(emin, emax + 1, shape).astype(np.uint16)
    b = (e << 10) | rng.integers(0, 1024, shape).astype(np.uint16) | (rng.integers(0, 2, shape).astype(np.uint16) << 15)
    return b.view(np.float16)


def f16_full(rng, n):
    """Sub-kind = index % 4: 0 any finite f16 with exponent field <= 22 (|x| < 2^8, subnormals included) and random C,
    1 uniform(-2, 2) operands, 2 A subnormal-only (exponent field 0) with B ordinary and C = 0 or tiny,
    3 both A and B in the lowest binades (subnormal x normal, products 2^-48 ..) with C = 0."""
    kinds = (np.arange(n) % 4).astype(np.int8)
    A = np.zeros((n, 16, 16), np.float16)
    B = np.zeros((n, 16, 16), np.float16)
    C = np.zeros((n, 16, 16), np.float32)
    for i in range(n):
        k = kinds[i]
        if k == 0:
            A[i], B[i], C[i] = f16_bits(rng, (16, 16), 22), f16_bits(rng, (16, 16), 22), rand_c(rng, (16, 16))
        elif k == 1:
            A[i] = rng.uniform(-2, 2, (16, 16)).astype(np.float16)
            B[i] = rng.uniform(-2, 2, (16, 16)).astype(np.float16)
            C[i] = rng.uniform(-4, 4, (16, 16)).astype(np.float32)
        elif k == 2:
            A[i] = f16_bits(rng, (16, 16), 0)
            B[i] = f16_bits(rng, (16, 16), 20, 10)
            C[i] = np.where(rng.random((16, 16)) < 0.5, 0, rand_f32_bits(rng, (16, 16), 127 - 30, 127 - 14))
        else:
            A[i] = f16_bits(rng, (16, 16), 2)
            B[i] = f16_bits(rng, (16, 16), 12)
    return A, B, C, kinds


def kperm(base_A8, base_B8, base_C, rng, nperm=8):
    """The same problems with the 16 K slots permuted (A columns and B rows together). perm 0 = identity,
    1 = reverse, 2 = swap halves, 3 = even/odd interleave, 4.. = random."""
    perms = [np.arange(16), np.arange(16)[::-1], np.r_[8:16, 0:8], np.r_[0:16:2, 1:16:2]]
    while len(perms) < nperm:
        perms.append(rng.permutation(16))
    perms = np.array(perms)
    nb = len(base_A8)
    A8 = np.concatenate([base_A8[:, :, p] for p in perms])
    B8 = np.concatenate([base_B8[:, p, :] for p in perms])
    C = np.concatenate([base_C] * nperm)
    return A8, B8, C, np.repeat(np.arange(nperm), nb).astype(np.int8), perms


def save(name, A, B, C, kinds, e4=True, **extra):
    n = len(A)
    assert n % 4 == 0
    os.makedirs(OUT, exist_ok=True)
    if e4:
        A8, B8 = A, B
        A16 = E4[A8].astype(np.float16)
        B16 = E4[B8].astype(np.float16)
        extra.update(A8=A8, B8=B8)
    else:
        A16, B16 = A, B
    np.savez_compressed(os.path.join(OUT, name + ".npz"), A=A16, B=B16, C=C.astype(np.float32), kind=kinds, **extra)
    A16.tofile(os.path.join(OUT, name + ".A16.bin"))
    B16.tofile(os.path.join(OUT, name + ".B16.bin"))
    C.astype(np.float32).tofile(os.path.join(OUT, name + ".C32.bin"))
    with np.errstate(over="ignore"):
        C.astype(np.float32).astype(np.float16).tofile(os.path.join(OUT, name + ".C16.bin"))
    print(f"{name}: {n} problems")


def int_class(rng, n, signed):
    lo, hi = (-128, 128) if signed else (0, 256)
    A = rng.integers(lo, hi, (n, 16, 16)).astype(np.int8 if signed else np.uint8)
    B = rng.integers(lo, hi, (n, 16, 16)).astype(np.int8 if signed else np.uint8)
    # C: random int32, a quarter of problems near the int32 limits so A B + C can overflow
    C = rng.integers(-(2 ** 31) if signed else 0, 2 ** 31 if signed else 2 ** 32, (n, 16, 16), dtype=np.int64)
    edge = (np.arange(n) % 4 == 3)
    span = 2 ** 20
    if signed:
        C[edge] = np.where(rng.random((int(edge.sum()), 16, 16)) < 0.5, 2 ** 31 - 1 - rng.integers(0, span, (int(edge.sum()), 16, 16)),
                           -(2 ** 31) + rng.integers(0, span, (int(edge.sum()), 16, 16)))
        C = C.astype(np.int32)
    else:
        C[edge] = 2 ** 32 - 1 - rng.integers(0, span, (int(edge.sum()), 16, 16))
        C = C.astype(np.uint32)
    name = "int_s8" if signed else "int_u8"
    np.savez_compressed(os.path.join(OUT, name + ".npz"), A=A, B=B, C=C, kind=edge.astype(np.int8))
    A.tofile(os.path.join(OUT, name + ".A.bin"))
    B.tofile(os.path.join(OUT, name + ".B.bin"))
    C.tofile(os.path.join(OUT, name + ".C.bin"))
    print(f"{name}: {n} problems")


def main():
    n_random = int(sys.argv[1]) if len(sys.argv) > 1 else 131072
    save("e4_random", *e4_random(np.random.default_rng(101), n_random))
    save("p1b", *p1b(np.random.default_rng(102), 8192))
    save("adv", *adv(np.random.default_rng(103), 8192))
    save("specials", *specials(np.random.default_rng(104), 2048))
    save("f16_full", *f16_full(np.random.default_rng(105), 4096), e4=False)
    r = np.random.default_rng(106)
    bA, bB, bC, _ = e4_random(r, 1024)
    pA, pB, pC, _ = p1b(r, 1024)
    A8, B8, C, pk, perms = kperm(np.concatenate([bA, pA]), np.concatenate([bB, pB]), np.concatenate([bC, pC]), r)
    save("kperm", A8, B8, C, pk, perms=perms)
    # position test: the first 16384 random problems, (a) shifted by one problem slot, (b) rows of A and columns of B
    # (output rows / columns) permuted, so every value lands on other lanes / registers
    z = np.load(os.path.join(OUT, "e4_random.npz"))
    A8, B8, C = z["A8"][:16384], z["B8"][:16384], z["C"][:16384]
    rp = np.random.default_rng(107)
    mperm, nperm_ = rp.permutation(16), rp.permutation(16)
    save("pos_shift", np.roll(A8, 1, 0), np.roll(B8, 1, 0), np.roll(C, 1, 0), np.zeros(16384, np.int8))
    save("pos_perm", A8[:, mperm, :], B8[:, :, nperm_], C[:, mperm][:, :, nperm_], np.zeros(16384, np.int8),
         mperm=mperm, nperm=nperm_)
    int_class(np.random.default_rng(108), 4096, True)
    int_class(np.random.default_rng(109), 4096, False)


if __name__ == "__main__":
    main()
