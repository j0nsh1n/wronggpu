"""Exact arithmetic for the model fits.

Two backends with one interface. Values are exact numbers; `rnd` rounds an exact number once, to nearest-even FP32 or
FP16 (overflow to inf, IEEE subnormals kept).

* F64: for E4M3 operands. Every product of two E4M3 values is a multiple of 2^-18 below 2^17.6, so any sum of up to 16
  products is a multiple of 2^-18 below 2^22 (40 bits) and float64 holds it exactly in any order. An exact number is
  then a pair (hi, lo) of float64 with hi + lo exact (lo from an error-free TwoSum), which is enough for one addition
  of an FP32 accumulator.
* INT: for arbitrary FP16 operands. An exact number is a Python int counting units of 2^-SCALE (object arrays).
"""
import math

import numpy as np

SCALE = 200


def _round_pair(hi, lo, dt):
    """Nearest-even rounding of the exact value hi + lo (|lo| <= ulp64(hi)/2) to dt. Only an exact dt-midpoint hi can
    be decided wrongly by rounding hi alone; lo then breaks the tie."""
    with np.errstate(over="ignore", invalid="ignore"):
        r = hi.astype(dt)
        r64 = r.astype(np.float64)
        up = np.nextafter(r, np.array(np.inf, dt)).astype(np.float64)
        dn = np.nextafter(r, np.array(-np.inf, dt)).astype(np.float64)
        other = np.where(hi > r64, up, dn)
        mid = (hi != r64) & np.isfinite(r64) & np.isfinite(other) & (hi == (r64 + other) / 2)
        fix = mid & (lo != 0) & ((lo > 0) == (other > r64))
        out = np.where(fix, other, r64)
    return out


def two_sum(a, b):
    with np.errstate(invalid="ignore"):
        s = a + b
        bb = s - a
        e = (a - (s - bb)) + (b - bb)
    e = np.where(np.isfinite(s), e, 0.0)
    return s, e


class F64:
    """Exact numbers as (hi, lo) float64 pairs; inputs must be sums of E4M3 products plus at most one FP32/FP16 value."""
    name = "f64"

    @staticmethod
    def const(x):
        x = np.asarray(x, np.float64)
        return (x, np.zeros_like(x))

    @staticmethod
    def psum(P, ks):
        s = P[..., ks].sum(-1) if len(ks) else np.zeros(P.shape[:-1])
        return (s, np.zeros_like(s))

    @staticmethod
    def add(x, y):
        assert not (np.any(x[1]) and np.any(y[1])), "F64 backend: at most one inexact pair"
        hi, lo = two_sum(x[0], y[0])
        return (hi, lo + x[1] + y[1])

    @staticmethod
    def rnd(x, dt=np.float32):
        hi, lo = two_sum(x[0], x[1])
        return _round_pair(hi, lo, dt)


def _to_int(v):
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        return v
    m, e = math.frexp(v)
    return int(m * 2.0 ** 53) * (1 << (e - 53 + SCALE)) if e - 53 + SCALE >= 0 else int(m * 2.0 ** 53) >> -(e - 53 + SCALE)


_to_int_v = np.frompyfunc(_to_int, 1, 1)


def _rnd_int(X, dt):
    if isinstance(X, float):
        return X
    if X == 0:
        return 0.0
    hi = X / (1 << SCALE)                     # correctly rounded to float64 (Python int true division)
    back = int(hi * 2.0 ** SCALE) if abs(hi) < 2.0 ** 800 else None
    lo = 0 if back is None else X - back
    r = _round_pair(np.array([hi]), np.array([float(np.sign(lo))]), dt)[0]
    return float(r)


_rnd32_v = np.frompyfunc(lambda X: _rnd_int(X, np.float32), 1, 1)
_rnd16_v = np.frompyfunc(lambda X: _rnd_int(X, np.float16), 1, 1)


class INT:
    name = "int"

    @staticmethod
    def const(x):
        return _to_int_v(np.asarray(x, np.float64))

    @staticmethod
    def psum(P, ks):
        if not len(ks):
            return np.zeros(P.shape[:-1], object) * 0
        s = P[..., ks[0]]
        for k in ks[1:]:
            s = s + P[..., k]
        return s

    @staticmethod
    def add(x, y):
        return x + y

    @staticmethod
    def rnd(x, dt=np.float32):
        f = _rnd32_v if dt == np.float32 else _rnd16_v
        return f(x).astype(np.float64)
