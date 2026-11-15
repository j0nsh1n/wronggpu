"""Shared pieces of the INT8 study: model reading, weight-matrix orientation, int8 schemes, exponent spans.

Nothing here writes model data. Callers print or save only aggregate numbers.
"""
import json
import struct

import numpy as np

# ---------------------------------------------------------------------------------------------------- model file

E4M3 = np.empty(256, np.float64)
for _c in range(256):
    _s = -1.0 if _c & 0x80 else 1.0
    _e, _m = (_c >> 3) & 0xF, _c & 7
    E4M3[_c] = np.nan if (_e == 15 and _m == 7) else (_s * _m * 2.0 ** -9 if _e == 0 else _s * (1 + _m / 8) * 2.0 ** (_e - 7))

DT = {0: ("<f4", 4), 1: ("u1", 1), 2: ("<i4", 4), 3: ("<f2", 2), 4: ("<u2", 2), 5: ("<u4", 4), 6: ("e4m3", 1)}


def read_wgcm(path):
    """{stage: {name: (dtype_code, float64 array)}}; e4m3 decoded to values. Reads in place, never copies the file."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"WGCMODEL":
        raise ValueError("not a WrongGPU model file")
    nst = int.from_bytes(data[12:16], "little")
    pos = 88
    out = {}

    def u(n):
        nonlocal pos
        v = int.from_bytes(data[pos:pos + n], "little")
        pos += n
        return v

    def s():
        nonlocal pos
        n = u(2)
        pos += n
        return data[pos - n:pos].decode()

    for _ in range(nst):
        st = s()
        pos += 32
        out[st] = {}
        for _ in range(u(4)):
            name, dt, rank = s(), u(1), u(1)
            shape = [u(4) for _ in range(rank)]
            off, size = u(8), u(8)
            fmt = DT[dt][0]
            raw = data[off:off + size]
            if fmt == "e4m3":
                a = E4M3[np.frombuffer(raw, np.uint8)]
            else:
                a = np.frombuffer(raw, fmt).astype(np.float64)
            out[st][name] = (dt, a.reshape(shape))
    return out


def matmul_tensors(model, map_path):
    """Tensors that feed a matrix multiply, as (stage, name, W2 [out, K]).

    Which: the map's op `fp8` / dtype `e4m3` (FP8 models), else float tensors of rank >= 2 with op `f16` (FP16
    models). Orientation: container 1 (M, L, K) stores [..out.., K] (K last, spec 90 names); container 2 (RR) stores
    [..K.., out] with a head axis first for Wq / Wv ([h, C, 32]: out = h*32) and Wup [Kc, 4, C] (K = Kc)."""
    mp = json.load(open(map_path))
    entries = mp.get("stages") or mp.get("groups")
    fp8_model = any((t.get("op") or {}).get("name") in ("fp8", "e4m3") for e in entries for t in e["tensors"])
    res = []
    for e in entries:
        st = e.get("stage") or e.get("group")
        if st not in model:
            continue
        for t in e["tensors"]:
            op = (t.get("op") or {}).get("name")
            if fp8_model:
                if op not in ("fp8", "e4m3"):
                    continue
            elif not (len(t["shape"]) >= 2 and op in ("f16", None) and np.prod(t["shape"]) >= 256):
                continue
            if t["name"] not in model[st]:
                continue
            dt, a = model[st][t["name"]]
            if dt == 6:     # RR layout
                n = t["name"]
                if n in ("Wq", "Wv") and a.ndim == 3:
                    W2 = np.concatenate([a[h].T for h in range(a.shape[0])], axis=0)
                elif n == "Wup":
                    W2 = a.reshape(a.shape[0], -1).T
                else:
                    W2 = a.reshape(-1, a.shape[-1]).T
            else:
                W2 = a.reshape(-1, a.shape[-1])
            res.append((st, t["name"], W2))
    return res, fp8_model

# ---------------------------------------------------------------------------------------------------- exponents


def exps(v, e4m3=True):
    """Exponent class per value: E4M3 E = max(floor(log2|v|) + 7, 1) (rr_step_i.hlsli's E); else floor(log2|v|).
    Zeros give a masked value (returned mask nz)."""
    a = np.abs(v)
    nz = a > 0
    with np.errstate(divide="ignore"):
        fl = np.floor(np.log2(np.where(nz, a, 1.0)))
    e = np.maximum(fl + 7, 1) if e4m3 else fl
    return e.astype(np.int64), nz


def lowbit(v):
    """Exponent of the lowest set bit of each value (exact for float values); +inf for zero."""
    a = np.abs(v)
    m, e = np.frexp(a)
    sig = np.round(np.ldexp(m, 53)).astype(np.uint64)
    low = sig & (~sig + np.uint64(1))
    with np.errstate(divide="ignore"):
        lb = np.log2(np.where(a > 0, low.astype(np.float64), 1.0))
    return np.where(a > 0, e - 53 + lb, np.inf)


def blocks(W, br=16, bk=16):
    """[R, K] -> [nR, nK, br*bk] zero padded."""
    R, K = W.shape
    Rp, Kp = -(-R // br) * br, -(-K // bk) * bk
    P = np.zeros((Rp, Kp))
    P[:R, :K] = W
    return P.reshape(Rp // br, br, Kp // bk, bk).transpose(0, 2, 1, 3).reshape(Rp // br, Kp // bk, br * bk)


def spans(X, axis, e4m3=True):
    """Exponent span (max E - min E over nonzeros) along axis; -1 where all zero."""
    e, nz = exps(X, e4m3)
    hi = np.where(nz, e, -10 ** 6).max(axis=axis)
    lo = np.where(nz, e, 10 ** 6).min(axis=axis)
    return np.where(nz.any(axis=axis), hi - lo, -1)


def bits_needed(X, axis):
    """Bits of magnitude of the integer form with the best power-of-two scale (lowest set bit): exact planes =
    ceil(bits / 7). -1 where all zero."""
    a = np.abs(X)
    nz = a > 0
    lb = np.where(nz, lowbit(X), np.inf).min(axis=axis)
    mx = a.max(axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.floor(np.log2(np.where(mx > 0, mx, 1.0)) - np.where(np.isfinite(lb), lb, 0)) + 1
    return np.where(nz.any(axis=axis), b, -1).astype(np.int64)


def digits_from_span(span, mbits):
    """rr_step_i.hlsli's digit count: ceil((mbits + span) / 7) signed 7-bit digits (mbits = 4 for E4M3)."""
    return np.where(span < 0, 0, np.ceil((mbits + np.maximum(span, 0)) / 7.0)).astype(np.int64)


def span_hist(sp):
    sp = sp[sp >= 0]
    n = max(1, sp.size)
    return {"n": int(sp.size), "le3": float(np.mean(sp <= 3)) if sp.size else 0.0,
            "le10": float(np.mean(sp <= 10)) if sp.size else 0.0, "gt10": float(np.mean(sp > 10)) if sp.size else 0.0,
            "hist": [int(np.count_nonzero(sp == k)) for k in range(0, 15)] + [int(np.count_nonzero(sp >= 15))],
            "mean": float(sp.mean()) if sp.size else 0.0, "_n": n}

# ---------------------------------------------------------------------------------------------------- schemes


def q_sym(W, s, qmax=127, qmin=None):
    qmin = -qmax if qmin is None else qmin
    s = np.where(s > 0, s, 1.0)
    return np.clip(np.round(W / s), qmin, qmax) * s


def s1_tensor(W):
    return q_sym(W, np.abs(W).max() / 127)


def s2_row(W):
    return q_sym(W, np.abs(W).max(axis=1, keepdims=True) / 127)


def s3_row_mse(W, fracs=np.linspace(0.25, 1.0, 31)):
    m = np.abs(W).max(axis=1, keepdims=True)
    best, berr = None, None
    for f in fracs:
        q = q_sym(W, m * f / 127)
        err = ((q - W) ** 2).sum(axis=1, keepdims=True)
        if best is None:
            best, berr = q, err
        else:
            take = err < berr
            best = np.where(take, q, best)
            berr = np.where(take, err, berr)
    return best


def pow2_block(W, imax, br=16, bk=16):
    """Per (br x bk) block power-of-two scale s = smallest 2^e with max|w| / s <= imax; round to nearest."""
    R, K = W.shape
    B = blocks(W, br, bk)
    mx = np.abs(B).max(axis=2, keepdims=True)
    with np.errstate(divide="ignore"):
        e = np.ceil(np.log2(np.where(mx > 0, mx, 1.0) / imax))
    s = np.exp2(e)
    Q = np.clip(np.round(B / s), -imax, imax) * s
    nR, nK = B.shape[:2]
    Q = Q.reshape(nR, nK, br, bk).transpose(0, 2, 1, 3).reshape(nR * br, nK * bk)
    return Q[:R, :K]


def s4_block1(W):
    return pow2_block(W, 127)


def s5_block2(W):
    return pow2_block(W, 127 * 128 + 127)


SCHEMES = {"S1": s1_tensor, "S2": s2_row, "S3": s3_row_mse, "S4": s4_block1, "S5": s5_block2}


def err_stats(W, Q):
    nz = W != 0
    d = Q - W
    return {"sq_err": float((d ** 2).sum()), "sq_ref": float((W ** 2).sum()), "nz": int(nz.sum()),
            "nz_to_zero": int((nz & (Q == 0)).sum()), "exact": int((d == 0).sum()), "n": int(W.size)}


def merge_err(a, b):
    return {k: a.get(k, 0) + b[k] for k in b}


def rel_rms(st):
    return (st["sq_err"] / st["sq_ref"]) ** 0.5 if st["sq_ref"] > 0 else 0.0
