"""Model M end to end on the CPU reference with int8 matrix multiplies (steps 2-4 of the study).

    python m_e2e.py <model.wgcm> --src <WrongGPU src root> [--render 128x72] [--cases pan,disocclusion,...] --json OUT

The network (enc0 -> enc1 ... dec1 -> dec0 -> post reconstruction) is the repo's float64 reference
(clean/reference), re-composed here so every matrix product goes through one hook. The hook quantizes the weight
matrix (scheme S1-S5) and optionally the activation operand (per tensor / per token / per 16x16 tile, one or two
planes). With no quantization the re-composition is checked equal to the reference functions.
Attention A V stays FP16 x FP16 as in the reference (gfx11 has an F16 matrix instruction for it).

Writes aggregate numbers only (verdicts, step histograms, PSNR, span shares, multiply counts). No stage dump, no
tensor, nothing per weight or per value is written.
"""
import argparse
import json
import math
import sys
import time

import numpy as np

import qlib as Q

P = argparse.ArgumentParser()
P.add_argument("model")
P.add_argument("--src", required=True)
P.add_argument("--render", default="128x72")
P.add_argument("--cases", default="pan,disocclusion,checker,exposure_bright")
P.add_argument("--runs", default="all")
P.add_argument("--json", required=True)
A = P.parse_args()
sys.path.insert(0, A.src)

from clean.reference import swin as RS, encoder as RE, decoder as RD, post as RP, exposure as RX  # noqa: E402
from clean.reference.fp8 import q8, f16  # noqa: E402
from clean.equiv import probes as PR  # noqa: E402
from clean.equiv.compare import compare_stage  # noqa: E402
from clean.runner.pipeline import read_wgcm, logical_tensors  # noqa: E402

# ---------------------------------------------------------------------------------------------------- quant context


def act_quant(X, how):
    """X [T, K] -> dequantized int8 form. how: tensor | token | tile1 | tile2."""
    nonneg = X.min() >= 0
    qmax, qmin = (255, 0) if nonneg else (127, -127)
    if how == "tensor":
        return Q.q_sym(X, np.abs(X).max() / qmax, qmax, qmin)
    if how == "token":
        return Q.q_sym(X, np.abs(X).max(axis=1, keepdims=True) / qmax, qmax, qmin)
    if how == "tile1":
        return Q.pow2_block(X, 127)
    if how == "tile2":
        return Q.pow2_block(X, 127 * 128 + 127)
    raise ValueError(how)


ACT_SCHEMES = ("tensor", "token", "tile1", "tile2")


class Ctx:
    def __init__(self, wscheme=None, ascheme=None, collect=False):
        self.ws, self.as_, self.collect = wscheme, ascheme, collect
        self.wcache = {}
        self.stats = {}
        self.stage = None

    def W(self, key, W):
        if self.ws in (None, "F32"):
            return W
        if key not in self.wcache:
            self.wcache[key] = Q.SCHEMES[self.ws](W)
        return self.wcache[key]

    def Aop(self, site, X):
        K = X.shape[-1]
        X2 = X.reshape(-1, K)
        if self.collect:
            self._record(site, X2)
        if self.as_ is None:
            return X
        return act_quant(X2, self.as_).reshape(X.shape)

    def _st(self, site):
        return self.stats.setdefault(f"{self.stage}.{site}", {
            "n": 0, "tile_spans": np.zeros(17, np.int64), "tensor_span": -1, "token_spans": np.zeros(17, np.int64),
            "err": {k: [0.0, 0.0, 0, 0] for k in ACT_SCHEMES}, "nonneg": True,
            "wmma": {"base": 0, "tile": 0, "row": 0}, "ovf": [0, 0], "maxbits": 0})

    def _record(self, site, X2):
        s = self._st(site)
        s["n"] += X2.size
        s["nonneg"] = bool(s["nonneg"] and X2.min() >= 0)
        sp_t = Q.spans(Q.blocks(X2), 2)
        s["tile_spans"] += np.bincount(np.clip(sp_t[sp_t >= 0], 0, 16), minlength=17)
        sp_r = Q.spans(X2, 1)
        s["token_spans"] += np.bincount(np.clip(sp_r[sp_r >= 0], 0, 16), minlength=17)
        s["tensor_span"] = max(s["tensor_span"], int(Q.spans(X2.ravel(), 0)))
        nz = X2 != 0
        for k in ACT_SCHEMES:
            d = act_quant(X2, k) - X2
            e = s["err"][k]
            e[0] += float((d ** 2).sum()); e[1] += float((X2 ** 2).sum())
            e[2] += int((nz & (X2 + d == 0)).sum()); e[3] += int(nz.sum())

    def count(self, site, X2, W2):
        """Exact adaptive digit planes (step iv): WMMAs per baseline WMMA for X2 [T, K] times W2 [R, K]^T, and the
        int32 bound of the per-row-scale variant. Weight and activation digits from exponent spans (rr_step_i)."""
        if not self.collect:
            return
        s = self._st(site)
        XB, WB = Q.blocks(X2), Q.blocks(W2)                       # [nT, nK, 256], [nR, nK, 256]
        dX = Q.digits_from_span(Q.spans(XB, 2), 4)                # [nT, nK]
        dW = Q.digits_from_span(Q.spans(WB, 2), 4)
        nT, nK = dX.shape
        nR = dW.shape[0]
        s["wmma"]["base"] += nT * nR * nK
        s["wmma"]["tile"] += int((dX.sum(0) * dW.sum(0)).sum())
        # per-row scales over the whole K (free): digits uniform per 16-row tile = max over its rows
        T, K = X2.shape
        R = W2.shape[0]
        rX = Q.digits_from_span(Q.spans(X2, 1), 4)
        rW = Q.digits_from_span(Q.spans(W2, 1), 4)
        tX = np.pad(rX, (0, -T % 16)).reshape(-1, 16).max(1)
        tW = np.pad(rW, (0, -R % 16)).reshape(-1, 16).max(1)
        s["wmma"]["row"] += int(tX.sum() * tW.sum() * nK)
        # int32 bound per output: sum |a'| |w'| with a' = a / 2^(Emin_row - 10)
        eX, nzX = Q.exps(X2)
        eW, nzW = Q.exps(W2)
        bX = np.where(nzX, eX, 99).min(1)
        bW = np.where(nzW, eW, 99).min(1)
        L = (np.abs(X2) @ np.abs(W2).T) * np.exp2(10 - bX)[:, None] * np.exp2(10 - bW)[None, :]
        s["ovf"][0] += int(np.count_nonzero(L >= 2.0 ** 31)); s["ovf"][1] += L.size
        s["maxbits"] = max(s["maxbits"], int(np.ceil(np.log2(max(L.max(), 1)))))


CTX = Ctx()


def mm(site, wkey, X, W):
    """X [..., K] @ W[out, K]^T through the hook."""
    Wq = CTX.W(wkey, W)
    Xq = CTX.Aop(site, X)
    if CTX.collect:
        CTX.count(site, X.reshape(-1, X.shape[-1]), W)
    if CTX.ws == "F32":
        return (Xq.astype(np.float32) @ Wq.T.astype(np.float32)).astype(np.float64)
    return Xq @ Wq.T

# ---------------------------------------------------------------------------------------------------- network


def block_tokens(xw, p, heads, gelu_fn, st):
    C = xw.shape[-1]
    lead = xw.shape[:-2]
    D = RS.HEAD_DIM
    a = q8(RS.rmsnorm(xw, p["gamma1"], C))
    u = q8(mm("qk", st + ".Wqk", a, p["Wqk"].reshape(heads * D, C)))          # [..., t, h*D]
    u = np.moveaxis(u.reshape(lead + (16, heads, D)), -2, -3)                 # [..., h, t, D]
    v = f16(mm("v", st + ".Wv", a, p["Wv"].reshape(heads * D, C)))
    v = np.moveaxis(v.reshape(lead + (16, heads, D)), -2, -3)
    uq = CTX.Aop("S", u)
    if CTX.collect:
        u2 = u.reshape(-1, 16, D)
        for i in range(0, u2.shape[0], max(1, u2.shape[0] // 64)):           # sampled windows for the count
            CTX.count("S", u2[i], u2[i])
    S = np.einsum("...htd,...hsd->...hts", uq, uq) + p["B"]
    z = RS.TAU * RS.soft_cap(S)
    z = z - z.max(axis=-1, keepdims=True)
    Aw = np.exp(z)
    Aw = f16(Aw / Aw.sum(axis=-1, keepdims=True))
    o = np.einsum("...hts,...hsd->...htd", Aw, v)
    o = np.moveaxis(o, -3, -2).reshape(lead + (16, heads * D))
    x1 = xw + mm("Wp", st + ".Wp", q8(o), p["Wp"]) + p["bp"]
    m = q8(RS.rmsnorm(x1, p["gamma2"], C))
    h = q8(gelu_fn(mm("W1", st + ".W1", m, p["W1"]) + p["b1"]))
    return x1 + mm("W2", st + ".W2", h, p["W2"]) + p["b2"]


def swin_block(x, p, offset, heads, gelu_fn, st):
    H, W, C = x.shape
    rows, cols = RS.window_grid(H, W, offset)
    out = np.full_like(x, np.nan)
    r = rows
    rr = RS.mirror(r, H)[:, None, :, None]
    cc = RS.mirror(cols, W)[None, :, None, :]
    xw = x[rr, cc]
    nl, nk = xw.shape[:2]
    x2 = block_tokens(xw.reshape(nl, nk, 16, C), p, heads, gelu_fn, st).reshape(nl, nk, 4, 4, C)
    rv = (r >= 0) & (r < H)
    cv = (cols >= 0) & (cols < W)
    sel = rv[:, None, :, None] & cv[None, :, None, :]
    R = np.broadcast_to(r[:, None, :, None], sel.shape)
    K = np.broadcast_to(cols[None, :, None, :], sel.shape)
    out[R[sel], K[sel]] = x2[sel]
    return out


def run_stage(st, p, x, offset, skip=None):
    CTX.stage = st
    cfg = RS.STAGES[st]
    p = {k: np.asarray(v, np.float64) for k, v in p.items()}
    if cfg["kind"] == "dec":
        y = np.asarray(x, np.float64)
        Hs, Ws, C = skip.shape
        e = q8(mm("We", st + ".We", y, p["We"]) + p["be"])
        r, c = np.arange(Hs), np.arange(Ws)
        sr, sc = RS.mirror(r // 2, y.shape[0]), RS.mirror(c // 2, y.shape[1])
        k = 2 * (r % 2)[:, None] + (c % 2)[None, :]
        ef = e[sr[:, None], sc[None, :]].reshape(Hs, Ws, 4, C)
        xin = np.take_along_axis(ef, k[:, :, None, None], axis=2)[:, :, 0, :] + skip
    else:
        xin = np.asarray(x, np.float64)
    x2 = swin_block(xin, p, offset, cfg["heads"], RS.gelu, st)
    out = q8(x2)
    if cfg["kind"] == "enc":
        z = np.concatenate([out[0::2, 0::2], out[0::2, 1::2], out[1::2, 0::2], out[1::2, 1::2]], axis=-1)
        return {"skip": out, "merged": q8(mm("Wm", st + ".Wm", z, p["Wm"]) + p["bm"])}
    return {"out": out}


def enc0(x, params):
    CTX.stage = "enc0"
    vals = RE.pixel_values(x, params)
    fw, fh = x["feature_size"]
    t = RE.dither_table(params["T"], x["frame"], fw, fh)
    qv = RE.dither_fp8(vals, t[..., None])
    W = np.asarray(params["W"], np.float64)
    a = qv.reshape(fh // 2, 2, fw // 2, 2, RE.N_VALUES)
    s = np.broadcast_to(np.asarray(params["b"], np.float64), (fh // 2, fw // 2, RE.N_FEATURES))
    for p in range(4):
        dx, dy = p % 2, p // 2
        s = f16(s + mm("W", f"enc0.W{p}", a[:, dy, :, dx, :], W[:, RE.N_VALUES * p:RE.N_VALUES * (p + 1)]))
    feats = q8(np.clip(s, -RE.FEATURE_CLAMP, RE.FEATURE_CLAMP))
    m_near, m_far, m_center = RE.motion_surfaces(x)
    return {"features": feats, "m_near": f16(m_near), "m_far": f16(m_far), "m_center": f16(m_center)}


def dec0(x, params):
    CTX.stage = "dec0"
    rw, rh, fw, fh, dw, dh = RD.check_inputs(x)
    Z = np.asarray(x["features"], np.float64)
    W = np.asarray(params["W"], np.float64)
    u = f16(np.asarray(params["b"], np.float64) + mm("W", "dec0.Wa", Z[..., :32], W[:, :32]))
    v = f16(u + mm("W", "dec0.Wb", Z[..., 32:], W[:, 32:]))
    h = q8(v)
    a = RD.pixel_values(h, rw, rh).astype(np.float32)
    Aa = RD.sigmoid16(a[..., 0:3])
    B = RD.softmax3(a[..., 3:6])
    C = RD.coefficient_c(a[..., 6], a[..., 7], a[..., 8], float(np.asarray(params["S"])), float(np.asarray(params["T"])))
    p = RD.previous_latent(x, rw, rh, dw, dh)
    L = RD.latent_update(a, p)
    cA, cB, cL = RD.unorm8_code(Aa), RD.unorm8_code(B), RD.snorm8_code(L)
    return {"A": cA / 255.0, "B": cB / 255.0, "C": C.astype(np.float32), "latent": cL / 127.0, "h": h}


ORDER = ["enc1", "enc2", "b0", "b1", "b2", "b3", "b4", "b5", "dec2", "dec1"]


def network(x, params, feed=None):
    res = {}
    e = enc0(x, params["enc0"])
    res["enc0.features"] = e["features"]
    y = e["features"] if feed is None else feed["enc0.features"]
    skips = {}
    for st in ORDER:
        off = RS.window_offset(st, 0)
        kind = RS.STAGES[st]["kind"]
        if kind == "enc":
            o = run_stage(st, params[st], y, off)
            res[st + ".skip"], res[st + ".merged"] = o["skip"], o["merged"]
            skips[st] = o["skip"] if feed is None else feed[st + ".skip"]
            y = o["merged"] if feed is None else feed[st + ".merged"]
        elif kind == "dec":
            o = run_stage(st, params[st], y, off, skips["enc2" if st == "dec2" else "enc1"])
            y = res[st] = o["out"]
        else:
            y = res[st] = run_stage(st, params[st], y, off)["out"]
        if feed is not None and kind != "enc":
            y = feed[st]
    xd = {"features": y, "motion": e["m_near"], "latent_prev": np.zeros_like(x["latent"]), "jitter": x["jitter"],
          "jitter_prev": x["jitter_prev"], "reset": x["reset"], "feature_size": x["feature_size"],
          "recon_size": x["recon_size"]}
    d = dec0(xd, params["dec0"])
    for k in ("h", "A", "B", "C", "latent"):
        res["dec0." + k] = d[k]
    dw, dh = x["recon_size"]
    if feed is not None:
        d = dict(d, A=feed["dec0.A"], B=feed["dec0.B"], C=feed["dec0.C"])
    xp = {"color": x["color"], "recon_size": x["recon_size"], "jitter": x["jitter"], "exposure": x["exposure"],
          "exposure_prev": x["exposure_prev"], "reset": True, "history": np.zeros((dh, dw, 3), np.float32),
          "A": d["A"], "B": d["B"], "C": d["C"], "m_near": e["m_near"], "m_center": e["m_center"],
          "m_far": e["m_far"]}
    res["post.display"] = RP.reconstruct(xp, params["post"]["s"])["display"]
    return res


FORMATS = {"dec0.A": "unorm8", "dec0.B": "unorm8", "dec0.C": "fp16", "dec0.latent": "snorm8", "post.display": "fp16"}

# ---------------------------------------------------------------------------------------------------- inputs


def probe_input(case, render, seed=1):
    rw, rh = render
    ow, oh = (3 * rw) // 2, (3 * rh) // 2
    dw, dh = (3 * ow + 1) // 2, (3 * oh + 1) // 2
    c = PR.Ctx(seed, (rw, rh), (ow, oh), 4, 8)
    arr, rec = PR.frame_arrays(case, c, 1, "m")
    color = arr["color"].astype(np.float32)
    pre = float(rec.get("pre_exposure") or 1.0)
    e = float(RX.exposure(color, pre))
    fw, fh = -(-rw // 32) * 32, -(-rh // 32) * 32
    return {"color": color, "depth": arr["depth"], "motion": arr["mv"].astype(np.float32),
            "mv_scale": np.array([1.0, 1.0], np.float32), "mv_bias": np.zeros(2, np.float32),
            "jitter": np.array(rec["jitter"], np.float32), "jitter_prev": np.array(rec["jitter"], np.float32),
            "exposure": e, "exposure_prev": e, "reset": True, "frame": 1, "depth_inverted": False,
            "feature_size": (fw, fh), "recon_size": (dw, dh),
            "reduced_history": np.zeros((dh // 2, dw // 2, 3), np.float32),
            "latent": np.zeros((rh, rw, 4), np.float32), "motion_history": np.zeros((rh, rw, 2), np.float32)}


def self_check(x, params, base):
    """The re-composition equals the repo's reference functions (no quantization)."""
    e = RE.encode(x, params["enc0"])
    ok = {"enc0": bool(np.array_equal(e["features"], base["enc0.features"]))}
    y, skips = e["features"], {}
    for st in ORDER:
        off = RS.window_offset(st, 0)
        kind = RS.STAGES[st]["kind"]
        if kind == "dec":
            o = RS.run_stage(st, params[st], y, off, skips["enc2" if st == "dec2" else "enc1"])
        else:
            o = RS.run_stage(st, params[st], y, off)
        if kind == "enc":
            skips[st] = o["skip"]
            ok[st] = bool(np.array_equal(o["skip"], base[st + ".skip"]) and np.array_equal(o["merged"],
                                                                                            base[st + ".merged"]))
            y = o["merged"]
        else:
            ok[st] = bool(np.array_equal(o["out"], base[st]))
            y = o["out"]
    return ok


def as_format(a, fmt):
    a = np.asarray(a, np.float64)
    if fmt == "unorm8":
        return np.rint(a * 255).astype(np.uint8)
    if fmt == "snorm8":
        return np.rint(a * 127).astype(np.int8)
    if fmt == "fp16":
        return a.astype(np.float16)
    return a.astype(np.float32)


def compare(res, base):
    rows = {}
    for k, ref in base.items():
        fmt = FORMATS.get(k, "fp8")
        st = compare_stage(as_format(res[k], fmt)[None], as_format(ref, fmt)[None], fmt, layout="FHWC", border=0)
        if "error" in st:
            rows[k] = {"verdict": "ERROR", "error": st["error"]}
            continue
        rows[k] = {"verdict": st["verdict"], "size": st["size"], "share_equal": st["share_equal"],
                   "max_step": st["max_step"], "mean_signed": st["mean_signed"], "over_2": st["over_2"] / st["values"],
                   "over_16": st["over_16"] / st["values"]}
    d, b = np.asarray(res["post.display"], np.float64), np.asarray(base["post.display"], np.float64)
    peak = float(b.max())
    mse = float(((d - b) ** 2).mean())
    h, h0 = np.asarray(res["dec0.h"], np.float64), np.asarray(base["dec0.h"], np.float64)
    rows["net_snr_db"] = 10 * math.log10((h0 ** 2).sum() / max(((h - h0) ** 2).sum(), 1e-300))
    rows["image"] = {"psnr_db": (10 * math.log10(peak ** 2 / mse)) if mse > 0 else math.inf, "peak": peak,
                     "max_abs": float(np.abs(d - b).max())}
    return rows


RUNS = [("F32acc", "F32", None), ("W:S1", "S1", None), ("W:S2", "S2", None), ("W:S3", "S3", None), ("W:S4", "S4", None), ("W:S5", "S5", None),
        ("W:S2+A:tensor", "S2", "tensor"), ("W:S2+A:token", "S2", "token"), ("W:S3+A:token", "S3", "token"),
        ("W:S4+A:tile1", "S4", "tile1"), ("W:S5+A:tile2", "S5", "tile2")]


def main():
    global CTX
    params = logical_tensors(read_wgcm(A.model))
    params = {s: {k: np.asarray(v, np.float64) for k, v in t.items()} for s, t in params.items()}
    params["enc0"]["T"] = params["enc0"]["T"].astype(np.float64)
    render = tuple(int(v) for v in A.render.split("x"))
    out = {"render": render, "cases": {}, "activations": {}, "wmma": {}}
    runs = RUNS if A.runs == "all" else [r for r in RUNS if r[0] in A.runs.split(",")]
    for case in A.cases.split(","):
        t0 = time.time()
        x = probe_input(case, render)
        CTX = Ctx(collect=True)
        base = network(x, params)
        out["cases"][case] = {"self_check": self_check(x, params, base), "runs": {}}
        for site, s in CTX.stats.items():
            agg = out["activations"].setdefault(site, {"n": 0, "tile_spans": [0] * 17, "token_spans": [0] * 17,
                                                       "tensor_span": -1, "err": {k: [0, 0, 0, 0] for k in ACT_SCHEMES},
                                                       "nonneg": True, "wmma": {"base": 0, "tile": 0, "row": 0},
                                                       "ovf": [0, 0], "maxbits": 0})
            agg["n"] += s["n"]
            agg["tile_spans"] = [a + int(b) for a, b in zip(agg["tile_spans"], s["tile_spans"])]
            agg["token_spans"] = [a + int(b) for a, b in zip(agg["token_spans"], s["token_spans"])]
            agg["tensor_span"] = max(agg["tensor_span"], s["tensor_span"])
            agg["nonneg"] = agg["nonneg"] and s["nonneg"]
            for k in ACT_SCHEMES:
                agg["err"][k] = [a + b for a, b in zip(agg["err"][k], s["err"][k])]
            for k in ("base", "tile", "row"):
                agg["wmma"][k] += s["wmma"][k]
            agg["ovf"] = [agg["ovf"][0] + s["ovf"][0], agg["ovf"][1] + s["ovf"][1]]
            agg["maxbits"] = max(agg["maxbits"], s["maxbits"])
        for name, ws, as_ in runs:
            CTX = Ctx(ws, as_)
            out["cases"][case]["runs"][name] = compare(network(x, params), base)
            CTX = Ctx(ws, as_)
            iso = compare(network(x, params, feed=base), base)
            out["cases"][case]["runs"][name]["isolated"] = {k: v["verdict"] for k, v in iso.items()
                                                             if isinstance(v, dict) and "verdict" in v}
            out["cases"][case]["runs"][name]["isolated_equal"] = {k: v["share_equal"] for k, v in iso.items()
                                                                  if isinstance(v, dict) and "share_equal" in v}
        out["cases"][case]["seconds"] = time.time() - t0
        print(case, "done", round(time.time() - t0, 1), "s", out["cases"][case]["self_check"], flush=True)
    with open(A.json, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
