import sys, numpy as np
from exact import F64
c = sys.argv[1]; n = int(sys.argv[2])
z = np.load(f"inputs/{c}.npz")
A = z["A"][:n].astype(np.float64); B = z["B"][:n].astype(np.float64); C = z["C"][:n].astype(np.float64)
D = np.fromfile(f"out/{c}.f32.0.bin", np.float32).reshape(-1, 16, 16)[:n].astype(np.float64)
P = A[:, :, None, :] * B.transpose(0, 2, 1)[:, None, :, :]
T = np.concatenate([P, C[..., None]], -1)       # 17 terms, C last
def expo(x):
    return np.frexp(x)[1] - 1
def fl(t, q, mode):
    if mode == "floor": return np.floor(t / q) * q
    if mode == "trunc": return np.trunc(t / q) * q
    if mode == "rne": return np.round(t / q) * q
def final(s, mode):
    if mode == "rn": return F64.rnd((s, np.zeros_like(s)))
    f = s.astype(np.float32).astype(np.float64)  # RN; then adjust for directed modes
    if mode == "rz":
        over = np.abs(f) > np.abs(s); f = np.where(over, np.nextafter(f.astype(np.float32), np.float32(0)).astype(np.float64), f); return f
    if mode == "floor":
        over = f > s; return np.where(over, np.nextafter(f.astype(np.float32), np.float32(-np.inf)).astype(np.float64), f)
for inc_c in (True, False):
    TT = T if inc_c else P
    mx = np.abs(TT).max(-1)
    E = np.where(mx > 0, expo(np.where(mx > 0, mx, 1)), 0)
    for F in (23, 24, 25, 26, 27, 28):
        q = np.exp2(E - F)[..., None]
        for mode in ("floor", "trunc"):
            s = fl(TT, q, mode).sum(-1)
            if not inc_c: s = s + C   # C exact added (not exact in f64 always, ok for exploration)
            for fm in ("rn", "rz", "floor"):
                M = final(s, fm)
                bad = int((M != D).sum())
                print(f"incC={inc_c} F={F} {mode:5s} final={fm:5s} mismatches {bad} / {D.size}")
