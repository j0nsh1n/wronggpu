import sys, numpy as np
from exact import F64
c = sys.argv[1]; n = int(sys.argv[2])
z = np.load(f"inputs/{c}.npz")
A = z["A"][:n].astype(np.float64); B = z["B"][:n].astype(np.float64); C = z["C"][:n].astype(np.float64)
D = np.fromfile(f"out/{c}.f32.0.bin", np.float32).reshape(-1, 16, 16)[:n].astype(np.float64)
A16 = z["A"][:n]; B16 = z["B"][:n]
P = A[:, :, None, :] * B.transpose(0, 2, 1)[:, None, :, :]
negz = (np.signbit(A16)[:, :, None, :] ^ np.signbit(B16).transpose(0, 2, 1)[:, None, :, :]) & (P == 0)
sel = (C == 0)
mx = np.abs(P).max(-1)
E = np.frexp(np.where(mx > 0, mx, 1))[1] - 1
def final(s, mode):
    f = s.astype(np.float32).astype(np.float64)
    if mode == "rn": return f
    nf = lambda x, d: np.nextafter(x.astype(np.float32), np.float32(d)).astype(np.float64)
    if mode == "floor": return np.where(f > s, nf(f, -np.inf), f)
    if mode == "rz": return np.where(np.abs(f) > np.abs(s), nf(f, 0), f)
for W in (23, 24, 25, 26):
    u = np.exp2(E - W)[..., None]
    for neg in ("ones", "floor-1", "floor"):
        mag = np.floor(np.abs(P) / u) * u
        if neg == "ones": T = np.where(P < 0, -mag - u, mag)
        elif neg == "floor-1": T = np.where(P < 0, np.floor(P / u) * u - u, mag)
        else: T = np.floor(P / u) * u
        for nz in (False, True):
            TT = np.where(negz, -u, T) if nz else T
            s = TT.sum(-1)
            for fm in ("rn", "floor", "rz"):
                M = final(s, fm)
                print(f"W={W} neg={neg:8s} negzero={nz!s:5} final={fm:5s} mismatch(C=0) {int((M != D)[sel].sum()):7d} / {int(sel.sum())}")
