import sys, numpy as np
from exact import F64
from models import MODELS
c = sys.argv[1]; n = int(sys.argv[2])
z = np.load(f"inputs/{c}.npz")
A = z["A"][:n].astype(np.float64); B = z["B"][:n].astype(np.float64); C = z["C"][:n].astype(np.float64)
D = np.fromfile(f"out/{c}.f32.0.bin", np.float32).reshape(-1, 16, 16)[:n]
P = A[:, :, None, :] * B.transpose(0, 2, 1)[:, None, :, :]
M = MODELS["M-exact"](P, C, F64, np.float32).astype(np.float32)
def oi(x):
    i = x.view(np.int32).astype(np.int64); return np.where(i < 0, -(i & 0x7fffffff), i)
d = oi(D) - oi(M)
print("ulp hist", np.unique(np.clip(d, -10, 10), return_counts=True))
kind = z["kind"][:n]
for k in range(8):
    s = kind == k
    print(k, (d[s] != 0).mean(), "C==0 subset", (d[s][C[s] == 0] != 0).mean())
# relative to sign: does D move toward zero vs exact?
ex = P.sum(-1) + C
nz = d != 0
print("toward zero frac", np.mean(np.abs(D[nz].astype(np.float64)) < np.abs(ex[nz])))
i = np.argwhere(nz)[:5]
for p, m, nn in i:
    terms = sorted(list(P[p, m, nn]) + [C[p, m, nn]], key=abs)
    print(D[p, m, nn], M[p, m, nn], ex[p, m, nn], "C", C[p, m, nn], "maxterm", max(abs(t) for t in terms))
