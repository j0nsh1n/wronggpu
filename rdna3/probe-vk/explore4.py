import sys, numpy as np
c = sys.argv[1]; n = int(sys.argv[2])
z = np.load(f"inputs/{c}.npz")
A = z["A"][:n].astype(np.float64); B = z["B"][:n].astype(np.float64); C = z["C"][:n].astype(np.float64)
D = np.fromfile(f"out/{c}.f32.0.bin", np.float32).reshape(-1, 16, 16)[:n].astype(np.float64)
P = A[:, :, None, :] * B.transpose(0, 2, 1)[:, None, :, :]
ex = P.sum(-1) + C
M = ex.astype(np.float32).astype(np.float64)
bad = np.argwhere((M != D) & (C == 0) & ((P != 0).sum(-1) <= 3))
for p, m, nn in bad[:8]:
    t = P[p, m, nn]; nzk = np.nonzero(t)[0]
    print("p", p, "m", m, "n", nn, "D", D[p, m, nn].hex(), "exact", ex[p, m, nn].hex(), "terms", [(int(k), t[k].hex(), A[p,m,k].hex(), B[p,k,nn].hex()) for k in nzk])
