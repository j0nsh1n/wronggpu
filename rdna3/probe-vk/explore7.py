import sys, numpy as np
c = sys.argv[1]; n = int(sys.argv[2])
z = np.load(f"inputs/{c}.npz")
A = z["A"][:n].astype(np.float64); B = z["B"][:n].astype(np.float64); C = z["C"][:n].astype(np.float64)
D = np.fromfile(f"out/{c}.f32.0.bin", np.float32).reshape(-1, 16, 16)[:n].astype(np.float64)
P = A[:, :, None, :] * B.transpose(0, 2, 1)[:, None, :, :]
ex = P.sum(-1) + C
rel = np.abs(D - ex) / np.maximum(np.abs(P).max(-1), np.abs(C))
idx = np.argwhere((rel > 1e-4) & (C == 0))
print("outputs with error > 1e-4 x max term (C=0):", len(idx), "of", int((C == 0).sum()))
for p, m, nn in idx[:6]:
    t = P[p, m, nn]; k = np.nonzero(t)[0]
    print(f"D {D[p,m,nn]!r} exact {ex[p,m,nn]!r}  terms {[(int(i), t[i]) for i in k]}")
