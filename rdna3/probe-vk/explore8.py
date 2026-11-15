import sys, numpy as np
c = sys.argv[1]; n = int(sys.argv[2])
z = np.load(f"inputs/{c}.npz")
A = z["A"][:n].astype(np.float64); B = z["B"][:n].astype(np.float64); C = z["C"][:n].astype(np.float64)
D = np.fromfile(f"out/{c}.f32.0.bin", np.float32).reshape(-1, 16, 16)[:n].astype(np.float64)
P = A[:, :, None, :] * B.transpose(0, 2, 1)[:, None, :, :]
ex = P.sum(-1) + C
for label, sel in (("C=0", C == 0), ("C!=0", C != 0)):
    T = np.concatenate([P, C[..., None]], -1)
    mx = np.abs(T).max(-1)
    E = np.frexp(np.where(mx > 0, mx, 1))[1] - 1
    err = (D - ex) / np.exp2(E - 24.0)
    e = err[sel]
    v, cnt = np.unique(np.round(e, 3), return_counts=True)
    order = np.argsort(-cnt)[:14]
    print(label, "error in units of 2^(E-24): most common", list(zip(v[order], cnt[order])), "min", e.min(), "max", e.max())
    ok = np.isclose(ex, ex.astype(np.float32))
    # how often is the error within [-k, k] units
    for k in (1, 2, 4, 8): print("   |err| <=", k, "units:", np.mean(np.abs(e) <= k))
