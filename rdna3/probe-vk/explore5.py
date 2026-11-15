import sys, numpy as np
c = sys.argv[1]; n = int(sys.argv[2])
z = np.load(f"inputs/{c}.npz")
A = z["A"][:n].astype(np.float64); B = z["B"][:n].astype(np.float64); C = z["C"][:n].astype(np.float64)
D = np.fromfile(f"out/{c}.f32.0.bin", np.float32).reshape(-1, 16, 16)[:n]
P = A[:, :, None, :] * B.transpose(0, 2, 1)[:, None, :, :]
ex = P.sum(-1) + C
M = ex.astype(np.float32)
repr_ = (M.astype(np.float64) == ex)
def oi(x):
    i = x.view(np.int32).astype(np.int64); return np.where(i < 0, -(i & 0x7fffffff), i)
d = oi(D) - oi(M)
mag = np.where(M < 0, -d, d)   # positive: |D| larger
neg = (P < 0).sum(-1) + (C < 0)
pos = (P > 0).sum(-1) + (C > 0)
lsb = D.view(np.uint32) & 1
s = repr_ & (C == 0)
print("C=0 exact-representable outputs:", s.sum(), "mismatch", (d[s] != 0).sum())
for nn in range(0, 5):
    t = s & (neg == nn)
    print(" #neg terms", nn, "count", t.sum(), "mismatch", (d[t] != 0).sum(), "mag-delta hist", np.unique(mag[t], return_counts=True))
t = s & (d != 0)
print("LSB of D in mismatches", np.unique(lsb[t], return_counts=True))
# all-negative-terms vs mixed
t = s & (pos == 0) & (neg > 0)
print("all terms negative:", t.sum(), "mismatch", (d[t] != 0).sum())
t = s & (pos > 0) & (neg == 0)
print("all terms positive:", t.sum(), "mismatch", (d[t] != 0).sum())
