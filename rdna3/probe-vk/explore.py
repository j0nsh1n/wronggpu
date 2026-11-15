import sys, numpy as np
from exact import F64
from models import MODELS
c = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
z = np.load(f"inputs/{c}.npz")
A = z["A"][:n].astype(np.float64); B = z["B"][:n].astype(np.float64); C = z["C"][:n].astype(np.float64)
D = np.fromfile(f"out/{c}.f32.0.bin", np.float32).reshape(-1, 16, 16)[:n].astype(np.float64)
P = A[:, :, None, :] * B.transpose(0, 2, 1)[:, None, :, :]   # [p, m, n, k]
for name, f in MODELS.items():
    M = f(P, C, F64, np.float32)
    bad = ~((M == D) | (np.isnan(M) & np.isnan(D)))
    print(f"{name:16s} {int(bad.sum()):8d} / {D.size}")
