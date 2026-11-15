import numpy as np, sys
rng = np.random.default_rng(1)
N = 8
A = rng.integers(1, 64, (N, 16, 16)).astype(np.float16)   # small integers: every sum exact
B = np.tile(np.eye(16, dtype=np.float16), (N, 1, 1))
C = np.zeros((N, 16, 16), np.float32)
A.tofile("diag/idA.bin"); B.tofile("diag/idB.bin"); C.tofile("diag/idC.bin")
np.savez("diag/id.npz", A=A, B=B, C=C)
A2 = rng.integers(-8, 8, (N, 16, 16)).astype(np.float16); B2 = rng.integers(-8, 8, (N, 16, 16)).astype(np.float16)
C2 = rng.integers(-100, 100, (N, 16, 16)).astype(np.float32)
A2.tofile("diag/iA.bin"); B2.tofile("diag/iB.bin"); C2.tofile("diag/iC.bin")
np.savez("diag/int.npz", A=A2, B=B2, C=C2)
