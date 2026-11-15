import numpy as np, subprocess, os, sys
rng = np.random.default_rng(2)
N = 64
env = dict(os.environ)
def run(A, B, C, tag):
    A.astype(np.float16).tofile(f"diag/{tag}A.bin"); B.astype(np.float16).tofile(f"diag/{tag}B.bin"); C.astype(np.float32).tofile(f"diag/{tag}C.bin")
    subprocess.run(["nice", "./wmma_probe", "run", "spv/f16_f32.spv", str(N), "2", "4", f"diag/{tag}A.bin", f"diag/{tag}B.bin", f"diag/{tag}C.bin", f"diag/{tag}", "3"], check=True, stdout=subprocess.DEVNULL)
    Ds = [np.fromfile(f"diag/{tag}.{r}.bin", np.float32).reshape(-1, 16, 16) for r in range(3)]
    ex = np.einsum("pmk,pkn->pmn", A.astype(np.float16).astype(np.float64), B.astype(np.float16).astype(np.float64)) + C.astype(np.float32)
    print(f"{tag:28s} mismatches {[int((D != ex).sum()) for D in Ds]} of {ex.size}; reps identical {all((Ds[0].view(np.uint32) == D.view(np.uint32)).all() for D in Ds)}")
    return Ds[0], ex
z = np.zeros((N, 16, 16))
run(rng.integers(0, 8, (N, 16, 16)), rng.integers(0, 8, (N, 16, 16)), z, "pos_pos_C0")
run(rng.integers(-8, 0, (N, 16, 16)), rng.integers(0, 8, (N, 16, 16)), z, "negA_posB_C0")
run(rng.integers(-8, 8, (N, 16, 16)), rng.integers(0, 8, (N, 16, 16)), z, "mixA_posB_C0")
run(rng.integers(0, 8, (N, 16, 16)), rng.integers(-8, 8, (N, 16, 16)), z, "posA_mixB_C0")
run(rng.integers(0, 8, (N, 16, 16)), rng.integers(0, 8, (N, 16, 16)), -rng.integers(0, 100, (N, 16, 16)), "pos_pos_negC")
# single non-zero product per output: A = diag(a), B random (mixed)
A = np.zeros((N, 16, 16)); idx = np.arange(16); A[:, idx, idx] = rng.integers(-8, 8, (N, 16))
D, ex = run(A, rng.integers(-8, 8, (N, 16, 16)), z, "single_product_mixed")
# two products: one positive, one negative
A = np.zeros((N, 16, 16)); A[:, :, 0] = 3; A[:, :, 1] = 1
B = np.zeros((N, 16, 16)); B[:, 0, :] = 5; B[:, 1, :] = -rng.integers(1, 8, (N, 16))
D, ex = run(A, B, z, "two_products_pm")
bad = np.argwhere(D != ex)
for p, m, n in bad[:6]:
    print("   ", D[p, m, n], ex[p, m, n], "B1", B[p, 1, n])
