"""Bundles inputs and raw GPU outputs per class into results/<class>.npz for replay on another GPU / harness.
Keys: A [N,16,16] float16 (row m, column k), B [N,16,16] float16 (row k, column n), C [N,16,16] float32,
A8 / B8 [N,16,16] uint8 E4M3 codes (E4M3 classes only), kind [N] sub-kind, D_f32 [N,16,16] float32 (run 0 of 3, D_f32_3runs_identical says whether all 3 agreed; 
v_wmma_f32_16x16x16_f16 on C), D_f16 [N,16,16] float16 (v_wmma_f16_16x16x16_f16 on C16 = RN16(C)), plus class extras."""
import os
import numpy as np
os.makedirs("results", exist_ok=True)
for c in ("e4_random", "p1b", "adv", "specials", "f16_full", "kperm", "pos_shift", "pos_perm"):
    z = dict(np.load(f"inputs/{c}.npz"))
    with np.errstate(over="ignore"):
        z["C16"] = z["C"].astype(np.float32).astype(np.float16)
    for suf, dt in (("f32", np.float32), ("f16", np.float16)):
        if os.path.exists(f"out/{c}.{suf}.0.bin"):
            runs = [np.fromfile(f"out/{c}.{suf}.{r}.bin", dt).reshape(-1, 16, 16) for r in range(3)]
            z[f"D_{suf}"] = runs[0]
            z[f"D_{suf}_3runs_identical"] = np.array(all(np.array_equal(runs[0].view(np.uint8), x.view(np.uint8)) for x in runs))
    np.savez_compressed(f"results/{c}.npz", **z)
for c, sufs, dt in (("int_s8", ("wrap", "sat"), np.int32), ("int_u8", ("wrap",), np.uint32)):
    z = dict(np.load(f"inputs/{c}.npz"))
    for s in sufs:
        runs = [np.fromfile(f"out/{c}.{s}.{r}.bin", dt).reshape(-1, 16, 16) for r in range(3)]
        z[f"D_{s}"] = runs[0]
        z[f"D_{s}_3runs_identical"] = np.array(all(np.array_equal(runs[0], x) for x in runs))
    np.savez_compressed(f"results/{c}.npz", **z)
print("packed", sorted(os.listdir("results")))
