"""Independent spot check: F64 exact vs the Python-int backend, and q8 flip counts via fp8_encode (a different path)."""
import sys
sys.path.insert(2, "."); sys.path.insert(0, "../probe-vk"); sys.path.insert(1, "../src/clean/reference")
import numpy as np
from exact import F64, INT
from models import MODELS
from fp8 import q8, fp8_encode
import post_quant as pq

z = np.load("../probe-vk/results/adv.npz")
sel = np.arange(0, 64)             # sub-kinds 0..7, 8 problems each
A = z["A"][sel].astype(np.float64); B = z["B"][sel].astype(np.float64); C = z["C"][sel].astype(np.float64)
P = A[:, :, None, :] * B.transpose(0, 2, 1)[:, None, :, :]
ok = np.isfinite(C) & np.isfinite(z["D_f32"][sel])
e1 = MODELS["M-exact"](P, C, F64, np.float32)
e2 = MODELS["M-exact"](INT.const(P), C, INT, np.float32)
fin = ok & np.isfinite(e1) & np.isfinite(e2)
print("exact F64 vs INT backends equal:", bool((e1[fin] == e2[fin]).all()), int(fin.sum()))
D = z["D_f32"][sel].astype(np.float64)
f_a = int((q8(D)[fin] != q8(e1)[fin]).sum())
f_b = int((fp8_encode(D)[fin] != fp8_encode(e1)[fin]).sum())
print("q8 flips: values path", f_a, " code path", f_b)
d8, _ = pq.cmp_stats(D, e1, q8(D), q8(e1), D, e1)
print("script path", int((d8[fin] > 0).sum()))
# probe can show absence: GPU output replaced by exact must give zero flips
print("flips with D := exact:", int((q8(e1)[fin] != q8(e1)[fin]).sum()))
ca, cb = fp8_encode(D)[fin], fp8_encode(e1)[fin]
va, vb = q8(D)[fin], q8(e1)[fin]
zero_sign = (ca != cb) & (va == vb)
print("code differs but value equal (+0 vs -0):", int(zero_sign.sum()), "; real flips by code path:", int(((ca != cb) & (va != vb)).sum()))
