"""B = all ones: D[m, n] = sum_k A[m, k] + C[m, n]. Each case = one row of A terms and one C."""
import numpy as np, subprocess
from fractions import Fraction
cases = []   # (list of up to 16 f16 terms, C)
def add(terms, c=0.0): cases.append((terms, c))
add([8, -1]); add([-8, 1]); add([1, -1]); add([-1, 1]); add([8], -1.0); add([-8], 1.0); add([3, 5]); add([-3, -5])
add([1, -2.0**-10]); add([1, -2.0**-14]); add([1], -2.0**-20); add([1], -2.0**-30); add([1], -2.0**-60); add([1], 2.0**-60)
add([-1], 2.0**-30); add([-1], -2.0**-30); add([1] + [0] * 7 + [-0.5]); add([1, -0.5]); add([2, -1]); add([4, -1]); add([4, -3])
add([1, -1, 1]); add([0.0], -1.0); add([-0.0], 1.0); add([1.0], -0.5)
add([1, 2.0**-24]); add([1, -2.0**-24]); add([1], -2.0**-24); add([1], -2.0**-23); add([1], -2.0**-25); add([1], -2.0**-26); add([1], -2.0**-27)
add([2.0**15], -1.0); add([2.0**15, -1]); add([1], -3 * 2.0**-26)
N = 4
A = np.zeros((N, 16, 16), np.float16); B = np.ones((N, 16, 16), np.float16); C = np.zeros((N, 16, 16), np.float32)
slots = [(p, m) for p in range(N) for m in range(16)]
for (t, c), (p, m) in zip(cases, slots):
    A[p, m, :len(t)] = t; C[p, m, :] = c
A.tofile("diag/tA.bin"); B.tofile("diag/tB.bin"); C.tofile("diag/tC.bin")
subprocess.run(["nice", "./wmma_probe", "run", "spv/f16_f32.spv", str(N), "2", "4", "diag/tA.bin", "diag/tB.bin", "diag/tC.bin", "diag/t", "1"], check=True, stdout=subprocess.DEVNULL)
D = np.fromfile("diag/t.0.bin", np.float32).reshape(N, 16, 16)
for (t, c), (p, m) in zip(cases, slots):
    ex = sum(Fraction(float(x)) for x in A[p, m].astype(np.float64)) + Fraction(float(C[p, m, 0]))
    d = D[p, m, 0]; same = np.all(D[p, m] == d)
    print(f"terms {t!s:28} C {c!r:24} -> {float(d).hex():24} exact {float(ex).hex():24} {'OK' if Fraction(float(d)) == ex else 'DIFF'}{'' if same else ' (row varies)'}")
