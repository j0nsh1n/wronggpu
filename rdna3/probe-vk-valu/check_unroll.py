"""Fail if any [[unroll]] loop in the given SPIR-V files has a constant trip bound above LIMIT.

RADV unrolls loops marked Unroll whatever their size; a bound of 2^31 (a constant-folded divide by zero) made the
driver's compiler grow to 16 GB. Usage: python3 check_unroll.py file.spv ...   (needs spirv-dis)
"""
import re
import subprocess
import sys

LIMIT = 1024


def bad_loops(path):
    text = subprocess.run(["spirv-dis", "--raw-id", path], check=True, capture_output=True, text=True).stdout
    lines = text.splitlines()
    consts = {}
    for l in lines:
        m = re.match(r"\s*(%\w+) = OpConstant %\w+ (-?\d+)$", l)
        if m:
            consts[m.group(1)] = int(m.group(2))
    bad = []
    for n, l in enumerate(lines):
        if "OpLoopMerge" not in l or "Unroll" not in l:
            continue
        for k in lines[n + 1:n + 12]:
            m = re.search(r"Op[SU]LessThan(?:Equal)? %\w+ %\w+ (%\w+)$", k)
            if m:
                bound = consts.get(m.group(1))
                if bound is not None and abs(bound) > LIMIT:
                    bad.append((n + 1, bound))
                break
    return bad


if __name__ == "__main__":
    rc = 0
    for p in sys.argv[1:]:
        for line, bound in bad_loops(p):
            print(f"{p}: unrolled loop at SPIR-V line {line} has bound {bound} > {LIMIT}")
            rc = 1
    sys.exit(rc)
