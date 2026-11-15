"""Extract the timed loop body from an ACO "Final Assembly" dump and count its instructions by class.

The loop is the span from the label that a backward conditional branch targets up to that branch. Usage:
    python3 disasm_loop.py disasm/X.txt [--list]
"""
import collections
import re
import sys

NOP = ("s_delay_alu", "s_waitcnt", "s_clause", "s_nop", "s_cbranch", "s_branch", "s_waitcnt_depctr", "s_set_inst_prefetch_distance")


def final_asm(path):
    text = open(path).read()
    i = text.index("Final Assembly")
    return [l.split(";")[0].rstrip() for l in text[i:].splitlines()[1:]]


def loop_body(lines):
    labels = {}
    for n, l in enumerate(lines):
        m = re.match(r"^(BB\d+):", l)
        if m:
            labels[m.group(1)] = n
    best = None
    for n, l in enumerate(lines):
        m = re.match(r"^\s+s_cbranch_\w+\s+(BB\d+)", l)
        if m and m.group(1) in labels and labels[m.group(1)] < n:
            span = (labels[m.group(1)], n)
            if best is None or span[1] - span[0] > best[1] - best[0]:
                best = span
    if best is None:
        return []
    return [l.strip() for l in lines[best[0]:best[1] + 1] if l.startswith("\t") or l.startswith(" ")]


def classify(op):
    if op.startswith("v_wmma"):
        return "wmma"
    if op.startswith("v_dual_"):
        return "vopd"
    if op.startswith("v_"):
        return "valu"
    if op.startswith("ds_"):
        return "lds"
    if op.startswith(("buffer_", "global_", "flat_", "scratch_")):
        return "vmem"
    if op.startswith(("s_load", "s_buffer_load")):
        return "smem"
    if op.startswith(NOP):
        return "ctl"
    if op.startswith("s_"):
        return "salu"
    return "other"


def summarize(path):
    body = loop_body(final_asm(path))
    cls = collections.Counter()
    ops = collections.Counter()
    for l in body:
        if not l or l.endswith(":"):
            continue
        op = l.split()[0]
        cls[classify(op)] += 1
        if classify(op) == "vopd":
            parts = l.split("::")
            ops[op] += 1
            ops[parts[1].split()[0] if len(parts) > 1 else "?"] += 1
        else:
            ops[op] += 1
    return body, cls, ops


if __name__ == "__main__":
    body, cls, ops = summarize(sys.argv[1])
    print("classes:", dict(cls))
    print("issue slots (valu + vopd + wmma):", cls["valu"] + cls["vopd"] + cls["wmma"])
    print("ops:", ", ".join(f"{k} {v}" for k, v in sorted(ops.items(), key=lambda kv: -kv[1])))
    if "--list" in sys.argv:
        print("\n".join(body))
