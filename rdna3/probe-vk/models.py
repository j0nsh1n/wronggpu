"""Candidate models of one 16x16x16 multiply-accumulate D = R(C + sum_k A[m,k] B[k,n]).

A model takes the exact products P[..., k] (k = 0..15 in the instruction's K order), the accumulator C, a backend from
exact.py and the result type; it returns the rounded result as float64. R = round to nearest even in that type."""
import numpy as np


def _groups_model(groups, c_first=True):
    """acc = C; for each group g: acc = R(acc + exact sum of the group's products)."""
    def f(P, C, X, dt):
        acc = None
        if c_first:
            acc = C
        for g in groups:
            s = X.psum(P, list(g))
            acc = s if acc is None else X.add(X.const(X.rnd(acc, dt)) if acc is not C else X.const(C), s)
            acc = X.const(X.rnd(acc, dt))
        if not c_first:
            acc = X.const(X.rnd(X.add(acc, X.const(C)), dt))
        return X.rnd(acc, dt)
    return f


def _tree_model(c_where="last"):
    """Pairwise tree over the 16 products with a rounding at every node, then C added (and rounded) at the end."""
    def f(P, C, X, dt):
        level = [X.const(X.rnd(X.psum(P, [k]), dt)) for k in range(16)]
        while len(level) > 1:
            level = [X.const(X.rnd(X.add(level[i], level[i + 1]), dt)) for i in range(0, len(level), 2)]
        return X.rnd(X.add(level[0], X.const(C)), dt)
    return f


def _pre_round_groups(groups):
    """Each group's product sum rounded on its own first, the group results then added to C in order (rounding each)."""
    def f(P, C, X, dt):
        acc = X.const(C)
        for g in groups:
            s = X.const(X.rnd(X.psum(P, list(g)), dt))
            acc = X.const(X.rnd(X.add(acc, s), dt))
        return X.rnd(acc, dt)
    return f


def _sum_then_c(groups):
    """Group sums rounded, added to each other with rounding, C added last."""
    def f(P, C, X, dt):
        acc = None
        for g in groups:
            s = X.const(X.rnd(X.psum(P, list(g)), dt))
            acc = s if acc is None else X.const(X.rnd(X.add(acc, s), dt))
        return X.rnd(X.add(acc, X.const(C)), dt)
    return f


def _exact(P, C, X, dt):
    return X.rnd(X.add(X.psum(P, list(range(16))), X.const(C)), dt)


R = range
MODELS = {
    "M-exact": _exact,
    "M-seq": _groups_model([[k] for k in R(16)]),
    "M-seq-rev": _groups_model([[k] for k in R(15, -1, -1)]),
    "M-g8": _groups_model([R(0, 8), R(8, 16)]),
    "M-g8-rev": _groups_model([R(8, 16), R(0, 8)]),
    "M-g8-evenodd": _groups_model([R(0, 16, 2), R(1, 16, 2)]),
    "M-g4": _groups_model([R(0, 4), R(4, 8), R(8, 12), R(12, 16)]),
    "M-g2": _groups_model([R(i, i + 2) for i in R(0, 16, 2)]),
    "M-tree": _tree_model(),
    "M-sum16-then-C": _sum_then_c([R(0, 16)]),
    "M-g8-pre": _pre_round_groups([R(0, 8), R(8, 16)]),
    "M-g8-then-C": _sum_then_c([R(0, 8), R(8, 16)]),
    "M-g4-then-C": _sum_then_c([R(0, 4), R(4, 8), R(8, 12), R(12, 16)]),
}
