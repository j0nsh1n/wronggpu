"""Weight side of the INT8 study, model-agnostic.

    python int8_study.py <model.wgcm> <map.json> [--label NAME] [--json OUT.json]

For every tensor that feeds a matrix multiply (map op fp8 / e4m3, or FP16 matrices): per int8 scheme the relative RMS
error and the share of nonzero weights that become 0; the exponent span per output row and per 16x16 block
(share <= 3, i.e. <= 4 binades: one int8 plane exact; <= 10, i.e. <= 11 binades: two planes exact); and the exact
digit-plane count of rr_step_i (S6). Prints and saves aggregate numbers only: no weight, no per-weight array.
"""
import argparse
import json
import sys

import numpy as np

import qlib as Q


def study(model_path, map_path):
    model = Q.read_wgcm(model_path)
    mats, fp8_model = Q.matmul_tensors(model, map_path)
    mbits = 4 if fp8_model else 11
    rows, agg = [], {k: {} for k in Q.SCHEMES}
    agg_span = {"row": [], "block": []}
    planes_tot = {"blocks": 0, "span_digits": 0, "bit_digits": 0, "d": [0, 0, 0, 0, 0]}
    for st, name, W in mats:
        r = {"tensor": f"{st}.{name}", "out": W.shape[0], "K": W.shape[1], "n": int(W.size)}
        for k, fn in Q.SCHEMES.items():
            e = Q.err_stats(W, fn(W))
            agg[k] = Q.merge_err(agg[k], e)
            r[k] = {"rel_rms": Q.rel_rms(e), "zeroed": e["nz_to_zero"] / max(1, e["nz"]), "exact": e["exact"] / e["n"]}
        sp_row = Q.spans(W, 1, fp8_model)
        B = Q.blocks(W)
        sp_blk = Q.spans(B, 2, fp8_model)
        agg_span["row"].append(sp_row.ravel())
        agg_span["block"].append(sp_blk.ravel())
        r["span_row"] = Q.span_hist(sp_row.ravel())
        r["span_block"] = Q.span_hist(sp_blk.ravel())
        d_span = Q.digits_from_span(sp_blk, mbits)
        d_bits = np.where(Q.bits_needed(B, 2) < 0, 0, np.ceil(np.maximum(Q.bits_needed(B, 2), 0) / 7.0)).astype(int)
        nb = int(np.count_nonzero(sp_blk >= 0))
        r["S6"] = {"planes_span_rule": float(d_span[sp_blk >= 0].mean()) if nb else 0.0,
                   "planes_bit_rule": float(d_bits[sp_blk >= 0].mean()) if nb else 0.0}
        planes_tot["blocks"] += nb
        planes_tot["span_digits"] += int(d_span[sp_blk >= 0].sum())
        planes_tot["bit_digits"] += int(d_bits[sp_blk >= 0].sum())
        for dd in d_span[sp_blk >= 0].ravel():
            planes_tot["d"][int(dd)] += 1
        rows.append(r)
    total = {k: {"rel_rms": Q.rel_rms(v), "zeroed": v["nz_to_zero"] / max(1, v["nz"]), "exact": v["exact"] / v["n"]}
             for k, v in agg.items()}
    total["span_row"] = Q.span_hist(np.concatenate(agg_span["row"]))
    total["span_block"] = Q.span_hist(np.concatenate(agg_span["block"]))
    nb = max(1, planes_tot["blocks"])
    total["S6"] = {"planes_span_rule": planes_tot["span_digits"] / nb, "planes_bit_rule": planes_tot["bit_digits"] / nb,
                   "blocks_by_digits": {str(i): planes_tot["d"][i] for i in (1, 2, 3)}}
    total["tensors"] = len(rows)
    total["elements"] = int(sum(r["n"] for r in rows))
    total["format"] = "e4m3" if fp8_model else "fp16"
    return rows, total


def md(rows, total, label):
    out = [f"### {label}: weights ({total['tensors']} matrices, {total['elements']} weights, {total['format']})", "",
           "| Tensor | out x K | S1 rms | S2 rms | S3 rms | S4 rms | S5 rms | S1/S2/S4 zeroed | rows span<=3 / <=10 |"
           " blocks span<=3 / <=10 | S6 planes/block |", "|" + "---|" * 11]

    def line(name, shape, r):
        z = "/".join(f"{100 * r[k]['zeroed']:.1f}" for k in ("S1", "S2", "S4"))
        return (f"| {name} | {shape} | " + " | ".join(f"{r[k]['rel_rms']:.2e}" for k in Q.SCHEMES) + f" | {z} % | "
                f"{100 * r['span_row']['le3']:.0f} / {100 * r['span_row']['le10']:.0f} % | "
                f"{100 * r['span_block']['le3']:.0f} / {100 * r['span_block']['le10']:.0f} % | "
                f"{r['S6']['planes_span_rule']:.2f} |")
    for r in rows:
        out.append(line(r["tensor"], f"{r['out']}x{r['K']}", r))
    out.append(line("**all**", "", total))
    out.append("")
    out.append(f"Block span histogram (span 0..14, 15+): {total['span_block']['hist']}; row span histogram: "
               f"{total['span_row']['hist']}. S6 blocks by digit count 1/2/3: {total['S6']['blocks_by_digits']}; "
               f"mean digits per block: span rule {total['S6']['planes_span_rule']:.3f}, lowest-bit rule "
               f"{total['S6']['planes_bit_rule']:.3f}.")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("map")
    ap.add_argument("--label", default="model")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rows, total = study(a.model, a.map)
    print(md(rows, total, a.label))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"label": a.label, "rows": rows, "total": total}, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
