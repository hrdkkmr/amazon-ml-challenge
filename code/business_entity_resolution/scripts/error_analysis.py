"""Error analysis for scored validation pairs.

Usage: python scripts/error_analysis.py <scored.parquet> <truth.parquet> [thr] [n]
scored: s1_row, src, t_row, p (+ features); truth: s1_row, src, t_row.
Writes experiments/errors_fp.tsv / errors_fn.tsv and prints samples.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from inference import one_to_one  # noqa: E402
from records import Enricher, RecordStore  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
SHOW = ["p", "n_tset", "n_ratio", "a_ptset", "a_cov_b", "num_house_eq", "state_agree",
        "s_total", "rank", "c_quick_rank", "t_n_s1", "t_s_total_rank"]


def main():
    sc = pd.read_parquet(sys.argv[1])
    truth = pd.read_parquet(sys.argv[2])
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 25
    key = lambda d: (d["s1_row"].to_numpy().astype(np.int64) << 27) + d["t_row"].to_numpy().astype(np.int64) * 4 + d["src"].to_numpy()
    sc["label"] = np.isin(key(sc), key(truth)).astype(int)
    sel = one_to_one(sc)
    sel = sel[sel["p"] >= thr]
    fp = sel[sel["label"] == 0]
    fn = sc[(sc["label"] == 1) & ~np.isin(key(sc), key(sel))]
    s1_has_truth = np.isin(fp["s1_row"], truth["s1_row"])
    print(f"FP {len(fp)} (on singleton S1: {(~s1_has_truth).sum()}), "
          f"FN-in-candidates {len(fn)}, missing-from-candidates "
          f"{len(truth) - sc['label'].sum()}")
    enr = Enricher("cache")
    s1 = RecordStore("cache", "train", 1, enr)
    t = {s: RecordStore("cache", "train", s, enr) for s in (2, 3)}
    rows = []
    for kind, d in (("FP", fp), ("FN", fn)):
        d = d.sample(min(n, len(d)), random_state=0)
        a = s1.get(d["s1_row"].to_numpy())
        for i, (_, r) in enumerate(d.iterrows()):
            b = t[int(r["src"])].get(np.array([r["t_row"]]))
            true_ids = truth[truth["s1_row"] == r["s1_row"]]
            rows.append({"kind": kind, "s1_name": a["name"][i], "s1_addr": a["addr"][i],
                         "t_name": b["name"][0], "t_addr": b["addr"][0],
                         "n_true": len(true_ids), **{c: r[c] for c in SHOW if c in r}})
    out = pd.DataFrame(rows)
    ed = Path("experiments")
    out[out.kind == "FP"].to_csv(ed / "errors_fp.tsv", sep="\t", index=False)
    out[out.kind == "FN"].to_csv(ed / "errors_fn.tsv", sep="\t", index=False)
    for _, r in out.iterrows():
        print(f"[{r.kind}] p={r.p:.2f} ntrue={r.n_true} | {r.s1_name} | {r.s1_addr}\n"
              f"      -> {r.t_name} | {r.t_addr} | "
              + " ".join(f"{c}={r[c]:.2f}" for c in SHOW[1:] if c in r))


if __name__ == "__main__":
    main()
