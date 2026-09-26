"""Small end-to-end smoke test of featurize -> train -> decide on the 20k-S1
blocking experiment pool (cache/exp_blocking_res.parquet).

Usage: python scripts/smoke_test.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from candidate_generation import select_candidates  # noqa: E402
from evaluation import macro_fbeta  # noqa: E402
from inference import select_expected_f, select_threshold  # noqa: E402
from pipeline import _label, _stores, _truth_rows  # noqa: E402
from training import (featurize, feature_columns, fit_model, make_executor, predict,  # noqa: E402
                      target_side_features)


def main():
    t0 = time.time()
    s1, t = _stores(type("C", (), {"cache_dir": "cache"})(), "train")
    s1_ids = s1.entity_ids()
    t_ids = {s: t[s].entity_ids() for s in (2, 3)}
    from config import Config
    truth = _truth_rows(Config(), s1_ids, t_ids)
    r = pd.read_parquet("cache/exp_blocking_res.parquet")
    s1_map = pd.Series(np.arange(len(s1_ids)), index=s1_ids)
    r["s1_row"] = s1_map[r["s1"]].to_numpy().astype(np.int32)
    r["t_row"] = -1
    for s in (2, 3):
        m = r["src"] == s
        r.loc[m, "t_row"] = pd.Series(np.arange(len(t_ids[s])), index=t_ids[s])[r.loc[m, "tgt"]].to_numpy()
    r["src"] = r["src"].astype(np.int8)
    r["t_row"] = r["t_row"].astype(np.int32)
    c = select_candidates(r, 10, 3, 3).reset_index(drop=True)
    c = target_side_features(c[["s1_row", "src", "t_row", "s_name", "s_addr", "s_cross", "rank",
                                "rank_name", "rank_addr"]].copy())
    print(f"loaded {len(c):,} candidate pairs ({time.time() - t0:.0f}s)")
    ex = make_executor(2)
    t1 = time.time()
    X = featurize(c, s1, t, ex, chunk_s1=5000)
    ex.shutdown()
    print(f"featurized {len(X):,} pairs in {time.time() - t1:.0f}s "
          f"({len(X) / (time.time() - t1):.0f} pairs/s)")
    X["label"] = _label(X, truth)
    s1u = np.unique(X["s1_row"])
    rng = np.random.default_rng(0)
    va = rng.choice(s1u, len(s1u) // 3, replace=False)
    is_va = np.isin(X["s1_row"], va)
    cols = feature_columns(X)
    def ids(d):
        return pd.DataFrame({"s1": d["s1_row"].to_numpy(),
                             "tgt": d["src"].to_numpy().astype(np.int64) * 10**8 + d["t_row"].to_numpy()})

    from training import fit_two_stage
    b = fit_two_stage(X[~is_va].reset_index(drop=True), cols, "hgb", 42, {})
    Xv = X[is_va].copy()
    Xv["p1"] = predict({"model": b["model"], "columns": cols}, Xv)
    Xv["p"] = predict(b, Xv)
    for thr in (0.5, 0.7, 0.8):
        print("stage1", thr, round(macro_fbeta(ids(select_threshold(Xv.assign(p=Xv["p1"]), thr)),
                                              ids(X[is_va & (X["label"] == 1)]), va)["f05"], 4))
    tv = truth[np.isin(truth["s1_row"], va)]
    print("(stage1 lines use in-candidate truth only; stage2 lines below use full truth)")

    def ids(d):
        return pd.DataFrame({"s1": d["s1_row"].to_numpy(),
                             "tgt": d["src"].to_numpy().astype(np.int64) * 10**8 + d["t_row"].to_numpy()})

    print("pair recall ceiling:", X.loc[is_va, "label"].sum() / len(tv))
    for thr in (0.3, 0.5, 0.7, 0.8, 0.9):
        print(thr, {k: round(v, 4) for k, v in macro_fbeta(ids(select_threshold(Xv, thr)), ids(tv), va).items()
                    if k != "per_s1"})
    print("expf", {k: round(v, 4) for k, v in macro_fbeta(ids(select_expected_f(Xv)), ids(tv), va).items()
                   if k != "per_s1"})
    Xv.to_parquet("cache/smoke_valid.parquet", index=False)
    tv.to_parquet("cache/smoke_truth.parquet", index=False)
    pd.Series(cols).to_csv("cache/smoke_cols.csv", index=False)
    print("n features", len(cols))


if __name__ == "__main__":
    main()
