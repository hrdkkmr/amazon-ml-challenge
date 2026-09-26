"""Fast model-variant comparison on cached features.

Trains on cache/train_feats.parquet (200k training-split S1) and evaluates
entity-level macro F0.5 on cache/valid_feats.parquet (60k validation S1, all
their candidates) against the FULL truth of those S1 (so blocking misses count
as false negatives).  Results are appended to experiments/model_runs.jsonl.

Usage: python scripts/exp_model.py [variant ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import Config  # noqa: E402
from evaluation import macro_fbeta  # noqa: E402
from inference import select_expected_f, select_threshold  # noqa: E402
from pipeline import _stores, _truth_rows  # noqa: E402
from training import feature_columns, fit_model, fit_two_stage, predict  # noqa: E402

VARIANTS = {
    "hgb_base": ("hgb", {}),
    "hgb_big": ("hgb", {"max_leaf_nodes": 127, "max_iter": 800, "learning_rate": 0.06,
                        "min_samples_leaf": 60}),
    "hgb_big2": ("hgb", {"max_leaf_nodes": 255, "max_iter": 1200, "learning_rate": 0.05,
                         "min_samples_leaf": 100}),
    "logreg": ("logreg", {}),
    "two_stage": ("two_stage", {}),
    "two_stage_big": ("two_stage", {"max_leaf_nodes": 255, "max_iter": 1200, "learning_rate": 0.05,
                                    "min_samples_leaf": 100}),
}


def ids(d):
    return pd.DataFrame({"s1": d["s1_row"].to_numpy(),
                         "tgt": d["src"].to_numpy().astype(np.int64) * 10**8 + d["t_row"].to_numpy()})


def main():
    cfg = Config.load("configs/default.json")
    names = sys.argv[1:] or ["hgb_base", "hgb_big"]
    tr = pd.read_parquet(cfg.path("train_feats.parquet"))
    va = pd.read_parquet(cfg.path("valid_feats.parquet"))
    cols = feature_columns(tr)
    s1, t = _stores(cfg, "train")
    truth = _truth_rows(cfg, s1.entity_ids(), {s: t[s].entity_ids() for s in (2, 3)})
    del s1, t
    va_s1 = np.unique(va["s1_row"].to_numpy())
    tv = ids(truth[np.isin(truth["s1_row"].to_numpy(), va_s1)])
    for name in names:
        kind, params = VARIANTS[name]
        t0 = time.time()
        if kind == "two_stage":
            b = fit_two_stage(tr, cols, "hgb", cfg.seed, params)
        else:
            b = {"model": fit_model(tr[cols].to_numpy(np.float32), tr["label"].to_numpy(),
                                    kind, cfg.seed, **params), "columns": cols}
        fit_s = time.time() - t0
        v = va[["s1_row", "src", "t_row"] + cols].copy()
        v["p"] = predict(b, v)
        res = {}
        for thr in np.round(np.arange(0.3, 0.96, 0.05), 2):
            r = macro_fbeta(ids(select_threshold(v, thr)), tv, va_s1)
            res[f"thr={thr:.2f}"] = round(r["f05"], 5)
        r = macro_fbeta(ids(select_expected_f(v)), tv, va_s1)
        res["expf"] = round(r["f05"], 5)
        best = max(res, key=res.get)
        n_iter = getattr(b["model"], "n_iter_", None)
        print(f"{name}: best {best} F0.5={res[best]} (fit {fit_s:.0f}s, iters {n_iter}) {res}",
              flush=True)
        with open(Path(cfg.experiments_dir) / "model_runs.jsonl", "a") as f:
            f.write(json.dumps({"variant": name, "params": params, "best": best,
                                "f05": res[best], "all": res, "fit_s": round(fit_s),
                                "n_train_rows": len(tr), "n_valid_s1": len(va_s1),
                                "n_iter": n_iter}) + "\n")
        if kind != "two_stage" and hasattr(b["model"], "n_iter_"):
            pd.to_pickle(b, cfg.path(f"exp_model_{name}.pkl"))


if __name__ == "__main__":
    main()
