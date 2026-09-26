"""Validation breakdown for the final model: per-country F0.5, error types.

Uses cache/scores_train.parquet (all training-split candidates scored) and
models/decision.json; evaluates on the held-out validation S1 only.
Writes experiments/final_report.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import Config  # noqa: E402
from data_io import load_norm  # noqa: E402
from evaluation import macro_fbeta  # noqa: E402
from inference import select_expected_f, select_threshold  # noqa: E402
from pipeline import _pair_key, _split_rows, _stores, _truth_rows  # noqa: E402


def ids(d):
    return pd.DataFrame({"s1": d["s1_row"].to_numpy(),
                         "tgt": d["src"].to_numpy().astype(np.int64) * 10**8 + d["t_row"].to_numpy()})


def main():
    cfg = Config.load("configs/default.json")
    s1, t = _stores(cfg, "train")
    s1_ids = s1.entity_ids()
    truth = _truth_rows(cfg, s1_ids, {s: t[s].entity_ids() for s in (2, 3)})
    del s1, t
    _, va = _split_rows(cfg, s1_ids)
    dec = json.loads(Path(cfg.model_dir, "decision.json").read_text())
    sc = pd.read_parquet(cfg.path("scores_train.parquet"))
    if dec["mode"] == "thr":
        sel = select_threshold(sc, dec["thr"], dec["o2o"])
    else:
        sel = select_expected_f(sc, dec["o2o"], miss_mass=dec["miss_mass"], floor=dec["floor"])
    sel = sel[np.isin(sel["s1_row"].to_numpy(), va)]
    tv = truth[np.isin(truth["s1_row"].to_numpy(), va)]
    country = load_norm(cfg.cache_dir, "train", 1, ["country"])["country"].astype(object).to_numpy()
    rep = {"decision": dec}
    r = macro_fbeta(ids(sel), ids(tv), va)
    rep["all"] = {k: v for k, v in r.items() if k != "per_s1"}
    per = r["per_s1"]
    n_true = tv.groupby("s1_row").size().reindex(va, fill_value=0).to_numpy()
    rep["f05_by_n_true"] = {int(k): round(float(v), 4) for k, v in
                            pd.Series(per.to_numpy()).groupby(np.minimum(n_true, 8)).mean().items()}
    for c in np.unique(country[va]):
        rows = va[country[va] == c]
        rr = macro_fbeta(ids(sel[np.isin(sel["s1_row"].to_numpy(), rows)]),
                         ids(tv[np.isin(tv["s1_row"].to_numpy(), rows)]), rows)
        rep[f"country={c}"] = {k: v for k, v in rr.items() if k != "per_s1"}
    # error decomposition
    cand = sc[np.isin(sc["s1_row"].to_numpy(), va)]
    in_cand = np.isin(_pair_key(tv), _pair_key(cand))
    matched = np.isin(_pair_key(tv), _pair_key(sel))
    fp = sel[~np.isin(_pair_key(sel), _pair_key(tv))]
    rep["errors"] = {
        "true_pairs": int(len(tv)),
        "fn_not_in_candidates": int((~in_cand).sum()),
        "fn_in_candidates_rejected": int((in_cand & ~matched).sum()),
        "fp_total": int(len(fp)),
        "fp_on_singleton_s1": int((~np.isin(fp["s1_row"].to_numpy(), tv["s1_row"].to_numpy())).sum()),
        "s1_with_perfect_f05": float((per.to_numpy() == 1.0).mean()),
    }
    for k, v in rep.items():
        print(k, v)
    Path(cfg.experiments_dir, "final_report.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
