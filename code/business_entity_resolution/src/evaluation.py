"""Metrics: entity-level macro F0.5, candidate recall, threshold sweeps."""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_s1(all_s1: np.ndarray, valid_frac: float, seed: int):
    """Deterministic split of S1 ids (by entity, never by pair)."""
    ids = np.sort(np.asarray(all_s1, dtype=object))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_valid = int(len(ids) * valid_frac)
    return ids[perm[n_valid:]], ids[perm[:n_valid]]


def macro_fbeta(pred: pd.DataFrame, truth: pd.DataFrame, s1_ids, beta: float = 0.5) -> dict:
    """Entity-level macro F-beta exactly as the challenge defines it.

    pred / truth: long DataFrames with columns s1, tgt (one row per pair).
    s1_ids: every S1 in the evaluation set (singletons included).
    Entity with no truth and no prediction scores 1; no truth but some
    prediction scores 0; otherwise standard F-beta of that entity's sets.
    """
    s1_ids = pd.Index(pd.unique(np.asarray(s1_ids, dtype=object)))
    b2 = beta * beta
    p = pred[["s1", "tgt"]].drop_duplicates()
    t = truth[["s1", "tgt"]].drop_duplicates()
    tp = p.merge(t, on=["s1", "tgt"]).groupby("s1").size()
    n_pred = p.groupby("s1").size().reindex(s1_ids, fill_value=0)
    n_true = t.groupby("s1").size().reindex(s1_ids, fill_value=0)
    tp = tp.reindex(s1_ids, fill_value=0)
    prec = np.where(n_pred > 0, tp / np.maximum(n_pred, 1), 0.0)
    rec = np.where(n_true > 0, tp / np.maximum(n_true, 1), 0.0)
    denom = b2 * prec + rec
    f = np.where(denom > 0, (1 + b2) * prec * rec / np.where(denom > 0, denom, 1), 0.0)
    both_empty = (n_pred.to_numpy() == 0) & (n_true.to_numpy() == 0)
    f = np.where(both_empty, 1.0, f)
    return {
        "f05": float(f.mean()),
        "precision_micro": float(tp.sum() / max(n_pred.sum(), 1)),
        "recall_micro": float(tp.sum() / max(n_true.sum(), 1)),
        "n_s1": int(len(s1_ids)),
        "singleton_acc": float(both_empty.sum() / max((n_true.to_numpy() == 0).sum(), 1)),
        "per_s1": pd.Series(f, index=s1_ids),
    }


def candidate_stats(cands: pd.DataFrame, truth: pd.DataFrame, s1_ids) -> dict:
    """Recall ceiling and candidate-count distribution for a candidate set."""
    s1_ids = pd.Index(pd.unique(np.asarray(s1_ids, dtype=object)))
    counts = cands.groupby("s1").size().reindex(s1_ids, fill_value=0)
    t = truth[truth["s1"].isin(s1_ids)]
    hit = t.merge(cands[["s1", "tgt"]].drop_duplicates(), on=["s1", "tgt"], how="left",
                  indicator=True)["_merge"].eq("both")
    per_s1_full = hit.groupby(t["s1"].to_numpy()).all()
    return {
        "pair_recall": float(hit.mean()) if len(hit) else 1.0,
        "s1_full_recall": float(per_s1_full.mean()) if len(per_s1_full) else 1.0,
        "mean": float(counts.mean()), "median": float(counts.median()),
        "p90": float(counts.quantile(0.9)), "p95": float(counts.quantile(0.95)),
        "max": int(counts.max()), "zero_frac": float((counts == 0).mean()),
        "total_pairs": int(len(cands)),
    }
