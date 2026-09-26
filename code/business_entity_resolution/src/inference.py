"""Decision rules (scores -> matches) and submission file writers.

Decision pipeline, applied identically on validation and test:

1. one-to-one on the target side: in the training truth every S2/S3 record
   belongs to at most one S1, so a target is only kept for the S1 that gives
   it the highest match probability;
2. per-S1 selection: keep candidates with probability >= ``thr`` (zero,
   one or many per S1 -- singletons get an empty list), or, with
   ``mode="expf"``, the prefix of candidates (sorted by probability) that
   maximizes the *expected* F0.5 of that S1.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


def one_to_one(scored: pd.DataFrame) -> pd.DataFrame:
    """Keep, for every (src, t_row), only the pair with the highest p."""
    s = scored.sort_values("p", ascending=False, kind="stable")
    return s.drop_duplicates(["src", "t_row"], keep="first")


def select_threshold(scored: pd.DataFrame, thr: float, o2o: bool = True) -> pd.DataFrame:
    s = one_to_one(scored) if o2o else scored
    return s[s["p"] >= thr]


def select_expected_f(scored: pd.DataFrame, o2o: bool = True, beta: float = 0.5,
                      miss_mass: float = 0.0, floor: float = 0.05) -> pd.DataFrame:
    """Per S1, choose the top-j candidates maximizing expected F-beta.

    With F = (1+b2) TP / (b2 * N_true + N_pred), approximate
    E[F(top-j)] ~ (1+b2) * sum_{i<=j} p_i / (b2 * (sum_all p + miss_mass) + j),
    and E[F(empty)] = P(no true match) ~ prod(1 - p_i).
    """
    s = one_to_one(scored) if o2o else scored
    s = s[s["p"] >= floor].sort_values(["s1_row", "p"], ascending=[True, False], kind="stable")
    if len(s) == 0:
        return s
    b2 = beta * beta
    s1 = s["s1_row"].to_numpy()
    p = s["p"].to_numpy().astype(np.float64)
    first = np.r_[0, np.flatnonzero(s1[1:] != s1[:-1]) + 1]
    size = np.diff(np.r_[first, len(s1)])
    grp = np.repeat(np.arange(len(first)), size)
    j = np.arange(len(s1)) - first[grp] + 1
    csum = np.cumsum(p)
    base = np.r_[0.0, csum[first[1:] - 1]] if len(first) > 1 else np.zeros(1)
    cum_in_grp = csum - base[grp]
    tot = np.add.reduceat(p, first)[grp]
    ef = (1 + b2) * cum_in_grp / (b2 * (tot + miss_mass) + j)
    log1m = np.log1p(-np.minimum(p, 1 - 1e-9))
    p_empty = np.exp(np.add.reduceat(log1m, first))
    # best j per group vs empty
    best_ef = np.maximum.reduceat(ef, first)
    arg = ef == best_ef[grp]
    # first j achieving the max
    cut_j = np.full(len(first), 0)
    idx = np.flatnonzero(arg)
    cut_j_vals = pd.Series(j[idx]).groupby(grp[idx]).min()
    cut_j[cut_j_vals.index.to_numpy()] = cut_j_vals.to_numpy()
    cut_j = np.where(best_ef > p_empty, cut_j, 0)
    return s[j <= cut_j[grp]]


def to_id_lists(pairs: pd.DataFrame, s1_ids_all: np.ndarray, s1_ids: np.ndarray,
                t_ids: dict) -> pd.Series:
    """Map (s1_row, src, t_row) pairs to one comma-joined id string per S1."""
    tg = np.empty(len(pairs), dtype=object)
    src = pairs["src"].to_numpy()
    for s in (2, 3):
        m = src == s
        tg[m] = t_ids[s][pairs["t_row"].to_numpy()[m]]
    d = pd.DataFrame({"s1": s1_ids_all[pairs["s1_row"].to_numpy()], "tgt": tg})
    d = d.drop_duplicates()
    lists = d.groupby("s1", sort=False)["tgt"].agg(lambda x: ",".join(sorted(x)))
    return lists.reindex(pd.Index(s1_ids)).fillna("")


def write_id_file(path: Path, s1_ids: np.ndarray, lists: pd.Series, col: str) -> None:
    """Write the official 2-column TSV (header + one row per S1, no quoting)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n",
                       escapechar=None)
        w.writerow(["source1_entity_id", col])
        for s1, ids in zip(s1_ids, lists.to_numpy()):
            f.write(f"{s1}\t{ids}\n")
