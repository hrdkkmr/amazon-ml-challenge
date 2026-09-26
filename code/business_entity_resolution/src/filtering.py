"""Blocking stage 2: a learned candidate filter using only blocking signals.

The candidate *pool* (union of top-15 total / top-5 name / top-5 address per
S1 and source) is pruned by a small gradient-boosted model that sees only
cheap, string-free signals of each pool pair:

  * the three IDF blocking scores and their sum, and the three ranks;
  * per-(S1, source) gaps to the best total / name / address score;
  * target-side competition: how many S1 retrieved this target, and the
    margin / rank of this S1's total score among them.

Pairs whose filter probability is below a threshold (chosen on validation
for a recall/size trade-off) are dropped.  The survivors are exactly the
candidates written to candidate_pairs.tsv and scored by the matcher.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import group_top2

FILTER_FEATS = ["s_name", "s_addr", "s_cross", "s_tot", "rank", "rank_name", "rank_addr",
                "g_tot", "g_name", "g_addr", "m_tot", "n_pool", "t_n", "t_margin", "t_rank"]


def filter_features(pool: pd.DataFrame) -> pd.DataFrame:
    """Add FILTER_FEATS columns to a pool DataFrame (in place, float32)."""
    tot = (pool["s_name"] + pool["s_addr"] + pool["s_cross"]).to_numpy().astype(np.float64)
    pool["s_tot"] = tot.astype(np.float32)
    s1 = pool["s1_row"].to_numpy().astype(np.int64)
    src = pool["src"].to_numpy()
    g = pd.factorize(s1 * 4 + src)[0]
    pool["n_pool"] = np.bincount(g)[g].astype(np.float32)
    for col, name in (("s_tot", "tot"), ("s_name", "name"), ("s_addr", "addr")):
        v = pool[col].to_numpy().astype(np.float64)
        m1, m2, rk = group_top2(g, v)
        pool[f"g_{name}"] = (m1 - v).astype(np.float32)
        if name == "tot":
            pool["m_tot"] = np.where(rk == 0, v - m2, v - m1).astype(np.float32)
    gt = pd.factorize(pool["t_row"].to_numpy().astype(np.int64) * 4 + src)[0]
    pool["t_n"] = np.bincount(gt)[gt].astype(np.float32)
    m1, m2, rk = group_top2(gt, tot)
    pool["t_margin"] = np.where(rk == 0, tot - m2, tot - m1).astype(np.float32)
    pool["t_rank"] = rk.astype(np.float32)
    return pool
