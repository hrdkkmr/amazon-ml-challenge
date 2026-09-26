"""Training-data construction, featurization and model fitting.

Flow for a split:
  pool (blocking output)  --select_candidates-->  candidates
  candidates + target-side competition features   (cache/cands_<split>.parquet)
  candidates of a set of S1 --featurize--> pairwise + S1-side context features
"""
from __future__ import annotations

import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_generation import iter_pool, select_candidates
from features import add_context_features, group_top2, pair_features

ID_COLS = ["s1_row", "src", "t_row"]


def cands_path(cache_dir: str, split: str) -> Path:
    return Path(cache_dir) / f"cands_{split}.parquet"


def target_side_features(c: pd.DataFrame) -> pd.DataFrame:
    """Competition among S1 entities for the same target (needs the full set)."""
    tot = (c["s_name"] + c["s_addr"] + c["s_cross"]).to_numpy().astype(np.float64)
    g = pd.factorize(c["t_row"].to_numpy().astype(np.int64) * 4 + c["src"].to_numpy())[0]
    c["t_n_s1"] = np.bincount(g)[g].astype(np.float32)
    m1, m2, rk = group_top2(g, tot)
    c["t_s_total_margin"] = np.where(rk == 0, tot - m2, tot - m1).astype(np.float32)
    c["t_s_total_rank"] = rk.astype(np.float32)
    return c


def build_candidates(cfg, split: str) -> pd.DataFrame:
    """Final candidate set = pool -> (learned blocking filter | fixed rank cut),
    plus target-side competition features.  This set is written unchanged to
    candidate_pairs.tsv and is exactly what the matcher scores."""
    from filtering import FILTER_FEATS, filter_features
    out = cands_path(cfg.cache_dir, split)
    use_filter = cfg.extra.get("use_filter", True)
    if use_filter:
        f = load_model(Path(cfg.model_dir) / "filter.pkl")
        thr = float(cfg.extra.get("filter_thr", 0.002))
        desc = f"filter q>={thr}"
    else:
        k = tuple(cfg.extra.get("cand_k", (10, 3, 3)))
        desc = f"cut {k}"
    parts, n_pool = [], 0
    for name, pool in iter_pool(cfg, split):
        n_pool += len(pool)
        if use_filter:
            pool = filter_features(pool)
            q = f["model"].predict_proba(pool[FILTER_FEATS].to_numpy(np.float32))[:, 1]
            keep = q >= thr
            c = pool[keep].reset_index(drop=True)
            c["q"] = q[keep].astype(np.float32)
        else:
            c = select_candidates(pool, *k).reset_index(drop=True)
        parts.append(c)
        print(f"    {name}: {len(pool):,} -> {len(c):,}", flush=True)
        del pool
    c = pd.concat(parts, ignore_index=True)
    c = target_side_features(c)
    c.to_parquet(out, index=False)
    print(f"  {split}: pool {n_pool:,} -> {len(c):,} candidate pairs ({desc}) -> {out.name}")
    return c


def featurize(cands: pd.DataFrame, s1_store, t_stores: dict, executor=None,
              chunk_s1: int = 40_000, sink=None) -> pd.DataFrame | None:
    """Compute features for ``cands`` in chunks of whole S1 entities (so the
    S1-side context features see every candidate of each S1).

    If ``sink`` is given it is called with each finished chunk DataFrame and
    nothing is accumulated (streaming mode); otherwise the concatenation is
    returned.
    """
    cands = cands.sort_values("s1_row", kind="stable").reset_index(drop=True)
    v = cands["s1_row"].to_numpy()
    s1_sorted = np.unique(v)
    out = []
    t0 = time.time()
    for i in range(0, len(s1_sorted), chunk_s1):
        lo, hi = s1_sorted[i], s1_sorted[min(i + chunk_s1, len(s1_sorted)) - 1]
        a, b = np.searchsorted(v, lo, "left"), np.searchsorted(v, hi, "right")
        ch = cands.iloc[a:b].reset_index(drop=True)
        parts = []
        for src in (2, 3):
            cs = ch[ch["src"] == src]
            if len(cs) == 0:
                continue
            A = s1_store.get(cs["s1_row"].to_numpy())
            B = t_stores[src].get(cs["t_row"].to_numpy())
            f = pair_features(A, B, executor)
            # name rarity: ambiguous (common) names need address evidence
            tf, sf = t_stores[src].core_counts(), s1_store.core_counts()
            f["t_core_freq"] = np.array([tf.get(x, 0) for x in B["core"]], dtype=np.float32)
            f["s1core_t_freq"] = np.array([tf.get(x, 0) for x in A["core"]], dtype=np.float32)
            f["s1_core_freq"] = np.array([sf.get(x, 0) for x in A["core"]], dtype=np.float32)
            f.index = cs.index
            parts.append(pd.concat([cs, f], axis=1))
        df = pd.concat(parts).sort_index()
        df = add_context_features(df)
        if sink is not None:
            sink(df)
        else:
            out.append(df)
        done = min(i + chunk_s1, len(s1_sorted))
        print(f"    featurized {done:,}/{len(s1_sorted):,} S1 "
              f"({time.time() - t0:.0f}s)", flush=True)
    if sink is None:
        return pd.concat(out, ignore_index=True) if out else None
    return None


NON_FEATURES = set(ID_COLS) | {"label", "s1", "tgt", "fold", "p"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def fit_model(X: pd.DataFrame, y: np.ndarray, kind: str = "hgb", seed: int = 42, **kw):
    t0 = time.time()
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=1.0))
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        params = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                      min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
                      validation_fraction=0.1, n_iter_no_change=30, random_state=seed)
        params.update(kw)
        m = HistGradientBoostingClassifier(**params)
    m.fit(X if isinstance(X, np.ndarray) else X.to_numpy(np.float32), y)
    print(f"  fitted {kind} on {len(y):,} rows x {X.shape[1]} features "
          f"({time.time() - t0:.0f}s)", flush=True)
    return m


P_CONTEXT = ["p1", "p1_rank", "p1_gap", "p1_margin", "p1_rank_src", "p1_gap_src",
             "p1_sum", "p1_n50", "p1_n80", "p1_other_max"]


def p_context(df: pd.DataFrame, p1: np.ndarray) -> pd.DataFrame:
    """Stage-2 context from stage-1 probabilities of all candidates of an S1."""
    s1 = df["s1_row"].to_numpy().astype(np.int64)
    g = pd.factorize(s1)[0]
    gs = pd.factorize(s1 * 4 + df["src"].to_numpy())[0]
    v = p1.astype(np.float64)
    out = pd.DataFrame(index=df.index)
    out["p1"] = p1
    m1, m2, rk = group_top2(g, v)
    out["p1_rank"] = rk
    out["p1_gap"] = m1 - v
    out["p1_margin"] = np.where(rk == 0, v - m2, v - m1)
    out["p1_other_max"] = np.where(rk == 0, m2, m1)
    m1s, _, rks = group_top2(gs, v)
    out["p1_rank_src"] = rks
    out["p1_gap_src"] = m1s - v
    out["p1_sum"] = np.bincount(g, weights=v)[g]
    out["p1_n50"] = np.bincount(g, weights=(v >= 0.5))[g]
    out["p1_n80"] = np.bincount(g, weights=(v >= 0.8))[g]
    return out.astype(np.float32)


def predict(model_bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Stage-1 probability, refined by the stage-2 model when present."""
    X = df[model_bundle["columns"]].to_numpy(np.float32)
    p1 = model_bundle["model"].predict_proba(X)[:, 1].astype(np.float32)
    if model_bundle.get("stage2") is None:
        return p1
    ctx = p_context(df, p1)
    X2 = np.hstack([X, ctx[P_CONTEXT].to_numpy(np.float32)])
    return model_bundle["stage2"].predict_proba(X2)[:, 1].astype(np.float32)


def fit_two_stage(X: pd.DataFrame, cols: list[str], kind: str, seed: int, params: dict,
                  n_folds: int = 2) -> dict:
    """Stage 1 cross-fitted by S1 (out-of-fold p1), stage 2 on features + p1 context."""
    y = X["label"].to_numpy()
    s1 = X["s1_row"].to_numpy()
    u = np.unique(s1)
    rng = np.random.default_rng(seed)
    fold_of = pd.Series(rng.integers(0, n_folds, len(u)), index=u)
    fold = fold_of.reindex(s1).to_numpy()
    oof = np.zeros(len(X), dtype=np.float32)
    for k in range(n_folds):
        tr = fold != k
        m = fit_model(X.loc[tr, cols], y[tr], kind, seed, **params)
        oof[~tr] = m.predict_proba(X.loc[~tr, cols].to_numpy(np.float32))[:, 1]
    m1 = fit_model(X[cols], y, kind, seed, **params)
    ctx = p_context(X, oof)
    X2 = np.hstack([X[cols].to_numpy(np.float32), ctx[P_CONTEXT].to_numpy(np.float32)])
    from sklearn.ensemble import HistGradientBoostingClassifier
    hp = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=80,
              l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
              n_iter_no_change=30, random_state=seed)
    m2 = HistGradientBoostingClassifier(**hp).fit(X2, y)
    return {"model": m1, "stage2": m2, "columns": cols, "oof_p1": oof}


def save_model(model, cols: list[str], path: Path, meta: dict | None = None) -> None:
    with open(path, "wb") as f:
        pickle.dump({"model": model, "columns": cols, "meta": meta or {}}, f)


def load_model(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def make_executor(n_workers: int):
    return ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
