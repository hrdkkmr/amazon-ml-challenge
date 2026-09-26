"""Command-line entry point.  Stages (run in this order; each one caches its
output and is skipped/restartable):

    preprocess    normalize the six raw TSVs into Parquet caches
    translit      learn Indic->Latin token dictionary + native state names
    pool_train    blocking on the training split -> candidate pool (per partition)
    filter_train  fit the learned blocking filter (string-free signals)
    cands_train   filtered candidates + target-side competition features
    train         featurize a sample of training-split S1, fit the matcher
    score_train   stream-featurize + score every training candidate
    evaluate      validation S1: candidate recall, F0.5 sweep, pick decision rule
    pool_test     blocking on the test split
    cands_test    filtered test candidates (== candidate_pairs.tsv)
    predict_test  score test candidates, decide, write both submission files
    validate      run the official validator + extra consistency checks

Usage:  python src/pipeline.py <stage|all> [--config configs/default.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config  # noqa: E402


# ----------------------------------------------------------------- helpers
def _stores(cfg: Config, split: str):
    from records import Enricher, RecordStore
    enr = Enricher(cfg.cache_dir)
    s1 = RecordStore(cfg.cache_dir, split, 1, enr)
    t = {s: RecordStore(cfg.cache_dir, split, s, enr) for s in (2, 3)}
    return s1, t


def _truth_rows(cfg: Config, s1_ids: np.ndarray, t_ids: dict) -> pd.DataFrame:
    """Training truth as (s1_row, src, t_row) + list of all S1 ids."""
    from data_io import load_ground_truth
    gt = load_ground_truth(cfg.data_dir)
    s1_map = pd.Series(np.arange(len(s1_ids)), index=s1_ids)
    out = []
    for s in (2, 3):
        g = gt[gt["tgt"].str.startswith(f"S{s}-")]
        t_map = pd.Series(np.arange(len(t_ids[s])), index=t_ids[s])
        out.append(pd.DataFrame({"s1_row": s1_map.reindex(g["s1"]).to_numpy(),
                                 "src": np.int8(s),
                                 "t_row": t_map.reindex(g["tgt"]).to_numpy()}))
    tr = pd.concat(out, ignore_index=True)
    assert tr.notna().all().all(), "truth ids missing from caches"
    tr = tr.astype({"s1_row": np.int32, "t_row": np.int32})
    return tr


def _pair_key(d: pd.DataFrame) -> np.ndarray:
    """Unique int64 per (s1_row, src, t_row)."""
    return ((d["s1_row"].to_numpy().astype(np.int64) << 27)
            + d["t_row"].to_numpy().astype(np.int64) * 4 + d["src"].to_numpy())


def _label(c: pd.DataFrame, truth: pd.DataFrame) -> np.ndarray:
    return np.isin(_pair_key(c), _pair_key(truth)).astype(np.int8)


def _split_rows(cfg: Config, s1_ids: np.ndarray):
    from evaluation import split_s1
    tr_ids, va_ids = split_s1(s1_ids, cfg.valid_frac, cfg.seed)
    m = pd.Series(np.arange(len(s1_ids)), index=s1_ids)
    return m[tr_ids].to_numpy(), m[va_ids].to_numpy()


def _log_experiment(cfg: Config, rec: dict) -> None:
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
    with open(Path(cfg.experiments_dir) / "experiments.jsonl", "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def _filter_sample(cfg: Config, tr_rows: np.ndarray):
    """(sample A for the candidate filter, remaining training-split rows)."""
    rng = np.random.default_rng(cfg.seed + 1)
    n = int(cfg.extra.get("n_filter_s1", 200_000))
    a = rng.choice(tr_rows, min(n, len(tr_rows) // 2), replace=False)
    return a, np.setdiff1d(tr_rows, a)


# ------------------------------------------------------------------ stages
def stage_preprocess(cfg: Config) -> None:
    from data_io import norm_path, preprocess_file, raw_path
    for split in ("train", "test"):
        for s in (1, 2, 3):
            dst = norm_path(cfg.cache_dir, split, s)
            if dst.exists():
                print(f"  cached: {dst.name}")
                continue
            preprocess_file(raw_path(cfg.data_dir, split, s), dst, cfg.read_chunk, cfg.n_workers)


def stage_translit(cfg: Config) -> None:
    """Learn indic->latin token dictionary and native-script state names from
    training truth pairs (only the provided training data is used)."""
    import pickle
    from data_io import load_ground_truth, load_norm
    from records import learn_state_map
    from translit import TranslitDict
    gt = load_ground_truth(cfg.data_dir)
    s1full = load_norm(cfg.cache_dir, "train", 1, ["entity_id", "name_norm", "script", "state"])
    s1 = s1full[s1full["script"] == "L"].set_index("entity_id")["name_norm"]
    s1state = s1full.set_index("entity_id")["state"]
    pairs, st_s1, st_tgt = [], [], []
    for src in (2, 3):
        t = load_norm(cfg.cache_dir, "train", src, ["entity_id", "name_norm", "addr_norm", "script"])
        print(f"  S{src} script mix:", t["script"].value_counts().to_dict())
        t = t[t["script"] != "L"]
        m = gt.merge(t, left_on="tgt", right_on="entity_id")
        m["lat"] = s1.reindex(m["s1"].to_numpy()).to_numpy()
        st_s1.append(s1state.reindex(m["s1"].to_numpy()).fillna("").astype(object))
        st_tgt.append(m["addr_norm"].astype(object))
        m = m.dropna(subset=["lat"])
        pairs += [(a.split(), b.split()) for a, b in zip(m["lat"].tolist(), m["name_norm"].tolist())]
    td = TranslitDict.learn(pairs)
    td.save(cfg.path("translit_dict.pkl"))
    print(f"  learned {len(td.mapping):,} token mappings from {len(pairs):,} pairs")
    sm = learn_state_map(None, pd.concat(st_s1), pd.concat(st_tgt))
    cfg.path("state_map.pkl").write_bytes(pickle.dumps(sm))
    print(f"  learned {len(sm)} native-script state names")


def stage_pool_train(cfg: Config) -> None:
    from candidate_generation import generate_pool
    generate_pool(cfg, "train", tuple(cfg.extra.get("pool_k", (15, 5, 5))))


def stage_pool_test(cfg: Config) -> None:
    from candidate_generation import generate_pool
    generate_pool(cfg, "test", tuple(cfg.extra.get("pool_k", (15, 5, 5))))


def stage_filter_train(cfg: Config) -> None:
    """Fit the learned blocking filter on pool pairs of sample-A training S1 and
    report recall / candidate count on validation S1."""
    from candidate_generation import iter_pool, select_candidates
    from filtering import FILTER_FEATS, filter_features
    from training import fit_model, save_model
    s1, t = _stores(cfg, "train")
    s1_ids = s1.entity_ids()
    truth = _truth_rows(cfg, s1_ids, {s: t[s].entity_ids() for s in (2, 3)})
    del s1, t
    tr_rows, va_rows = _split_rows(cfg, s1_ids)
    a, _ = _filter_sample(cfg, tr_rows)
    rng = np.random.default_rng(cfg.seed + 2)
    va_rows = rng.choice(va_rows, min(150_000, len(va_rows)), replace=False)
    trs, vas = [], []
    for _, pool in iter_pool(cfg, "train"):
        pool = filter_features(pool)
        trs.append(pool[np.isin(pool["s1_row"].to_numpy(), a)])
        vas.append(pool[np.isin(pool["s1_row"].to_numpy(), va_rows)])
        del pool
    tr, va = pd.concat(trs, ignore_index=True), pd.concat(vas, ignore_index=True)
    del trs, vas
    m = fit_model(tr[FILTER_FEATS].to_numpy(np.float32), _label(tr, truth), "hgb", cfg.seed,
                  max_leaf_nodes=63, max_iter=300)
    save_model(m, FILTER_FEATS, Path(cfg.model_dir) / "filter.pkl", {"n_s1": len(a)})
    va["label"] = _label(va, truth)
    va["q"] = m.predict_proba(va[FILTER_FEATS].to_numpy(np.float32))[:, 1]
    n_true = int(np.isin(truth["s1_row"].to_numpy(), va_rows).sum())
    rep = {"pool": [float(va["label"].sum() / n_true), len(va) / len(va_rows)]}
    b = select_candidates(va, 10, 3, 3)
    rep["cut(10,3,3)"] = [float(b["label"].sum() / n_true), len(b) / len(va_rows)]
    for thr in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02):
        k = va[va["q"] >= thr]
        rep[f"q>={thr}"] = [float(k["label"].sum() / n_true), len(k) / len(va_rows)]
    for k_, (r, n) in rep.items():
        print(f"  {k_:12s} pair recall {r:.4f}  candidates/S1 {n:.2f}")
    _log_experiment(cfg, {"stage": "filter_train", "validation_recall_vs_size": rep})


def stage_cands_train(cfg: Config) -> None:
    from training import build_candidates
    build_candidates(cfg, "train")


def stage_cands_test(cfg: Config) -> None:
    from training import build_candidates
    build_candidates(cfg, "test")


def stage_train(cfg: Config) -> None:
    """Featurize a sample of *training-split* S1 entities and fit the matcher."""
    from training import (cands_path, feature_columns, featurize, fit_model, make_executor,
                          save_model)
    feats_path = cfg.path("train_feats.parquet")
    s1, t = _stores(cfg, "train")
    s1_ids = s1.entity_ids()
    t_ids = {s: t[s].entity_ids() for s in (2, 3)}
    truth = _truth_rows(cfg, s1_ids, t_ids)
    tr_rows, _ = _split_rows(cfg, s1_ids)
    if not feats_path.exists():
        c = pd.read_parquet(cands_path(cfg.cache_dir, "train"))
        _, rest = _filter_sample(cfg, tr_rows)     # disjoint from the filter's sample
        rng = np.random.default_rng(cfg.seed)
        n = int(cfg.extra.get("n_train_s1", 200_000))
        sample = rng.choice(rest, min(n, len(rest)), replace=False)
        c = c[np.isin(c["s1_row"].to_numpy(), sample)]
        ex = make_executor(cfg.n_workers)
        X = featurize(c, s1, t, ex)
        if ex:
            ex.shutdown()
        X["label"] = _label(X, truth)
        X.to_parquet(feats_path, index=False)
    X = pd.read_parquet(feats_path)
    cols = feature_columns(X)
    print(f"  training rows {len(X):,}, positives {X['label'].mean():.3f}, features {len(cols)}")
    params = cfg.extra.get("model_params", {})
    if cfg.extra.get("two_stage", True):
        from training import fit_two_stage
        import pickle
        b = fit_two_stage(X, cols, cfg.model, cfg.seed, params)
        b.pop("oof_p1")
        b["meta"] = {"n_rows": len(X), "kind": cfg.model, "two_stage": True}
        with open(Path(cfg.model_dir) / "matcher.pkl", "wb") as f:
            pickle.dump(b, f)
    else:
        y = X["label"].to_numpy()
        Xn = X[cols].to_numpy(np.float32)   # single float32 copy; drop the DataFrame
        n_rows = len(X)
        del X
        model = fit_model(Xn, y, cfg.model, cfg.seed, **params)
        save_model(model, cols, Path(cfg.model_dir) / "matcher.pkl",
                   {"n_rows": n_rows, "kind": cfg.model})


def _score_split(cfg: Config, split: str, out_path: Path, keep_rows=None,
                 keep_path: Path | None = None) -> None:
    """Stream-featurize every candidate of ``split`` and store p per pair.
    Features of S1 rows in ``keep_rows`` are also kept (for analysis)."""
    from training import cands_path, featurize, load_model, make_executor, predict
    bundle = load_model(Path(cfg.model_dir) / "matcher.pkl")
    s1, t = _stores(cfg, split)
    c = pd.read_parquet(cands_path(cfg.cache_dir, split))
    scores, kept = [], []

    def sink(df):
        df["p"] = predict(bundle, df)
        scores.append(df[["s1_row", "src", "t_row", "p"]].copy())
        if keep_rows is not None:
            k = df[np.isin(df["s1_row"].to_numpy(), keep_rows)]
            if len(k):
                kept.append(k)

    ex = make_executor(cfg.n_workers)
    featurize(c, s1, t, ex, chunk_s1=int(cfg.extra.get("score_chunk_s1", 60_000)), sink=sink)
    if ex:
        ex.shutdown()
    sc = pd.concat(scores, ignore_index=True)
    sc.to_parquet(out_path, index=False)
    print(f"  scored {len(sc):,} pairs -> {out_path.name}")
    if keep_path is not None and kept:
        pd.concat(kept, ignore_index=True).to_parquet(keep_path, index=False)


def stage_score_train(cfg: Config) -> None:
    s1_ids = _stores(cfg, "train")[0].entity_ids()
    _, va_rows = _split_rows(cfg, s1_ids)
    rng = np.random.default_rng(1)
    keep = rng.choice(va_rows, min(60_000, len(va_rows)), replace=False)
    _score_split(cfg, "train", cfg.path("scores_train.parquet"), keep,
                 cfg.path("valid_feats.parquet"))


def stage_evaluate(cfg: Config) -> None:
    """Validation: candidate recall + F0.5 for several decision rules."""
    from evaluation import candidate_stats, macro_fbeta
    from inference import select_expected_f, select_threshold
    from training import cands_path
    s1, t = _stores(cfg, "train")
    s1_ids = s1.entity_ids()
    t_ids = {s: t[s].entity_ids() for s in (2, 3)}
    truth = _truth_rows(cfg, s1_ids, t_ids)
    _, va_rows = _split_rows(cfg, s1_ids)
    va_set = np.isin
    c = pd.read_parquet(cands_path(cfg.cache_dir, "train"), columns=["s1_row", "src", "t_row"])
    sc = pd.read_parquet(cfg.path("scores_train.parquet"))

    def to_ids(d):
        tg = np.where(d["src"].to_numpy() == 2,
                      t_ids[2][np.minimum(d["t_row"].to_numpy(), len(t_ids[2]) - 1)],
                      t_ids[3][np.minimum(d["t_row"].to_numpy(), len(t_ids[3]) - 1)])
        return pd.DataFrame({"s1": s1_ids[d["s1_row"].to_numpy()], "tgt": tg})

    va_ids = s1_ids[va_rows]
    tr_va = truth[va_set(truth["s1_row"].to_numpy(), va_rows)]
    t_ids_df = to_ids(tr_va)
    cstats = candidate_stats(to_ids(c[va_set(c["s1_row"].to_numpy(), va_rows)]), t_ids_df, va_ids)
    print("  candidate stats (validation S1):", json.dumps(cstats))
    results = {}
    for o2o in (True, False):
        for thr in np.round(np.arange(0.10, 0.96, 0.05), 2):
            sel = select_threshold(sc, thr, o2o)
            sel = sel[va_set(sel["s1_row"].to_numpy(), va_rows)]
            r = macro_fbeta(to_ids(sel), t_ids_df, va_ids)
            results[f"thr={thr:.2f},o2o={o2o}"] = {k: v for k, v in r.items() if k != "per_s1"}
    for mm in (0.0, 0.05, 0.1):
        for fl in (0.05, 0.2):
            sel = select_expected_f(sc, True, miss_mass=mm, floor=fl)
            sel = sel[va_set(sel["s1_row"].to_numpy(), va_rows)]
            r = macro_fbeta(to_ids(sel), t_ids_df, va_ids)
            results[f"expf,mm={mm},floor={fl}"] = {k: v for k, v in r.items() if k != "per_s1"}
    for k, v in results.items():
        print(f"  {k:28s} F0.5={v['f05']:.5f} P={v['precision_micro']:.4f} "
              f"R={v['recall_micro']:.4f} singleton_acc={v['singleton_acc']:.4f}")
    best = max(results, key=lambda k: results[k]["f05"])
    print("  BEST:", best, results[best])
    dec = {"rule": best, "o2o": "o2o=True" in best or best.startswith("expf")}
    if best.startswith("thr"):
        dec.update(mode="thr", thr=float(best.split(",")[0].split("=")[1]))
    else:
        parts = dict(x.split("=") for x in best.split(",")[1:])
        dec.update(mode="expf", miss_mass=float(parts["mm"]), floor=float(parts["floor"]))
    dec["valid_f05"] = results[best]["f05"]
    Path(cfg.model_dir, "decision.json").write_text(json.dumps(dec, indent=1))
    _log_experiment(cfg, {"stage": "evaluate", "cand_k": cfg.extra.get("cand_k", (10, 3, 3)),
                          "candidate_stats": cstats, "best_rule": best,
                          "best": results[best], "all": results,
                          "model": cfg.model, "model_params": cfg.extra.get("model_params", {}),
                          "n_train_s1": cfg.extra.get("n_train_s1", 200_000)})


def stage_predict_test(cfg: Config) -> None:
    from inference import select_expected_f, select_threshold, to_id_lists, write_id_file
    from training import cands_path
    scores_path = cfg.path("scores_test.parquet")
    if not scores_path.exists():
        _score_split(cfg, "test", scores_path)
    dec = json.loads(Path(cfg.model_dir, "decision.json").read_text())
    s1, t = _stores(cfg, "test")
    s1_ids = s1.entity_ids()
    t_ids = {s: t[s].entity_ids() for s in (2, 3)}
    sc = pd.read_parquet(scores_path)
    c = pd.read_parquet(cands_path(cfg.cache_dir, "test"), columns=["s1_row", "src", "t_row"])
    assert len(sc) == len(c), "scores must cover exactly the saved candidate set"
    if dec["mode"] == "thr":
        sel = select_threshold(sc, dec["thr"], dec["o2o"])
    else:
        sel = select_expected_f(sc, dec["o2o"], miss_mass=dec["miss_mass"], floor=dec["floor"])
    out = Path(cfg.output_dir)
    write_id_file(out / "candidate_pairs.tsv", s1_ids, to_id_lists(c, s1_ids, s1_ids, t_ids),
                  "candidate_entity_ids")
    write_id_file(out / "matching_results.tsv", s1_ids, to_id_lists(sel, s1_ids, s1_ids, t_ids),
                  "matched_entity_ids")
    n_match = sel.groupby("s1_row").size()
    print(f"  wrote {out / 'matching_results.tsv'}: {len(sel):,} matches, "
          f"{len(n_match):,}/{len(s1_ids):,} S1 with >=1 match; "
          f"candidates {len(c):,} ({len(c) / len(s1_ids):.1f}/S1)")


def stage_validate(cfg: Config) -> None:
    from validation import validate_outputs
    ok = validate_outputs(cfg)
    if not ok:
        raise SystemExit(1)


STAGES = {"preprocess": stage_preprocess, "translit": stage_translit,
          "pool_train": stage_pool_train, "filter_train": stage_filter_train,
          "cands_train": stage_cands_train,
          "train": stage_train, "score_train": stage_score_train, "evaluate": stage_evaluate,
          "pool_test": stage_pool_test, "cands_test": stage_cands_test,
          "predict_test": stage_predict_test, "validate": stage_validate}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=list(STAGES) + ["all"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    stages = list(STAGES) if "all" in args.stages else args.stages
    for st in stages:
        t = time.time()
        print(f"== stage {st}", flush=True)
        STAGES[st](cfg)
        print(f"== stage {st} done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
