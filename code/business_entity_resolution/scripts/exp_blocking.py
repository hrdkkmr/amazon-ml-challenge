"""Blocking recall experiment on a sample of validation S1 entities.

Usage: python scripts/exp_blocking.py [n_sample] [df_cap] [k]
Prints recall@K per source and country plus candidate-count statistics.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from blocking import build_indexes, retrieve  # noqa: E402
from config import Config  # noqa: E402
from data_io import load_ground_truth  # noqa: E402
from evaluation import split_s1  # noqa: E402
from records import Enricher, load_records  # noqa: E402

n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
cap = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
K = int(sys.argv[3]) if len(sys.argv) > 3 else 50

cfg = Config.load()
proc = psutil.Process()
peak = 0
enr = Enricher(cfg.cache_dir)
gt = load_ground_truth(cfg.data_dir)
_, valid_ids = split_s1(gt.attrs["all_s1"], cfg.valid_frac, cfg.seed)
rng = np.random.default_rng(0)
sample = set(rng.choice(valid_ids, n_sample, replace=False).tolist())
truth = gt[gt["s1"].isin(sample)]
res_all = []
for country in ["US", "India"]:
    q = load_records(cfg.cache_dir, "train", 1, enr, country)
    q = q[q["entity_id"].isin(sample)].reset_index(drop=True)
    for src in (2, 3):
        t0 = time.time()
        t = load_records(cfg.cache_dir, "train", src, enr, country)
        idx = build_indexes(t, cap)
        peak = max(peak, proc.memory_info().rss)
        r = retrieve(q, idx, K, K, K)
        peak = max(peak, proc.memory_info().rss)
        r["s1"] = q["entity_id"].to_numpy()[r["q_pos"]]
        r["tgt"] = t["entity_id"].to_numpy()[r["t_pos"]]
        r["country"] = country
        r["src"] = src
        res_all.append(r[["s1", "tgt", "rank", "s_name", "s_addr", "s_cross", "rank_name", "rank_addr", "country", "src"]])
        del t, idx
        print(f"  {country} S{src} done {time.time() - t0:.0f}s rss={proc.memory_info().rss / 2**30:.2f}GB", flush=True)
res = pd.concat(res_all, ignore_index=True)
m = truth.merge(res, on=["s1", "tgt"], how="left")
m["src"] = np.where(m["tgt"].str.startswith("S2-"), 2, 3)
out = {"n_sample": n_sample, "df_cap": cap, "n_true_pairs": len(truth)}
def sel(d, kt, kn, ka):
    return (d["rank"] < kt) | ((d["rank_name"] < kn) & (d["s_name"] > 0)) | ((d["rank_addr"] < ka) & (d["s_addr"] > 0))
for kt, kn, ka in [(5, 0, 0), (10, 0, 0), (20, 0, 0), (50, 0, 0), (5, 2, 2), (5, 3, 3), (8, 3, 3),
                   (8, 5, 3), (10, 3, 3), (10, 5, 5), (15, 5, 5), (20, 10, 10), (30, 20, 20), (50, 50, 50)]:
    hit = m["rank"].notna() & sel(m.fillna(99999), kt, kn, ka)
    per_src = {s: float(hit[m["src"] == s].mean()) for s in (2, 3)}
    ncand = sel(res, kt, kn, ka).sum() / n_sample
    out[f"recall@{kt},{kn},{ka}"] = [round(float(hit.mean()), 4), round(per_src[2], 4), round(per_src[3], 4), round(float(ncand), 1)]
    print(f"K={kt},{kn},{ka} recall={hit.mean():.4f} S2={per_src[2]:.4f} S3={per_src[3]:.4f} cand/S1={ncand:.1f}")
out["peak_rss_gb"] = round(peak / 2**30, 2)
print("peak RSS GB", out["peak_rss_gb"])
miss = m[m["rank"].isna()]
print("missed pairs (no candidate at all):", len(miss), "of", len(m))
Path(cfg.experiments_dir, "blocking_runs.jsonl").open("a").write(json.dumps(out) + "\n")
res.to_parquet(Path(cfg.cache_dir, "exp_blocking_res.parquet"))
miss.to_parquet(Path(cfg.cache_dir, "exp_blocking_miss.parquet"))
