"""Memory-safe profile of every dataset file (chunked reads, hashed keys).

Writes experiments/data_profile.json and prints a human-readable summary.
Usage: python scripts/inspect_data.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHUNK = 500_000
SOURCES = {
    "train": ["train_source1", "train_source2", "train_source3"],
    "test": ["test_source1", "test_source2", "test_source3"],
}


def id_to_int(s: pd.Series) -> np.ndarray:
    return s.str.slice(3).astype(np.int64).to_numpy()


def profile_source(path: Path) -> dict:
    n = 0
    nulls = {}
    countries = {}
    ids, keys = [], []
    name_len, addr_len = [], []
    bad_prefix = 0
    t = time.time()
    for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                          na_values=[""], chunksize=CHUNK, quoting=3):
        n += len(ch)
        for c in ch.columns:
            nulls[c] = nulls.get(c, 0) + int(ch[c].isna().sum())
        for k, v in ch["country"].fillna("<NA>").value_counts().items():
            countries[k] = countries.get(k, 0) + int(v)
        prefix = path.stem.split("_")[1].replace("source", "S") + "-"
        bad_prefix += int((~ch["entity_id"].str.startswith(prefix)).sum())
        ids.append(id_to_int(ch["entity_id"]))
        norm = (ch["business_name"].fillna("").str.lower().str.strip() + "|" +
                ch["business_address"].fillna("").str.lower().str.strip())
        keys.append(pd.util.hash_pandas_object(norm, index=False).to_numpy())
        name_len.append(ch["business_name"].fillna("").str.len().to_numpy(np.int32))
        addr_len.append(ch["business_address"].fillna("").str.len().to_numpy(np.int32))
        cols = list(ch.columns)
    ids = np.concatenate(ids)
    keys = np.concatenate(keys)
    nl, al = np.concatenate(name_len), np.concatenate(addr_len)
    return {
        "rows": n, "columns": cols, "size_mb": round(path.stat().st_size / 2**20, 1),
        "null_counts": nulls, "countries": countries,
        "unique_ids": int(np.unique(ids).size), "bad_prefix": bad_prefix,
        "dup_name_addr_rows": int(n - np.unique(keys).size),
        "name_len_p50_p99": [int(np.percentile(nl, 50)), int(np.percentile(nl, 99))],
        "addr_len_p50_p99": [int(np.percentile(al, 50)), int(np.percentile(al, 99))],
        "seconds": round(time.time() - t, 1),
    }


def profile_gt(path: Path) -> dict:
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    lists = gt["matched_entity_ids"].str.split(",")
    lists = lists.apply(lambda l: [x for x in l if x])
    n_match = lists.str.len()
    flat = pd.Series([x for l in lists for x in l])
    n_s2 = lists.apply(lambda l: sum(x.startswith("S2-") for x in l))
    n_s3 = n_match - n_s2
    target_dup = int(flat.duplicated().sum())
    return {
        "rows": len(gt), "unique_s1": int(gt["source1_entity_id"].nunique()),
        "singleton_rate": float((n_match == 0).mean()),
        "matches_per_s1_dist": {int(k): int(v) for k, v in n_match.value_counts().sort_index().items()},
        "s2_per_s1_dist": {int(k): int(v) for k, v in n_s2.value_counts().sort_index().items()},
        "s3_per_s1_dist": {int(k): int(v) for k, v in n_s3.value_counts().sort_index().items()},
        "total_pairs": int(len(flat)),
        "targets_matched_to_multiple_s1": target_dup,
    }


def main():
    out = {}
    for split, names in SOURCES.items():
        for name in names:
            p = ROOT / "dataset" / split / f"{name}.tsv"
            print("profiling", p.name, flush=True)
            out[name] = profile_source(p)
            print(json.dumps(out[name], indent=1), flush=True)
    print("profiling ground truth", flush=True)
    out["train_ground_truth"] = profile_gt(ROOT / "dataset" / "train" / "train_ground_truth.tsv")
    print(json.dumps(out["train_ground_truth"], indent=1))
    (ROOT / "experiments").mkdir(exist_ok=True)
    (ROOT / "experiments" / "data_profile.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
