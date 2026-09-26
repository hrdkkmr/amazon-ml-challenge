"""Candidate generation for a whole split.

For each S1 country (an open set: whatever labels appear in the data) and
each target source, build the blocking indexes on that country's targets and
retrieve a *candidate pool* for every S1 of that country.  The pool is saved
with all blocking scores and ranks so the final candidate cut can be tuned
without re-running retrieval:

    cache/pool_<split>_k<K>/<country>_s<src>.parquet
        s1_row, src, t_row, s_name, s_addr, s_cross, rank, rank_name, rank_addr

Rows index into the normalized caches (``records.load_records(...)["row"]``).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from blocking import FAMILIES, build_indexes, retrieve_parallel
from concurrent.futures import ProcessPoolExecutor
from data_io import load_norm
from records import Enricher, load_records

POOL_COLS = ["s1_row", "src", "t_row", "s_name", "s_addr", "s_cross", "rank",
             "rank_name", "rank_addr"]


def pool_dir(cfg, split: str) -> Path:
    """Directory holding one Parquet file per (country, source) partition."""
    k = "-".join(str(x) for x in cfg.extra.get("pool_k", (15, 5, 5)))
    return Path(cfg.cache_dir) / f"pool_{split}_k{k}"


def iter_pool(cfg, split: str):
    """Yield the pool one (country, source) partition at a time.  Every
    target belongs to exactly one partition, so target-side competition
    features computed per partition are exact."""
    d = pool_dir(cfg, split)
    assert (d / "_DONE").exists(), f"pool {d} incomplete - run pool_{split}"
    for part in sorted(d.glob("*.parquet")):
        yield part.stem, pd.read_parquet(part)


def generate_pool(cfg, split: str, k=(15, 5, 5)) -> Path:
    out = pool_dir(cfg, split)
    if (out / "_DONE").exists():
        print(f"  cached: {out.name}")
        return out
    out.mkdir(exist_ok=True)
    enr = Enricher(cfg.cache_dir)
    proc = psutil.Process()
    countries = load_norm(cfg.cache_dir, split, 1, ["country"])["country"].astype(object)
    countries = sorted(countries.unique().tolist())
    peak = 0
    idx_dir = Path(cfg.cache_dir) / "idx"
    idx_dir.mkdir(exist_ok=True)
    ex = ProcessPoolExecutor(max_workers=int(cfg.extra.get("retrieve_workers", 6)))
    for country in countries:
        q = None
        for src in (2, 3):
            part = out / f"{country.replace(' ', '_')}_s{src}.parquet"
            if part.exists():               # checkpoint: partition already done
                print(f"  [{split}] {country} S{src}: cached partition", flush=True)
                continue
            if q is None:
                q = load_records(cfg.cache_dir, split, 1, enr, country)
            t0 = time.time()
            t = load_records(cfg.cache_dir, split, src, enr, country)
            print(f"  [{split}] {country} S{src}: {len(q):,} S1 vs {len(t):,} targets", flush=True)
            if len(t) == 0 or len(q) == 0:
                continue
            idx = build_indexes(t, cfg.df_cap_name)
            prefix = str(idx_dir / f"{split}_{country.replace(' ', '_')}_s{src}")
            r = retrieve_parallel(q, idx, k, prefix, ex)
            del idx
            peak = max(peak, proc.memory_info().rss)
            df = pd.DataFrame({
                "s1_row": q["row"].to_numpy()[r["q_pos"].to_numpy()],
                "src": np.int8(src),
                "t_row": t["row"].to_numpy()[r["t_pos"].to_numpy()],
            })
            for c in POOL_COLS[3:]:
                df[c] = r[c].to_numpy()
            df.to_parquet(part.with_suffix(".tmp"), index=False, compression="zstd")
            part.with_suffix(".tmp").replace(part)
            del t, r, df
            for f in idx_dir.glob("*.npy"):
                try:
                    f.unlink()     # workers may still hold the mmap on Windows
                except OSError:
                    pass
            print(f"    done in {time.time() - t0:.0f}s, rss {proc.memory_info().rss / 2**30:.2f} GB",
                  flush=True)
        del q
    ex.shutdown()
    for f in idx_dir.glob("*.npy"):
        f.unlink(missing_ok=True)
    (out / "_DONE").write_text("ok")
    print(f"  pool written: {out.name}, peak rss {peak / 2**30:.2f} GB")
    return out


def select_candidates(pool: pd.DataFrame, k_total: int, k_name: int, k_addr: int) -> pd.DataFrame:
    """Final candidate cut from a pool (union of the three rankings)."""
    m = (pool["rank"] < k_total)
    if k_name:
        m |= (pool["rank_name"] < k_name) & (pool["s_name"] > 0)
    if k_addr:
        m |= (pool["rank_addr"] < k_addr) & (pool["s_addr"] > 0)
    return pool[m]
