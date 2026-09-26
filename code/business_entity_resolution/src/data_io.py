"""Chunked TSV reading and normalized-record caching (Parquet).

Raw files are streamed in chunks, normalized in a process pool and written to
``cache/norm_<split>_<source>.parquet``.  Downstream stages read only the
columns they need from these caches, never the raw TSVs.
"""
from __future__ import annotations

import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from normalization import normalize_address, normalize_name

RAW_COLS = ["entity_id", "business_name", "business_address", "country"]
NORM_COLS = ["name_norm", "name_core", "name_alt", "is_dom", "script",
             "addr_norm", "addr_nums", "state"]


def raw_path(data_dir: str, split: str, source: int) -> Path:
    return Path(data_dir) / split / f"{split}_source{source}.tsv"


def norm_path(cache_dir: str, split: str, source: int) -> Path:
    return Path(cache_dir) / f"norm_{split}_s{source}.parquet"


def read_tsv_chunks(path: Path, chunksize: int):
    """Yield DataFrames of string columns; empty fields become ''."""
    yield from pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                           na_filter=False, quoting=csv.QUOTE_NONE,
                           chunksize=chunksize, encoding="utf-8")


def _normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    names = [normalize_name(x) for x in df["business_name"].tolist()]
    addrs = [normalize_address(a, c) for a, c in
             zip(df["business_address"].tolist(), df["country"].tolist())]
    out = pd.DataFrame({
        "entity_id": df["entity_id"].to_numpy(),
        "country": df["country"].str.strip().to_numpy(),
    })
    n = list(zip(*names)) if names else [[]] * 5
    out["name_norm"], out["name_core"], out["name_alt"] = n[0], n[1], n[2]
    out["is_dom"] = np.asarray(n[3], dtype=np.int8)
    out["script"] = n[4]
    a = list(zip(*addrs)) if addrs else [[]] * 3
    out["addr_norm"], out["addr_nums"], out["state"] = a[0], a[1], a[2]
    return out


def preprocess_file(src: Path, dst: Path, chunksize: int, n_workers: int) -> int:
    """Normalize one raw TSV into a Parquet cache. Returns row count."""
    t = time.time()
    tmp = dst.with_suffix(".tmp")
    writer = None
    n = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for out in ex.map(_normalize_chunk, read_tsv_chunks(src, chunksize),
                          buffersize=2 * n_workers):  # bounded read-ahead
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            n += len(out)
    writer.close()
    tmp.replace(dst)
    print(f"  {src.name}: {n:,} rows -> {dst.name} ({time.time() - t:.0f}s)", flush=True)
    return n


def load_norm(cache_dir: str, split: str, source: int, columns=None) -> pd.DataFrame:
    """Load a normalized cache (Arrow-backed strings keep memory compact)."""
    return pd.read_parquet(norm_path(cache_dir, split, source), columns=columns,
                           dtype_backend="pyarrow")


def load_ground_truth(data_dir: str) -> pd.DataFrame:
    """Return long-format truth: columns s1, tgt (one row per true pair) plus
    the list of all S1 ids (including singletons) as attribute ``all_s1``."""
    gt = pd.read_csv(Path(data_dir) / "train" / "train_ground_truth.tsv", sep="\t",
                     dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    long = gt.assign(tgt=gt["matched_entity_ids"].str.split(",")).explode("tgt")
    long = long[long["tgt"].astype(bool)][["source1_entity_id", "tgt"]]
    long.columns = ["s1", "tgt"]
    long.attrs["all_s1"] = gt["source1_entity_id"].to_numpy()
    return long.reset_index(drop=True)
