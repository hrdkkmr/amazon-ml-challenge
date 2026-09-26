"""Blocking: sparse inverted indexes over rare name/address key combinations.

Every record emits three *families* of blocking keys (all order-insensitive,
because sources reorder both name words and address components):

  name   n:<tok>            name core token
         c:<concat>         concatenated core tokens (matches domain-style names)
         nn:<t1|t2>         unordered pair of name tokens
  addr   a:<tok>            informative address token (generic words removed)
         aa:<t1|t2>         unordered pair of address tokens
  cross  na:<name|addr>     name token x address token

Single tokens are often very common in this data (e.g. "beacon", "telecom",
"des moines"), but their *combinations* are rare, so pair keys survive the
document-frequency cap that removes uninformative keys.

Keys are hashed to stable 64-bit integers.  For each (target source, country)
and key family we build a CSR postings matrix (key -> target records) keeping
only keys with df <= cap.  A query scores targets by the sum of IDF weights of
shared keys, computed as sparse matrix products in small query batches; the
top-K targets per query and source are kept.  Nothing is compared all-vs-all.
"""
from __future__ import annotations

import re
import time
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp

_HASH_KEY = "er2026blockkey16"   # fixed -> deterministic across runs/processes
FAMILIES = ("name", "addr", "cross")

ADDR_STOP = {
    "street", "road", "avenue", "drive", "lane", "court", "place", "boulevard",
    "circle", "way", "trail", "parkway", "highway", "terrace", "square", "unit",
    "floor", "building", "near", "opposite", "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest", "and", "of", "the", "plot",
    "flat", "door", "house", "block", "sector", "nagar", "city", "county", "district",
    "post", "box", "pmb", "cdp", "township", "main", "cross", "ground", "first",
    "second", "office", "shop", "village", "taluka", "colony", "complex", "area",
    "phase", "stage", "layout", "extension", "india", "usa", "us", "france", "po",
    "bldg", "apartment", "apartments", "tower", "wing", "gali", "marg", "chowk",
    "rue", "avenue", "de", "du", "des", "la", "le", "les", "chemin", "allee", "bis",
}
_ORD_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
_ALNUM_RE = re.compile(r"^[a-z]*(\d+)[a-z]*$")
_ORD_WORDS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
              "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10"}
MAX_NAME_TOKS = 6
MAX_ADDR_TOKS = 10


def name_tokens(core: str) -> list[str]:
    out = []
    for t in core.split():
        if (len(t) > 1 or t.isdigit()) and t not in out:
            out.append(t)
    return out[:MAX_NAME_TOKS]


def addr_tokens(addr: str, state: str) -> list[str]:
    """Informative address tokens (order of appearance, deduplicated).

    Ordinals are reduced to digits (18th/18st -> 18, sixth -> 6) and
    alphanumeric house numbers also emit their digit part (b239 -> 239).
    """
    out = []
    for t in addr.split():
        if t == "," or t == state:
            continue
        t = _ORD_WORDS.get(t, t)
        m = _ORD_RE.match(t)
        if m:
            t = m.group(1)
        if t in ADDR_STOP or (len(t) < 2 and not t.isdigit()):
            continue
        if t not in out:
            out.append(t)
        if not t.isdigit():
            m = _ALNUM_RE.match(t)
            if m and m.group(1) not in out:
                out.append(m.group(1))
    return out[:MAX_ADDR_TOKS]


def record_keys(core: str, is_dom: int, addr: str, state: str) -> tuple[list, list, list]:
    nt = name_tokens(core)
    at = addr_tokens(addr, state)
    name = ["n:" + t for t in nt]
    if nt:
        name.append("c:" + "".join(nt))   # "holy lutheran church" ~ "holylutheranchurch"
    name += ["nn:" + "|".join(sorted(p)) for p in combinations(nt, 2)]
    addr_k = ["a:" + t for t in at]
    addr_k += ["aa:" + "|".join(sorted(p)) for p in combinations(at, 2)]
    cross = ["na:" + n + "|" + a for n in nt for a in at]
    return name, addr_k, cross


def hash_keys(keys: list[str]) -> np.ndarray:
    if not keys:
        return np.zeros(0, dtype=np.uint64)
    return pd.util.hash_array(np.asarray(keys, dtype=object), hash_key=_HASH_KEY,
                              categorize=False)


def record_key_arrays(df: pd.DataFrame, chunk: int = 100_000) -> dict:
    """{family: (row_positions int32, key_hashes uint64)} for all records.

    Keys are hashed chunk by chunk so that at most ``chunk`` records' worth of
    Python strings exist at any time.
    """
    cols = (df["core"].tolist(), df["is_dom"].tolist(), df["addr"].tolist(),
            df["state"].tolist())
    acc = {f: ([], []) for f in FAMILIES}
    n = len(cols[0])
    for c0 in range(0, n, chunk):
        rows = {f: [] for f in FAMILIES}
        keys = {f: [] for f in FAMILIES}
        for i in range(c0, min(c0 + chunk, n)):
            for f, k in zip(FAMILIES, record_keys(cols[0][i], cols[1][i], cols[2][i], cols[3][i])):
                keys[f].extend(k)
                rows[f].extend([i] * len(k))
        for f in FAMILIES:
            acc[f][0].append(np.asarray(rows[f], dtype=np.int32))
            acc[f][1].append(hash_keys(keys[f]))
    return {f: (np.concatenate(acc[f][0]) if acc[f][0] else np.zeros(0, np.int32),
                np.concatenate(acc[f][1]) if acc[f][1] else np.zeros(0, np.uint64))
            for f in FAMILIES}


class KeyIndex:
    """CSR postings (key -> target rows) for one key family."""

    def __init__(self, rows: np.ndarray, hashes: np.ndarray, n_targets: int, df_cap: int):
        order = np.argsort(hashes)
        h = hashes[order]
        post = rows[order]
        del order
        if len(h):
            starts = np.flatnonzero(np.r_[True, h[1:] != h[:-1]])
        else:
            starts = np.zeros(0, dtype=np.int64)
        vocab = h[starts]
        df = np.diff(np.r_[starts, len(h)]).astype(np.int32)
        del h
        keep = df <= df_cap
        post = post[np.repeat(keep, df)]
        self.vocab = vocab[keep]
        self.df = df[keep]
        self.idf = np.log1p(n_targets / self.df).astype(np.float32)
        # int32 indptr (nnz < 2**31) so scipy never upcasts/copies the indices
        indptr = np.zeros(len(self.vocab) + 1, dtype=np.int32)
        np.cumsum(self.df, out=indptr[1:])
        self.postings = sp.csr_matrix(
            (np.ones(len(post), dtype=np.float32), post, indptr),
            shape=(len(self.vocab), n_targets))
        self.n_dropped = int((~keep).sum())

    def query_matrix(self, rows: np.ndarray, hashes: np.ndarray, n_q: int) -> sp.csr_matrix:
        if len(self.vocab) == 0:
            return sp.csr_matrix((n_q, 0), dtype=np.float32)
        pos = np.searchsorted(self.vocab, hashes)
        pos[pos >= len(self.vocab)] = 0
        hit = self.vocab[pos] == hashes
        r, c = rows[hit], pos[hit]
        return sp.csr_matrix((self.idf[c], (r, c)), shape=(n_q, len(self.vocab)))

    def nbytes(self) -> int:
        p = self.postings
        return p.data.nbytes + p.indices.nbytes + p.indptr.nbytes + self.vocab.nbytes

    _ARRAYS = ("vocab", "idf", "indptr", "indices", "data")

    def save(self, prefix: str) -> None:
        arr = {"vocab": self.vocab, "idf": self.idf, "indptr": self.postings.indptr,
               "indices": self.postings.indices, "data": self.postings.data}
        for k, v in arr.items():
            np.save(f"{prefix}_{k}.npy", v)
        np.save(f"{prefix}_shape.npy", np.array(self.postings.shape, dtype=np.int64))

    @classmethod
    def load(cls, prefix: str) -> "KeyIndex":
        """Memory-mapped load: pages are shared by all processes via the OS cache."""
        obj = cls.__new__(cls)
        a = {k: np.load(f"{prefix}_{k}.npy", mmap_mode="r") for k in cls._ARRAYS}
        shape = tuple(np.load(f"{prefix}_shape.npy"))
        obj.vocab, obj.idf = a["vocab"], a["idf"]
        obj.postings = sp.csr_matrix((a["data"], a["indices"], a["indptr"]), shape=shape,
                                     copy=False)
        obj.n_dropped = 0
        return obj


def build_indexes(t: pd.DataFrame, df_cap: int) -> dict:
    """Build one KeyIndex per key family for target records ``t``."""
    t0 = time.time()
    arrs = record_key_arrays(t)
    idx = {}
    for f in FAMILIES:
        r, h = arrs.pop(f)
        idx[f] = KeyIndex(r, h, len(t), df_cap)
        del r, h
    print(f"    index: {len(t):,} targets | " + " | ".join(
        f"{f} keys {len(i.vocab):,} (dropped {i.n_dropped})" for f, i in idx.items()) +
        f" | {sum(i.nbytes() for i in idx.values()) / 2**20:.0f} MB ({time.time() - t0:.0f}s)",
        flush=True)
    return idx


def _rank_within(row: np.ndarray) -> np.ndarray:
    """Position of each element within its run of equal (sorted) row ids."""
    if len(row) == 0:
        return np.zeros(0, dtype=np.int64)
    first = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1]
    return np.arange(len(row)) - np.repeat(first, np.diff(np.r_[first, len(row)]))


def _topk_mask(row: np.ndarray, primary: np.ndarray, secondary: np.ndarray, k: int):
    """Boolean mask of the k best entries per row by (primary, secondary) desc,
    plus the rank array (in original element order).  Scores are quantized
    into one packed int64 sort key (row | -primary | -secondary)."""
    qp = np.clip(np.round(primary * 64), 0, 2**20 - 1).astype(np.int64)
    qs = np.clip(np.round(secondary * 16), 0, 2**12 - 1).astype(np.int64)
    key = (row.astype(np.int64) << 32) | ((2**20 - 1 - qp) << 12) | (2**12 - 1 - qs)
    order = np.argsort(key, kind="stable")
    rank_sorted = _rank_within(row[order])
    rank = np.empty(len(row), dtype=np.int64)
    rank[order] = rank_sorted
    return (rank < k) & (primary > 0), rank


def retrieve(q: pd.DataFrame, idx: dict, k_total: int, k_name: int = 0, k_addr: int = 0,
             batch: int = 1000, verbose: bool = True) -> pd.DataFrame:
    """Candidate targets per query row: union of three per-query rankings.

      * top ``k_total`` by total score (name + addr + cross)
      * top ``k_name``  by name score   (finds matches whose address is empty/garbled)
      * top ``k_addr``  by address score (finds matches with alias / garbled names)

    Returns q_pos, t_pos, s_name, s_addr, s_cross, rank, rank_name, rank_addr.
    """
    out = []
    t0 = time.time()
    nt = idx["name"].postings.shape[1]
    todo = [(b0, min(b0 + batch, len(q))) for b0 in range(0, len(q), batch)][::-1]
    while todo:
        b0, b1 = todo.pop()
        qb = q.iloc[b0:b1]
        n = len(qb)
        arrs = record_key_arrays(qb)
        try:
            coos = []
            for f in FAMILIES:
                r, h = arrs[f]
                coos.append((idx[f].query_matrix(r, h, n) @ idx[f].postings).tocoo())
            code = np.concatenate([c.row.astype(np.int64) * nt + c.col for c in coos])
        except MemoryError:
            if n == 1:
                raise
            mid = (b0 + b1) // 2          # split the batch and retry
            todo += [(mid, b1), (b0, mid)]
            print(f"    MemoryError on batch of {n}; splitting", flush=True)
            continue
        if len(code) == 0:
            continue
        uniq, inv = np.unique(code, return_inverse=True)
        parts, off = [], 0
        for c in coos:
            parts.append(np.bincount(inv[off:off + c.nnz], weights=c.data, minlength=len(uniq)))
            off += c.nnz
        s_name, s_addr, s_cross = parts
        tot = s_name + s_addr + s_cross
        row = (uniq // nt).astype(np.int32)
        keep, rank = _topk_mask(row, tot, tot, k_total)
        _, rank_n = _topk_mask(row, s_name, tot, max(k_name, 1))
        _, rank_a = _topk_mask(row, s_addr, tot, max(k_addr, 1))
        if k_name:
            keep |= (rank_n < k_name) & (s_name > 0)
        if k_addr:
            keep |= (rank_a < k_addr) & (s_addr > 0)
        sel = np.flatnonzero(keep)
        out.append(pd.DataFrame({
            "q_pos": row[sel] + b0,
            "t_pos": (uniq[sel] % nt).astype(np.int32),
            "s_name": s_name[sel].astype(np.float32),
            "s_addr": s_addr[sel].astype(np.float32),
            "s_cross": s_cross[sel].astype(np.float32),
            "rank": np.minimum(rank[sel], 32767).astype(np.int16),
            "rank_name": np.minimum(rank_n[sel], 32767).astype(np.int16),
            "rank_addr": np.minimum(rank_a[sel], 32767).astype(np.int16),
        }))
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        {c: [] for c in ["q_pos", "t_pos", "s_name", "s_addr", "s_cross", "rank",
                         "rank_name", "rank_addr"]})
    if verbose:
        print(f"    retrieved {len(res):,} pairs for {len(q):,} queries "
              f"({time.time() - t0:.0f}s)", flush=True)
    return res


# ------------------------------------------------------------ parallel driver
_WORKER_IDX: dict = {}


def _worker_retrieve(args):
    prefix, cols, offset, k = args
    if _WORKER_IDX.get("prefix") != prefix:
        _WORKER_IDX.clear()
        _WORKER_IDX["prefix"] = prefix
        _WORKER_IDX["idx"] = {f: KeyIndex.load(f"{prefix}_{f}") for f in FAMILIES}
    q = pd.DataFrame(cols)
    r = retrieve(q, _WORKER_IDX["idx"], *k, verbose=False)
    r["q_pos"] += offset
    return r


def retrieve_parallel(q: pd.DataFrame, idx: dict, k: tuple, prefix: str, executor,
                      chunk: int = 10_000) -> pd.DataFrame:
    """Same result as ``retrieve`` but spread over a process pool; the index is
    written once to ``prefix``*.npy and memory-mapped by the workers."""
    t0 = time.time()
    for f in FAMILIES:
        idx[f].save(f"{prefix}_{f}")
    jobs = []
    for c0 in range(0, len(q), chunk):
        sub = q.iloc[c0:c0 + chunk]
        cols = {c: sub[c].tolist() for c in ("core", "is_dom", "addr", "state")}
        jobs.append((prefix, cols, c0, k))
    parts = list(executor.map(_worker_retrieve, jobs))
    res = pd.concat(parts, ignore_index=True)
    print(f"    retrieved {len(res):,} pairs for {len(q):,} queries "
          f"({time.time() - t0:.0f}s, parallel)", flush=True)
    return res
