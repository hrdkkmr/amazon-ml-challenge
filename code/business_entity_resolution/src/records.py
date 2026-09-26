"""Load normalized caches and apply the learned transliteration layer.

``load_records`` returns one DataFrame per (split, source) with the columns
used by blocking and feature extraction:

    entity_id, country, name, core, alt, is_dom, script, addr, nums, state

``name``/``core`` are fully Latin (Indic tokens translated with the learned
dictionary, unknown ones rule-transliterated); ``addr`` has native-script
state names replaced by state codes learned from training pairs.
"""
from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from data_io import load_norm
from normalization import FILLER, HONORIFICS, LEGAL_TOKENS
from translit import TranslitDict, has_indic, rule_translit

_DROP = LEGAL_TOKENS | FILLER | HONORIFICS


def core_of(name: str) -> str:
    toks = [t for t in name.split() if t not in _DROP]
    return " ".join(toks) if toks else name


def learn_state_map(s1_addr: pd.Series, s1_state: pd.Series, tgt_addr: pd.Series) -> dict:
    """Map native-script address components to Latin state codes by voting
    over aligned (S1 state, target component) training pairs."""
    votes: dict[str, Counter] = defaultdict(Counter)
    for st, addr in zip(s1_state.tolist(), tgt_addr.tolist()):
        if not st or not addr or not has_indic(addr):
            continue
        for comp in addr.split(" , "):
            if has_indic(comp):
                votes[comp][st] += 1
    out = {}
    for comp, c in votes.items():
        best, n = c.most_common(1)[0]
        if n >= 5 and n / sum(c.values()) >= 0.6:
            out[comp] = best
    return out


class Enricher:
    def __init__(self, cache_dir: str):
        self.td = TranslitDict.load(Path(cache_dir) / "translit_dict.pkl")
        p = Path(cache_dir) / "state_map.pkl"
        self.state_map = pickle.loads(p.read_bytes()) if p.exists() else {}

    def name(self, s: str) -> str:
        return self.td.text(s)

    def addr(self, addr: str, state: str) -> tuple[str, str]:
        if not addr or not has_indic(addr):
            return addr, state
        comps = []
        for comp in addr.split(" , "):
            if has_indic(comp):
                code = self.state_map.get(comp)
                if code:
                    state = state or code
                    comp = code
                else:
                    comp = " ".join(rule_translit(t) for t in comp.split())
            comps.append(comp)
        return " , ".join(comps), state


def load_records(cache_dir: str, split: str, source: int, enricher: Enricher,
                 country: str | None = None, rows=None) -> pd.DataFrame:
    """Load + enrich records; optionally only one country or given cache rows."""
    cols = ["entity_id", "country", "name_norm", "name_alt", "is_dom", "script",
            "addr_norm", "addr_nums", "state"]
    df = load_norm(cache_dir, split, source, cols)
    if country is not None:
        df = df[df["country"] == country]
    if rows is not None:
        df = df.iloc[np.unique(np.asarray(rows))]
    rows = df.index.to_numpy(np.int32)   # position in the full cache file
    df = df.reset_index(drop=True)
    names = df["name_norm"].astype(object).tolist()
    scripts = df["script"].astype(object).tolist()
    name = [enricher.name(n) if sc != "L" else n for n, sc in zip(names, scripts)]
    addrs = [enricher.addr(a, s) for a, s in zip(df["addr_norm"].astype(object).tolist(),
                                                  df["state"].astype(object).tolist())]
    out = pd.DataFrame({
        "row": rows,
        "entity_id": df["entity_id"].astype(object).to_numpy(),
        "country": df["country"].astype(object).to_numpy(),
        "name": name,
        "core": [core_of(n) for n in name],
        "alt": df["name_alt"].astype(object).to_numpy(),
        "is_dom": df["is_dom"].to_numpy(np.int8),
        "script": df["script"].astype(object).to_numpy(),
        "addr": [a for a, _ in addrs],
        "nums": df["addr_nums"].astype(object).to_numpy(),
        "state": [s for _, s in addrs],
    })
    return out


class RecordStore:
    """Compact (Arrow-backed) normalized records of one (split, source); rows
    are materialized + enriched into Python strings only on request."""

    COLS = ["entity_id", "country", "name_norm", "name_alt", "is_dom", "script",
            "addr_norm", "addr_nums", "state"]

    def __init__(self, cache_dir: str, split: str, source: int, enricher: Enricher):
        self.df = load_norm(cache_dir, split, source, self.COLS)
        self.enr = enricher
        self.ids = None

    def __len__(self) -> int:
        return len(self.df)

    def entity_ids(self) -> np.ndarray:
        if self.ids is None:
            self.ids = self.df["entity_id"].astype(object).to_numpy()
        return self.ids

    def get(self, rows: np.ndarray) -> dict:
        """Aligned lists (one entry per element of ``rows``, duplicates allowed)
        for the feature columns: name, core, alt, is_dom, script, addr, nums, state."""
        uniq, inv = np.unique(np.asarray(rows), return_inverse=True)
        sub = self.df.iloc[uniq]
        names = sub["name_norm"].astype(object).tolist()
        scripts = sub["script"].astype(object).tolist()
        name = [self.enr.name(n) if sc != "L" else n for n, sc in zip(names, scripts)]
        addrs = [self.enr.addr(a, s) for a, s in zip(sub["addr_norm"].astype(object).tolist(),
                                                      sub["state"].astype(object).tolist())]
        cols = {
            "name": name,
            "core": [core_of(n) for n in name],
            "alt": sub["name_alt"].astype(object).tolist(),
            "is_dom": sub["is_dom"].to_numpy(np.int8).tolist(),
            "script": scripts,
            "addr": [a for a, _ in addrs],
            "nums": sub["addr_nums"].astype(object).tolist(),
            "state": [s for _, s in addrs],
        }
        inv = inv.tolist()
        return {k: [v[i] for i in inv] for k, v in cols.items()}

    _freq = None

    def core_counts(self) -> dict:
        """{core name: number of records in this file with that core} (cached)."""
        if self._freq is None:
            names = self.df["name_norm"].astype(object).tolist()
            scripts = self.df["script"].astype(object).tolist()
            cores = [core_of(self.enr.name(n) if sc != "L" else n) for n, sc in zip(names, scripts)]
            self._freq = pd.Series(cores).value_counts().to_dict()
        return self._freq
