"""Pairwise features, computed only for candidate pairs.

Two groups:
  * string similarities via rapidfuzz ``process.cpdist`` (vectorized C++,
    multi-threaded) on aligned lists of S1 / target strings;
  * token-set features (coverage, numbers, state, legal-form agreement,
    phonetic overlap) in a pure-Python pass parallelised over chunks.

Context features (competition among candidates of the same S1 and among S1s
competing for the same target) are added later in ``add_context_features``.
"""
from __future__ import annotations



import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from blocking import addr_tokens
from normalization import LEGAL_TOKENS
from translit import phonetic

# columns needed from records.load_records
REC_COLS = ["name", "core", "alt", "is_dom", "script", "addr", "nums", "state"]

_SCRIPT = {"L": 0, "I": 1, "M": 2}
_LEGAL_GROUPS = {"pvt": "pvt", "ltd": "ltd", "llc": "llc", "inc": "inc", "corp": "inc",
                 "co": "co", "llp": "llp", "pllc": "llc", "pc": "pc", "plc": "ltd",
                 "sarl": "sarl", "sas": "sas", "sa": "sa", "eurl": "eurl", "lp": "lp"}


_PHONE = re.compile(r"\s*\b\d{7,}\b")


def strip_phone(core: str) -> str:
    """Drop phone-number-like digit runs appended to names."""
    return _PHONE.sub("", core).strip() or core


def _flat_addr(a: str) -> str:
    return a.replace(" , ", " ")


def _cov(small: list, big: set, fuzzy: bool) -> float:
    """Fraction of tokens of ``small`` present in ``big`` (exactly, or with
    >= 85 similarity when ``fuzzy``)."""
    if not small:
        return -1.0
    hit = 0
    for t in small:
        if t in big:
            hit += 1
        elif fuzzy and len(t) > 3:
            for u in big:
                if abs(len(u) - len(t)) <= 2 and fuzz.ratio(t, u) >= 85:
                    hit += 1
                    break
    return hit / len(small)


def _py_features(args) -> np.ndarray:
    """Token-level features for a chunk of pairs; returns float32 matrix."""
    a_core, b_core, a_name, b_name, a_addr, b_addr, a_state, b_state, a_nums, b_nums = args
    n = len(a_core)
    out = np.full((n, len(PY_FEATS)), -1.0, dtype=np.float32)
    for i in range(n):
        ac, bc = a_core[i].split(), b_core[i].split()
        acs, bcs = set(ac), set(bc)
        an_full, bn_full = a_name[i].split(), b_name[i].split()
        la = {_LEGAL_GROUPS[t] for t in an_full if t in _LEGAL_GROUPS}
        lb = {_LEGAL_GROUPS[t] for t in bn_full if t in _LEGAL_GROUPS}
        at = addr_tokens(a_addr[i], a_state[i])
        bt = addr_tokens(b_addr[i], b_state[i])
        ats, bts = set(at), set(bt)
        an = a_nums[i].split()
        bn = b_nums[i].split()
        ans, bns = set(an), set(bn)
        pa = {phonetic(t) for t in ac}
        pb = {phonetic(t) for t in bc}
        inter = len(acs & bcs)
        out[i, 0] = inter / max(len(acs | bcs), 1)                     # name jaccard
        out[i, 1] = _cov(ac, bcs, True)                                  # S1 name covered by target
        out[i, 2] = _cov(bc, acs, True)                                  # target name covered by S1
        out[i, 3] = len(pa & pb) / max(len(pa | pb), 1)                  # phonetic jaccard
        out[i, 4] = float(bool(ac) and bool(bc) and ac[0] == bc[0])      # first token equal
        out[i, 5] = float(sorted(ac) == sorted(bc))                      # same bag of tokens
        out[i, 6] = -1.0 if not (la and lb) else float(bool(la & lb))    # legal-form agreement
        out[i, 7] = float(bool(la) and bool(lb) and not (la & lb))       # legal-form conflict
        out[i, 8] = len(ac)
        out[i, 9] = len(bc)
        out[i, 10] = _cov(bt, ats, True)                                 # target addr covered by S1
        out[i, 11] = _cov(at, bts, True)                                 # S1 addr covered by target
        out[i, 12] = len(at)
        out[i, 13] = len(bt)
        if ans and bns:
            out[i, 14] = len(ans & bns) / len(ans | bns)                 # number jaccard
            out[i, 15] = len(bns - ans) / len(bns)                       # target numbers unseen in S1
            out[i, 16] = float(an[0] in bns or bn[0] in ans)             # first (house) number shared
            longa = {x for x in ans if len(x) >= 5}
            longb = {x for x in bns if len(x) >= 5}
            out[i, 17] = -1.0 if not (longa and longb) else float(bool(longa & longb))  # zip/PIN
        else:
            out[i, 14] = out[i, 15] = out[i, 16] = out[i, 17] = -1.0
        sa, sb = a_state[i], b_state[i]
        out[i, 18] = -1.0 if not (sa and sb) else float(sa == sb)        # state agreement
        out[i, 19] = float(not b_addr[i])                               # target address empty
        if an and bn:
            # house numbers are perturbed in true matches (39 ~ 3941, 254 ~ 25, 158 ~ 159)
            out[i, 20] = fuzz.ratio(an[0], bn[0])
            best, contain = 0.0, 0.0
            for x in an[:6]:
                for y in bn[:6]:
                    r = fuzz.ratio(x, y)
                    if r > best:
                        best = r
                    if len(x) >= 2 and len(y) >= 2 and (x in y or y in x):
                        contain = 1.0
            out[i, 21] = best
            out[i, 22] = contain
        # acronym of the S1 name inside the target name ("bombay opinion" -> "bo..")
        if len(ac) >= 2:
            acr = "".join(t[0] for t in ac)
            bcat = "".join(bc)
            out[i, 23] = float(bcat.startswith(acr) or acr == bcat)
        if len(bc) >= 2:
            acr = "".join(t[0] for t in bc)
            out[i, 24] = float("".join(ac).startswith(acr) or acr == "".join(ac))
        out[i, 25] = float(any(len(t) >= 7 and t.isdigit() for t in bn_full))  # phone-like token
    return out


PY_FEATS = ["n_jacc", "n_cov_a", "n_cov_b", "n_phon_jacc", "n_first_eq", "n_bag_eq",
            "legal_agree", "legal_conflict", "n_ntok_a", "n_ntok_b", "a_cov_b", "a_cov_a",
            "a_ntok_a", "a_ntok_b", "num_jacc", "num_unseen_b", "num_house_eq", "zip_agree",
            "state_agree", "a_empty_b", "num_first_ratio", "num_best_ratio", "num_contain",
            "acr_a_in_b", "acr_b_in_a", "b_phone_tok"]


def _cp(scorer, a, b, **kw) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def pair_features(A: dict, B: dict, executor=None, chunk: int = 25_000) -> pd.DataFrame:
    """A, B: dicts of aligned lists (one entry per pair) with keys REC_COLS."""
    n = len(A["core"])
    A = dict(A)
    B = dict(B)
    A["core"] = [strip_phone(c) for c in A["core"]]
    B["core"] = [strip_phone(c) for c in B["core"]]
    f = {}
    f["n_ratio"] = _cp(fuzz.ratio, A["core"], B["core"])
    f["n_partial"] = _cp(fuzz.partial_ratio, A["core"], B["core"])
    f["n_tsort"] = _cp(fuzz.token_sort_ratio, A["core"], B["core"])
    f["n_tset"] = _cp(fuzz.token_set_ratio, A["core"], B["core"])
    f["n_jw"] = _cp(JaroWinkler.normalized_similarity, A["core"], B["core"])
    f["n_full_ratio"] = _cp(fuzz.ratio, A["name"], B["name"])
    f["n_wratio"] = _cp(fuzz.WRatio, A["name"], B["name"])
    a_cat = [c.replace(" ", "") for c in A["core"]]
    b_cat = [c.replace(" ", "") for c in B["core"]]
    f["n_concat_ratio"] = _cp(fuzz.ratio, a_cat, b_cat)
    f["n_concat_partial"] = _cp(fuzz.partial_ratio, a_cat, b_cat)
    has_alt = np.array([bool(x) for x in B["alt"]])
    alt = _cp(fuzz.token_set_ratio, A["core"], [x or "" for x in B["alt"]])
    f["n_alt_tset"] = np.where(has_alt, alt, -1).astype(np.float32)
    fa = [_flat_addr(x) for x in A["addr"]]
    fb = [_flat_addr(x) for x in B["addr"]]
    b_empty = np.array([not x for x in fb])
    for nm, sc in [("a_ratio", fuzz.ratio), ("a_tset", fuzz.token_set_ratio),
                   ("a_tsort", fuzz.token_sort_ratio), ("a_partial", fuzz.partial_ratio),
                   ("a_ptset", fuzz.partial_token_set_ratio)]:
        v = _cp(sc, fa, fb)
        f[nm] = np.where(b_empty, -1, v).astype(np.float32)
    f["b_script"] = np.array([_SCRIPT.get(s, 0) for s in B["script"]], dtype=np.float32)
    f["b_is_dom"] = np.asarray(B["is_dom"], dtype=np.float32)
    f["b_has_alt"] = has_alt.astype(np.float32)
    df = pd.DataFrame(f)

    jobs = []
    for c0 in range(0, n, chunk):
        sl = slice(c0, c0 + chunk)
        jobs.append((A["core"][sl], B["core"][sl], A["name"][sl], B["name"][sl],
                     A["addr"][sl], B["addr"][sl], A["state"][sl], B["state"][sl],
                     A["nums"][sl], B["nums"][sl]))
    if executor is not None and len(jobs) > 1:
        mats = list(executor.map(_py_features, jobs))
    else:
        mats = [_py_features(j) for j in jobs]
    py = np.vstack(mats) if mats else np.zeros((0, len(PY_FEATS)), np.float32)
    for j, nm in enumerate(PY_FEATS):
        df[nm] = py[:, j]
    return df


# ---------------------------------------------------------------- context
def group_top2(gid: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-element (group max, group second max or -1, rank within group [0=best])
    for dense group ids ``gid``."""
    order = np.lexsort((-v, gid))
    g = gid[order]
    first = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    size = np.diff(np.r_[first, len(g)])
    top1 = v[order[first]]
    top2 = np.where(size > 1, v[order[np.minimum(first + 1, len(g) - 1)]], -1.0)
    grp_of_sorted = np.repeat(np.arange(len(first)), size)
    rank_sorted = np.arange(len(g)) - first[grp_of_sorted]
    rank = np.empty(len(v), dtype=np.int64)
    rank[order] = rank_sorted
    gidx = np.empty(len(v), dtype=np.int64)
    gidx[order] = grp_of_sorted
    return top1[gidx], top2[gidx], rank


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """S1-side competition features.  ``df`` must hold *all* candidates of the
    S1 entities it contains (target-side features are computed separately on
    the full candidate set, see ``training.target_side_features``)."""
    df["s_total"] = (df["s_name"] + df["s_addr"] + df["s_cross"]).astype(np.float32)
    quick = (df["n_tset"].to_numpy() + np.maximum(df["a_ptset"].to_numpy(), 0)) / 2
    df["quick"] = quick.astype(np.float32)
    s1 = df["s1_row"].to_numpy().astype(np.int64)
    g_s1src = pd.factorize(s1 * 4 + df["src"].to_numpy())[0]
    g_s1 = pd.factorize(s1)[0]
    df["c_n_cands"] = np.bincount(g_s1src)[g_s1src].astype(np.float32)
    for col in ["s_total", "n_tset", "quick", "a_ptset", "n_ratio"]:
        v = df[col].to_numpy().astype(np.float64)
        m1, m2, rk = group_top2(g_s1src, v)
        df[f"c_{col}_gap"] = (m1 - v).astype(np.float32)
        df[f"c_{col}_margin"] = np.where(rk == 0, v - m2, v - m1).astype(np.float32)
        df[f"c_{col}_rank"] = rk.astype(np.float32)
    v = df["quick"].to_numpy().astype(np.float64)
    m1, _, rk = group_top2(g_s1, v)
    df["c_quick_rank_all"] = rk.astype(np.float32)
    df["c_quick_gap_all"] = (m1 - v).astype(np.float32)
    return df
