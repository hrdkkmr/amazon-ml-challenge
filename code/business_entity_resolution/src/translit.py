"""Indic-script -> Latin transliteration.

Two layers, both built only from the challenge data / hand-written rules:

1. ``rule_translit``: a deterministic character-level transliterator for all
   Brahmi-derived Unicode blocks (Devanagari, Bengali, Gurmukhi, Gujarati,
   Oriya, Tamil, Telugu, Kannada, Malayalam).  These blocks share the ISCII
   layout, so one offset table covers all of them.
2. ``TranslitDict``: a token dictionary (indic token -> latin token) learned
   from *training* ground-truth pairs whose Latin S1 name and non-Latin target
   name have the same number of tokens (positional alignment + voting).
   Unknown tokens fall back to layer 1.

``phonetic`` maps any Latin token to a coarse consonant skeleton so that rule
transliterations ("baalaajii") and English spellings ("balaji") collide.
"""
from __future__ import annotations

import pickle
import re
from collections import Counter, defaultdict

_BLOCKS = range(0x0900, 0x0E00, 0x80)

_VOWELS = {0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu",
           0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai",
           0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li"}
_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch",
         0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny", 0x1F: "t", 0x20: "th",
         0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d",
         0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
         0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l",
         0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s",
         0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "d",
         0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_MATRAS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri",
           0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o",
           0x4A: "o", 0x4B: "o", 0x4C: "au", 0x57: "au", 0x62: "li", 0x63: "li"}
_VIRAMA = 0x4D
_NASAL = {0x01: "n", 0x02: "n", 0x03: "h"}
_SKIP = {0x3C, 0x00, 0x70, 0x71, 0x64, 0x65}   # nukta, misc signs, dandas


def _is_indic(cp: int) -> bool:
    return 0x0900 <= cp < 0x0E00


def rule_translit(tok: str) -> str:
    """Character-level transliteration of one token (non-Indic chars kept)."""
    out = []
    pending_a = False           # inherent vowel of the previous consonant
    for ch in tok:
        cp = ord(ch)
        if not _is_indic(cp):
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(ch)
            continue
        off = (cp - 0x0900) % 0x80
        if 0x66 <= off <= 0x6F:                      # digits
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(str(off - 0x66))
        elif off in _CONS:
            if pending_a:
                out.append("a")
            out.append(_CONS[off])
            pending_a = True
        elif off in _MATRAS:
            out.append(_MATRAS[off])
            pending_a = False
        elif off == _VIRAMA:
            pending_a = False
        elif off in _VOWELS:
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(_VOWELS[off])
        elif off in _NASAL:
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(_NASAL[off])
        # else: skip (nukta, avagraha, dandas ...)
    # schwa deletion: final inherent 'a' is dropped (pending_a stays unused)
    return "".join(out)


_PHON_SUBS = [
    (re.compile(r"ph"), "f"), (re.compile(r"w"), "v"), (re.compile(r"ck"), "k"),
    (re.compile(r"q"), "k"), (re.compile(r"c(?=[eiy])"), "s"), (re.compile(r"c"), "k"),
    (re.compile(r"z"), "j"), (re.compile(r"x"), "ks"), (re.compile(r"g(?=[ei])"), "j"),
    (re.compile(r"([bcdfgjklmnprstv])h"), r"\1"), (re.compile(r"sh"), "s"),
]
_VOWEL_RE = re.compile(r"[aeiouy]+")
_REPEAT = re.compile(r"(.)\1+")


def phonetic(tok: str) -> str:
    """Coarse consonant skeleton of a Latin token (first letter kept)."""
    if not tok:
        return ""
    t = tok.lower()
    for rx, rep in _PHON_SUBS:
        t = rx.sub(rep, t)
    head, rest = t[0], t[1:]
    rest = _VOWEL_RE.sub("", rest)
    if head in "aeiouy":
        head = "a"
    return _REPEAT.sub(r"\1", head + rest)


def has_indic(text: str) -> bool:
    return any(_is_indic(ord(c)) for c in text)


class TranslitDict:
    """Learned indic-token -> latin-token dictionary with rule fallback."""

    def __init__(self, mapping: dict | None = None):
        self.mapping = mapping or {}

    # ------------------------------------------------------------ learning
    @classmethod
    def learn(cls, pairs, min_count: int = 2, min_share: float = 0.5) -> "TranslitDict":
        """``pairs``: iterable of (latin_name_tokens, target_name_tokens)."""
        votes: dict[str, Counter] = defaultdict(Counter)
        for lat, tgt in pairs:
            if len(lat) != len(tgt):
                continue
            for a, b in zip(lat, tgt):
                if has_indic(b) and not has_indic(a):
                    votes[b][a] += 1
        mapping = {}
        for tok, cnt in votes.items():
            best, n = cnt.most_common(1)[0]
            if n >= min_count and n / sum(cnt.values()) >= min_share:
                mapping[tok] = best
        return cls(mapping)

    # --------------------------------------------------------------- apply
    def token(self, tok: str) -> str:
        if not has_indic(tok):
            return tok
        m = self.mapping.get(tok)
        return m if m is not None else rule_translit(tok)

    def text(self, text: str) -> str:
        if not text or not has_indic(text):
            return text
        return " ".join(self.token(t) for t in text.split())

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.mapping, f)

    @classmethod
    def load(cls, path) -> "TranslitDict":
        with open(path, "rb") as f:
            return cls(pickle.load(f))
