"""Deterministic text normalization for business names and addresses.

Every function here is pure (same input -> same output) and uses only
hand-written rules; no external data or services are consulted.

Outputs per record (see ``normalize_record``):
    name_norm   lower-cased, accent-free, punctuation-free name with canonical
                legal suffixes (e.g. "private limited" -> "pvt ltd")
    name_core   name_norm without legal suffixes / honorifics / filler words
    name_alt    the part after "d/b/a", "formerly", "aka" (else "")
    name_dom    1 if the name looks like a web domain (root kept in name_norm)
    name_script "L" latin, "I" indic (or other non-latin), "M" mixed
    addr_norm   normalized address: abbreviations expanded to one canonical
                form, state names mapped to codes, leading zeros stripped
    addr_nums   space-joined numeric tokens of the address (house / unit / PIN)
    state       canonical state code if detected ("" otherwise)
"""
from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------- characters
def _build_latin_fold() -> dict:
    """Map accented Latin characters (U+0080..U+024F) to their ASCII base."""
    table = {}
    for cp in range(0x80, 0x250):
        ch = chr(cp)
        base = "".join(c for c in unicodedata.normalize("NFKD", ch)
                       if not unicodedata.combining(c))
        if base and base.isascii() and base != ch:
            table[cp] = base.lower()
    table.update({ord("ß"): "ss", ord("æ"): "ae", ord("œ"): "oe", ord("ø"): "o",
                  ord("đ"): "d", ord("ł"): "l", ord("’"): "'", ord("‘"): "'",
                  ord("–"): "-", ord("—"): "-"})
    return table


_LATIN_FOLD = _build_latin_fold()
_NON_WORD = re.compile(r"[^\wऀ-ॣ०-෿]+", re.UNICODE)  # keep Indic vowel signs
_APOS = re.compile(r"['`´]")
_AMP = re.compile(r"\s*&\s*|\s*\+\s*")
_ALNUM_HYPHEN = re.compile(r"(?<=[a-z0-9])[-/.](?=[0-9])|(?<=[0-9])[-.](?=[a-z])")
_LEADING_ZERO = re.compile(r"\b0+(?=\d)")
_DIGIT = re.compile(r"\d")
_NUMTOK = re.compile(r"\d+")
_WS = re.compile(r"\s+")


def fold(text: str) -> str:
    """NFKC + lowercase + strip Latin diacritics (non-Latin scripts untouched)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    return text.translate(_LATIN_FOLD)


def script_of(text: str) -> str:
    """'L' if only ASCII letters, 'I' if only non-Latin letters, 'M' if both."""
    has_lat = has_non = False
    for ch in text:
        if ch.isalpha():
            if ch.isascii():
                has_lat = True
            else:
                has_non = True
            if has_lat and has_non:
                return "M"
    return "I" if has_non and not has_lat else "L"


# --------------------------------------------------------------------- names
# canonical legal forms; the key is a regex over the tokenized name
_LEGAL_PATTERNS = [
    (r"\bprivate\s+limited\b|\bpvt\.?\s*ltd\b|\bpvtltd\b", "pvt ltd"),
    (r"\bl\s*l\s*c\b", "llc"),
    (r"\bl\s*l\s*p\b", "llp"),
    (r"\bp\s*l\s*l\s*c\b", "pllc"),
    (r"\bincorporated\b", "inc"),
    (r"\bcorporation\b", "corp"),
    (r"\bcompany\b", "co"),
    (r"\blimited\b", "ltd"),
    (r"\bprivate\b|\bprvt\b", "pvt"),
    (r"\bsoci[eé]t[eé]\b", "ste"),
]
_LEGAL_RE = [(re.compile(p), r) for p, r in _LEGAL_PATTERNS]
LEGAL_TOKENS = {
    "pvt", "ltd", "llc", "llp", "pllc", "inc", "corp", "co", "pc", "plc", "lp",
    "sarl", "sas", "sa", "eurl", "sasu", "ste", "gmbh", "opc", "pa", "lc",
}
HONORIFICS = {"mr", "mrs", "ms", "m", "s", "shri", "sri", "shree", "smt", "dr",
              "messrs", "the", "m/s"}
FILLER = {"services", "service", "center", "centre", "group", "and", "of", "enterprises",
          "enterprise", "solutions", "company", "the"}
_ALIAS_RE = re.compile(r"\b(?:d\s*/?\s*b\s*/?\s*a|dba|formerly(?:\s+known\s+as)?|f\s*/\s*k\s*/\s*a|fka|a\s*/\s*k\s*/\s*a|aka|trading\s+as|t\s*/\s*a)\b[:\s]*")
_DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)*)\.(?:com|net|org|in|co\.in|co|biz|info|us|fr|io)$")
_MS_RE = re.compile(r"\bm\s*/\s*s\b")


def _canon_legal(s: str) -> str:
    for rx, rep in _LEGAL_RE:
        s = rx.sub(rep, s)
    return s


def normalize_name(raw: str) -> tuple[str, str, str, int, str]:
    """Return (name_norm, name_core, name_alt, is_domain, script)."""
    s = fold(raw or "").strip()
    script = script_of(s)
    is_dom = 0
    m = _DOMAIN_RE.match(s.replace(" ", ""))
    if m and " " not in s.strip():
        s = m.group(1).replace(".", " ").replace("-", " ")
        is_dom = 1
    s = _MS_RE.sub(" ", s)
    s = _APOS.sub("", s)
    s = _AMP.sub(" and ", s)
    alt = ""
    am = _ALIAS_RE.search(s)
    if am:
        alt = s[am.end():]
        s = s[:am.start()] + " " + alt   # keep both sides in the main form
    s = _WS.sub(" ", _NON_WORD.sub(" ", s)).strip()
    s = _canon_legal(s)
    toks = s.split()
    # drop leading honorifics, keep order otherwise
    while toks and toks[0] in HONORIFICS:
        toks = toks[1:]
    norm = " ".join(toks)
    core = " ".join(t for t in toks if t not in LEGAL_TOKENS and t not in FILLER
                    and t not in HONORIFICS)
    if not core:
        core = norm
    if alt:
        alt = _canon_legal(_WS.sub(" ", _NON_WORD.sub(" ", alt)).strip())
        alt = " ".join(t for t in alt.split() if t not in LEGAL_TOKENS and t not in FILLER)
    return norm, core, alt, is_dom, script


# ----------------------------------------------------------------- addresses
_STREET_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "avn": "avenue", "dr": "drive", "drv": "drive", "blvd": "boulevard", "ln": "lane",
    "ct": "court", "crt": "court", "pl": "place", "hwy": "highway", "pkwy": "parkway",
    "pky": "parkway", "trl": "trail", "cir": "circle", "sq": "square", "ter": "terrace",
    "terr": "terrace", "tpke": "turnpike", "expy": "expressway", "fwy": "freeway",
    "cv": "cove", "pt": "point", "xing": "crossing", "aly": "alley", "plz": "plaza",
    "mt": "mount", "ft": "fort", "hts": "heights", "jct": "junction", "rte": "route",
    "rt": "route", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "apt": "unit", "apartment": "unit", "ste": "unit", "suite": "unit", "rm": "unit",
    "room": "unit", "bldg": "building", "bld": "building", "fl": "floor", "flr": "floor",
    "opp": "opposite", "nr": "near", "mg": "mahatma gandhi", "stn": "station",
    "marg": "road", "rasta": "road", "gali": "lane", "nagr": "nagar", "ngr": "nagar",
    "sec": "sector", "sect": "sector", "ph": "phase", "no": "", "number": "", "null": "",
    "none": "", "na": "", "nan": "", "c/o": "co", "dist": "district", "distt": "district",
    "tq": "taluka", "tal": "taluka", "po": "post", "ps": "police", "bd": "boulevard",
    "bis": "", "ter.": "", "chem": "chemin", "rte.": "route", "imp": "impasse",
    "all": "allee", "pl.": "place", "fbg": "faubourg", "qu": "quai", "sq.": "square",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "ts",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk", "jammu & kashmir": "jk",
    "ladakh": "la", "puducherry": "py", "pondicherry": "py", "chandigarh": "ch",
    "dadra and nagar haveli": "dn", "daman and diu": "dd", "lakshadweep": "ld",
    "andaman and nicobar islands": "an",
}
# alternative codes seen in Indian addresses
IN_CODE_ALIASES = {"ts": "ts", "tg": "ts", "or": "od", "ua": "uk", "ut": "uk",
                   "cg": "cg", "ct": "cg"}


def _phrase_re(mapping: dict) -> re.Pattern:
    keys = sorted(mapping, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")


_US_STATE_RE = _phrase_re(US_STATES)
_IN_STATE_RE = _phrase_re(IN_STATES)
_US_CODES = set(US_STATES.values())
_IN_CODES = set(IN_STATES.values()) | set(IN_CODE_ALIASES)


def normalize_address(raw: str, country: str) -> tuple[str, str, str]:
    """Return (addr_norm, addr_nums, state_code).

    Components keep their comma order (joined by ' , ') so that callers can
    build within-component bigrams; similarity features ignore the commas.
    """
    s = fold(raw or "")
    if not s:
        return "", "", ""
    s = _APOS.sub("", s)
    s = _AMP.sub(" and ", s)
    s = s.replace("#", " ")
    s = _ALNUM_HYPHEN.sub("", s)
    s = _LEADING_ZERO.sub("", s)
    c = (country or "").strip().lower()
    state = ""
    if c in ("us", "usa", "united states"):
        rx, codes, full = _US_STATE_RE, _US_CODES, US_STATES
    elif c == "india":
        rx, codes, full = _IN_STATE_RE, _IN_CODES, IN_STATES
    else:
        rx, codes, full = None, set(), {}
    comps = []
    for comp in s.split(","):
        comp = _WS.sub(" ", _NON_WORD.sub(" ", comp)).strip()
        if not comp:
            continue
        if rx is not None:
            m = rx.fullmatch(comp)
            if m:
                state = full[m.group(1)]
                comp = state
            elif comp in codes:
                state = IN_CODE_ALIASES.get(comp, comp) if c == "india" else comp
                comp = state
        toks = []
        for t in comp.split():
            r = _STREET_ABBR.get(t, t)
            if r:
                toks.append(r)
        if toks:
            comps.append(" ".join(toks))
    norm = " , ".join(comps)
    nums = " ".join(_NUMTOK.findall(norm))
    return norm, nums, state


def addr_tokens(addr_norm: str, state: str = "") -> list[str]:
    """Flat token list without commas and without the state code."""
    return [t for t in addr_norm.split() if t != "," and t != state]
