"""Submission validation: the official validator plus stricter own checks.

Own checks (streaming, stdlib only):
  * exact headers, 2 tab-separated columns, no quoting
  * every test S1 appears exactly once in both files (and nothing else)
  * ID lists: only S2-/S3- ids, no duplicates, ids exist in the test files
  * every matched id is also a candidate of the same S1 (hard failure here)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read_ids(path: Path) -> set:
    with open(path, encoding="utf-8") as f:
        next(f)
        return {line.split("\t", 1)[0] for line in f if line.strip()}


def _iter_rows(path: Path, header: list[str], errors: list):
    """Yield (s1, id_list) per data row, recording format problems."""
    with open(path, encoding="utf-8", newline="") as f:
        head = f.readline().rstrip("\n").split("\t")
        if head != header:
            errors.append(f"{path.name}: bad header {head}")
        for ln, line in enumerate(f, 2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                errors.append(f"{path.name}:{ln}: expected 2 columns, got {len(parts)}")
                continue
            s1, ids = parts
            lst = ids.split(",") if ids else []
            if any(not x for x in lst):
                errors.append(f"{path.name}:{ln}: empty id in list")
            if len(set(lst)) != len(lst):
                errors.append(f"{path.name}:{ln}: duplicate ids in list")
            yield s1, lst


def own_checks(matching: Path, candidates: Path, test_dir: Path) -> list[str]:
    """Stream both files in lockstep (they are written in the same S1 order)."""
    errors: list[str] = []
    s1_all = _read_ids(test_dir / "test_source1.tsv")
    targets = _read_ids(test_dir / "test_source2.tsv") | _read_ids(test_dir / "test_source3.tsv")
    seen: set = set()
    n_m = n_c = n_empty = not_cand = bad = 0
    it_c = _iter_rows(candidates, ["source1_entity_id", "candidate_entity_ids"], errors)
    for s1, mlist in _iter_rows(matching, ["source1_entity_id", "matched_entity_ids"], errors):
        c_row = next(it_c, None)
        if c_row is None or c_row[0] != s1:
            errors.append(f"candidate file out of sync at S1 {s1}")
            break
        clist = c_row[1]
        if s1 in seen:
            errors.append(f"duplicate S1 row {s1}")
        seen.add(s1)
        cset = set(clist)
        not_cand += sum(1 for x in mlist if x not in cset)
        for x in mlist + clist:
            if not (x.startswith("S2-") or x.startswith("S3-")) or x not in targets:
                bad += 1
        n_m += len(mlist)
        n_c += len(clist)
        n_empty += not mlist
    if next(it_c, None) is not None:
        errors.append("candidate file has extra rows")
    if seen != s1_all:
        errors.append(f"S1 set differs from test_source1 (missing {len(s1_all - seen)}, "
                      f"extra {len(seen - s1_all)})")
    if bad:
        errors.append(f"{bad} invalid/unknown ids")
    if not_cand:
        errors.append(f"{not_cand} matched ids are not in candidate_pairs for their S1")
    print(f"  own checks: {len(seen):,} S1 rows, {n_m:,} matches, {n_c:,} candidates "
          f"({n_c / max(len(seen), 1):.2f}/S1), {n_empty:,} S1 with empty match list")
    return errors


def official(matching: Path, candidates: Path, test_dir: Path) -> bool:
    cmd = [sys.executable, str(ROOT / "utils" / "validate_submission.py"),
           "--matching", str(matching), "--candidate", str(candidates),
           "--test-dir", str(test_dir), "--check-ids"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("  official validator output:\n    " + "\n    ".join(r.stdout.strip().splitlines()))
    if r.stderr.strip():
        print(r.stderr)
    return r.returncode == 0


def validate_outputs(cfg) -> bool:
    out = Path(cfg.output_dir)
    test_dir = Path(cfg.data_dir) / "test"
    matching, candidates = out / "matching_results.tsv", out / "candidate_pairs.tsv"
    errors = own_checks(matching, candidates, test_dir)
    for e in errors:
        print("  ERROR:", e)
    ok_official = official(matching, candidates, test_dir)
    ok = ok_official and not errors
    print("  VALIDATION", "PASSED" if ok else "FAILED")
    return ok
