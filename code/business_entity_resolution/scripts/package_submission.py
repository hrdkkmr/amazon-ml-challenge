"""Assemble FINAL_SUBMISSION/ in the official layout and zip it.

FINAL_SUBMISSION/
├── output/matching_results.tsv, candidate_pairs.tsv
├── code/business_entity_resolution/   src/ configs/ scripts/ utils/ models/
│                                      README.md requirements.txt METHODOLOGY.md
└── Documentation_template.md          (filled-in methodology)

Usage: python scripts/package_submission.py [team_name]
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
team = sys.argv[1] if len(sys.argv) > 1 else "TEAM"
dst = ROOT / "FINAL_SUBMISSION"
code = dst / "code" / "business_entity_resolution"

if dst.exists():
    shutil.rmtree(dst)
(dst / "output").mkdir(parents=True)
code.mkdir(parents=True)

for f in ("matching_results.tsv", "candidate_pairs.tsv"):
    shutil.copy2(ROOT / "output" / f, dst / "output" / f)

ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
for d in ("src", "configs", "scripts", "utils"):
    shutil.copytree(ROOT / d, code / d, ignore=ignore)
(code / "models").mkdir()
for f in ("matcher.pkl", "filter.pkl", "decision.json"):
    shutil.copy2(ROOT / "models" / f, code / "models" / f)
for f in ("translit_dict.pkl", "state_map.pkl"):      # learned from training data
    shutil.copy2(ROOT / "cache" / f, code / "models" / f)
for f in ("README.md", "requirements.txt", "METHODOLOGY.md"):
    shutil.copy2(ROOT / f, code / f)
shutil.copy2(ROOT / "METHODOLOGY.md", dst / "Documentation_template.md")
for f in ("SUMMARY.md", "final_report.json", "experiments.jsonl", "model_runs.jsonl",
          "blocking_runs.jsonl", "data_profile.json"):
    if (ROOT / "experiments" / f).exists():
        (code / "experiments").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "experiments" / f, code / "experiments" / f)

# validate the packaged files with the official validator
r = subprocess.run([sys.executable, str(ROOT / "utils" / "validate_submission.py"),
                    "--matching", str(dst / "output" / "matching_results.tsv"),
                    "--candidate", str(dst / "output" / "candidate_pairs.tsv"),
                    "--test-dir", str(ROOT / "dataset" / "test")],
                   capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    sys.exit("validator failed - package not zipped")

zpath = ROOT / f"{team}_submission.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(dst.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(dst))
print(f"wrote {zpath} ({zpath.stat().st_size / 2**20:.0f} MB)")
