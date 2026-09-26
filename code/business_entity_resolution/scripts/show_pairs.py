"""Print normalized S1/target records for pairs in a parquet file (debug helper).
Usage: python scripts/show_pairs.py <pairs.parquet> [n] [split]"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from records import Enricher, load_records
sys.stdout.reconfigure(encoding="utf-8")
p = pd.read_parquet(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
split = sys.argv[3] if len(sys.argv) > 3 else "train"
p = p.sample(min(n, len(p)), random_state=1)
enr = Enricher("cache")
recs = {}
for s in (1, 2, 3):
    r = load_records("cache", split, s, enr)
    ids = set(p["s1"]) | set(p["tgt"])
    recs.update({e: (nm, a) for e, nm, a in zip(r.entity_id, r.name, r.addr) if e in ids})
for _, row in p.iterrows():
    extra = {k: row[k] for k in p.columns if k not in ("s1", "tgt")}
    print(row.s1, recs.get(row.s1), "\n   ", row.tgt, recs.get(row.tgt), extra)
