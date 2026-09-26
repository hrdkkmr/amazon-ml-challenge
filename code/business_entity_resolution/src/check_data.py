from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

train_dir = ROOT / "dataset" / "train"
test_dir = ROOT / "dataset" / "test"

files = [
    train_dir / "train_source1.tsv",
    train_dir / "train_source2.tsv",
    train_dir / "train_source3.tsv",
    train_dir / "train_ground_truth.tsv",
    test_dir / "test_source1.tsv",
    test_dir / "test_source2.tsv",
    test_dir / "test_source3.tsv",
]

for file in files:
    print("\n" + "=" * 70)
    print(file)

    if not file.exists():
        print("❌ FILE NOT FOUND")
        continue

    df = pd.read_csv(file, sep="\t")

    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("\nFirst 3 rows:")
    print(df.head(3))