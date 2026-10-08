# Peek at NF-UQ-NIDS-v2: file size, columns, dtypes, label columns (reads only 2,000 rows)
# Usage: python -m src.nf_peek
from pathlib import Path
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "data" / "nf_uq_v2"
files = [p for p in SRC_DIR.rglob("*") if p.suffix.lower() in (".csv", ".parquet")]
if not files:
    raise SystemExit(f"No CSV/Parquet found under {SRC_DIR}")
for p in files:
    print(f"{p}   {p.stat().st_size / 1e9:.2f} GB")

src = max(files, key=lambda p: p.stat().st_size)
print("\nPeeking:", src)
df = pd.read_parquet(src).head(2000) if src.suffix == ".parquet" else pd.read_csv(src, nrows=2000)
print(f"\nColumns ({len(df.columns)}):")
print(df.dtypes.to_string())
for col in ["Label", "Attack", "Dataset"]:
    if col in df.columns:
        print(f"\n{col} values in first 2,000 rows:")
        print(df[col].value_counts().to_string())
print("\nFirst 3 rows:")
print(df.head(3).T.to_string())