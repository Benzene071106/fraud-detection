# Prepare CIC-IDS2017 Thursday web-attack day as extra TRAINING data
# Usage: python -m src.prep_2017_web

# %% Imports + config
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

SRC = ROOT / "data" / "raw_2017" / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
REF = ROOT / "data" / "processed" / "all_days.parquet"
OUT = ROOT / "data" / "processed" / "web2017.parquet"

RENAME = {
    "Total Length of Fwd Packets": "Fwd Packets Length Total",
    "Total Length of Bwd Packets": "Bwd Packets Length Total",
    "Min Packet Length": "Packet Length Min",
    "Max Packet Length": "Packet Length Max",
    "Average Packet Size": "Avg Packet Size",
    "Init_Win_bytes_forward": "Init Fwd Win Bytes",
    "Init_Win_bytes_backward": "Init Bwd Win Bytes",
    "act_data_pkt_fwd": "Fwd Act Data Packets",
    "min_seg_size_forward": "Fwd Seg Size Min",
}

# %% Load + rename
d = pd.read_csv(SRC, encoding="latin1", low_memory=False)
d.columns = d.columns.str.strip()
d = d.rename(columns=RENAME).drop(columns=["Destination Port", "Fwd Header Length.1"], errors="ignore")
print(f"Loaded {len(d):,} rows")

# %% Verify columns against 2018
ref_cols = [c for c in pq.read_schema(REF).names if c not in ("Label", "family", "day")]
missing = [c for c in ref_cols if c not in d.columns]
extra = [c for c in d.columns if c not in ref_cols + ["Label"]]
print("Missing vs 2018:", missing, "  (expected: ['Protocol'])")
print("Extra in 2017  :", extra, "  (expected: [])")
if set(missing) - {"Protocol"} or extra:
    raise SystemExit("Column mapping problem - send me the two lists above.")

num = [c for c in ref_cols if c != "Protocol"]

# %% Clean
d[num] = d[num].apply(pd.to_numeric, errors="coerce")
before = len(d)
d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=num)
d = d.drop_duplicates(subset=num + ["Label"])
print(f"After removing inf/NaN/duplicates: {len(d):,} (dropped {before - len(d):,})")

lab = d["Label"].astype(str).str.strip()
d["Label"] = np.select(
    [lab.eq("BENIGN"), lab.str.contains("Brute Force"), lab.str.contains("XSS"), lab.str.contains("Sql")],
    ["Benign", "Brute Force -Web", "Brute Force -XSS", "SQL Injection"],
    default="Other",
)
d = d[d["Label"] != "Other"]
d["family"] = np.where(d["Label"] == "Benign", "Benign", "Web")
print("\nLabels:")
print(d["Label"].value_counts().to_string())

# %% Twin check: drop Web rows that are exact copies of a Benign row
h = pd.util.hash_pandas_object(d[num].astype("float64"), index=False).values
is_web = (d["family"] == "Web").values
benign_h = np.unique(h[~is_web])
twin = is_web & np.isin(h, benign_h)
print(f"\nWeb rows with an exact Benign twin: {twin.sum():,} of {is_web.sum():,} "
      f"({twin.sum() / is_web.sum():.1%}) -> removed")
d = d[~twin]

# %% Save
d[num] = d[num].astype("float32")
d["day"] = "2017-Thu-WebAttacks"
d[num + ["Label", "family", "day"]].to_parquet(OUT, index=False)
print(f"\nSaved {len(d):,} rows ({(d['family'] == 'Web').sum():,} Web, "
      f"{(d['family'] == 'Benign').sum():,} Benign) to {OUT}")