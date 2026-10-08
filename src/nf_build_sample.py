# NF-UQ-NIDS-v2: stream the 13.7 GB CSV, normalize labels, sample per (Dataset, Attack), clean, save Parquet
# Usage: python -m src.nf_build_sample

# %% Imports + config
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

SRC = ROOT / "data" / "nf_uq_v2" / "NF-UQ-NIDS-v2.csv"
OUT = ROOT / "data" / "processed" / "nfuq_sample.parquet"
META = ROOT / "models" / "nfuq_sample_config.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
SEED = 42
ATTACK_CAP = 300_000          # per (source dataset, attack type)
BENIGN_CAP = 1_000_000        # per source dataset

IP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
TEXT_COLS = ["Attack", "Dataset"]
PORTS = ["L4_SRC_PORT", "L4_DST_PORT"]
NF_FINGERPRINTS = ["MIN_TTL", "MAX_TTL", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"]

CANON = {
    "benign": "Benign", "ddos": "DDoS", "dos": "DoS", "reconnaissance": "Reconnaissance",
    "scanning": "Scanning", "injection": "Injection", "xss": "XSS", "password": "Password",
    "brute force": "Brute Force", "bot": "Bot", "infilteration": "Infiltration",
    "exploits": "Exploits", "fuzzers": "Fuzzers", "backdoor": "Backdoor", "generic": "Generic",
    "analysis": "Analysis", "theft": "Theft", "shellcode": "Shellcode", "mitm": "MITM",
    "worms": "Worms", "ransomware": "Ransomware",
}

all_cols = pacsv.open_csv(SRC).schema.names
NUM_COLS = [c for c in all_cols if c not in IP_COLS + TEXT_COLS]
types = {c: pa.float64() for c in NUM_COLS}
types.update({c: pa.string() for c in IP_COLS + TEXT_COLS})
read_opts = pacsv.ReadOptions(block_size=64 << 20)

def stream(columns):
    return pacsv.open_csv(SRC, read_options=read_opts,
                          convert_options=pacsv.ConvertOptions(include_columns=columns, column_types=types))

# %% Pass 1: count rows per (Dataset, Attack)
t0 = time.time()
counts = {}
for batch in stream(TEXT_COLS):
    d = batch.to_pandas()
    for k, v in (d["Dataset"] + "|" + d["Attack"]).value_counts().items():
        counts[k] = counts.get(k, 0) + v
total = sum(counts.values())
print(f"Pass 1 done: {total:,} rows in {(time.time() - t0) / 60:.1f} min")

raw_attacks = sorted({k.split("|")[1] for k in counts})
unknown = [a for a in raw_attacks if a.strip().lower() not in CANON]
print("Raw attack names:", raw_attacks)
if unknown:
    print("WARNING - unmapped attack names (kept as-is):", unknown)

p_map = {}
for k, n in counts.items():
    attack = k.split("|")[1].strip().lower()
    cap = BENIGN_CAP if attack == "benign" else ATTACK_CAP
    p_map[k] = min(1.0, cap / n)

# %% Pass 2: stream all columns except IPs, sample per group (kept as float64 here)
t0 = time.time()
rng = np.random.default_rng(SEED)
parts = []
for batch in stream(NUM_COLS + TEXT_COLS):
    d = batch.to_pandas()
    p = (d["Dataset"] + "|" + d["Attack"]).map(p_map).values
    parts.append(d[rng.random(len(d)) < p].copy())
data = pd.concat(parts, ignore_index=True)
del parts
print(f"Pass 2 done: sampled {len(data):,} rows in {(time.time() - t0) / 60:.1f} min")

# %% Values beyond float32 range = corrupted entries in the source CSV
F32_MAX = np.finfo(np.float32).max
too_big = data[NUM_COLS].abs() > F32_MAX
bad = too_big.any(axis=1).values
print(f"\nRows with values beyond float32 range: {bad.sum():,} ({bad.mean():.3%})")
print("  by column:", too_big.sum()[lambda s: s > 0].to_dict())
print("  by class:", data.loc[bad, "Attack"].value_counts().to_dict())
print("  by source:", data.loc[bad, "Dataset"].value_counts().to_dict())
data = data[~bad].copy()
data[NUM_COLS] = data[NUM_COLS].astype("float32")

# %% Normalize labels + clean
data["Attack"] = data["Attack"].str.strip().map(lambda a: CANON.get(a.lower(), a))
FEATURES = [c for c in NUM_COLS if c != "Label"]

wrap = (data["FLOW_DURATION_MILLISECONDS"] > 4.2e9).mean()
print(f"\nFLOW_DURATION wraparound (> 4.2e9 ms): {wrap:.2%} of rows")

before = len(data)
data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
print(f"Removed inf/NaN rows: {before - len(data):,}")

before = len(data)
data = data.drop_duplicates(subset=FEATURES + ["Attack", "Dataset"])
print(f"Removed exact duplicates: {before - len(data):,} ({(before - len(data)) / before:.1%})")

h = pd.util.hash_pandas_object(data[FEATURES], index=False).values
conflict = pd.Series(data["Attack"].values).groupby(h).transform("nunique").values > 1
print(f"Rows whose exact features appear with >1 attack label: {conflict.sum():,} ({conflict.mean():.2%})")
print("  by class:", pd.Series(data["Attack"].values[conflict]).value_counts().to_dict())

# %% Summary + save
print("\nRows per class:")
print(data["Attack"].value_counts().to_string())
print("\nRows per source dataset x class:")
print(pd.crosstab(data["Attack"], data["Dataset"]).to_string())

data.to_parquet(OUT, index=False)
with open(META, "w") as f:
    json.dump({"source": SRC.name, "rows": int(len(data)), "attack_cap": ATTACK_CAP,
               "benign_cap": BENIGN_CAP, "features": FEATURES, "ports": PORTS,
               "nf_fingerprints": NF_FINGERPRINTS,
               "classes": sorted(data["Attack"].unique().tolist())}, f, indent=2)
print(f"\nSaved {len(data):,} rows -> {OUT}")
print(f"Saved config -> {META}")