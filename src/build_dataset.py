# %% Imports + config
from pathlib import Path
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
BENIGN_TARGET = 1_500_000

FAMILY = {
    "Benign": "Benign",
    "SSH-Bruteforce": "BruteForce",
    "FTP-BruteForce": "BruteForce",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "DDOS attack-HOIC": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS",
    "DoS attacks-GoldenEye": "DoS",
    "DoS attacks-Slowloris": "DoS",
    "DoS attacks-Hulk": "DoS",
    "DoS attacks-SlowHTTPTest": "DoS",
    "Bot": "Bot",
    "Brute Force -Web": "Web",
    "Brute Force -XSS": "Web",
    "SQL Injection": "Web",
    "Infilteration": "Infiltration",
}

# %% Pass 1: count benign per day, check columns match
files = sorted(RAW_DIR.glob("*.parquet"))
benign_counts, columns = {}, None
for path in files:
    d = pd.read_parquet(path, columns=["Label"])
    benign_counts[path.name] = int((d["Label"].astype(str) == "Benign").sum())
    cols = pd.read_parquet(path).columns.tolist() if columns is None else None
    if columns is None:
        columns = cols

total_benign = sum(benign_counts.values())
frac = min(1.0, BENIGN_TARGET / total_benign)
print(f"Total benign: {total_benign:,}  -> sampling fraction {frac:.4f}")

# %% Pass 2: load, map, downsample, combine
parts = []
for path in files:
    d = pd.read_parquet(path)
    if d.columns.tolist() != columns:
        missing = set(columns) ^ set(d.columns)
        raise SystemExit(f"Column mismatch in {path.name}: {missing}")

    d["Label"] = d["Label"].astype(str)
    unknown = set(d["Label"].unique()) - set(FAMILY)
    if unknown:
        raise SystemExit(f"Unmapped labels in {path.name}: {unknown}")

    d["family"] = d["Label"].map(FAMILY)
    d["day"] = path.name.split("_")[0]

    benign = d[d["family"] == "Benign"].sample(frac=frac, random_state=SEED)
    attacks = d[d["family"] != "Benign"]
    parts.append(pd.concat([benign, attacks]))
    print(f"{path.name.split('_')[0]:<30} benign kept={len(benign):>8,}  attacks={len(attacks):>8,}")

data = pd.concat(parts, ignore_index=True)

# %% Summary + save
print("\nRows per family:")
print(data["family"].value_counts().to_string())
print("\nRows per original label:")
print(data["Label"].value_counts().to_string())
print(f"\nTotal rows: {len(data):,}   Columns: {data.shape[1]}")

out_path = OUT_DIR / "all_days.parquet"
data.to_parquet(out_path, index=False)
print("Saved to", out_path)