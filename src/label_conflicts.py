# Step A: how many Infiltration / Web rows have a benign "twin"?
# %% Imports + config
from pathlib import Path
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

RAW_DIR = ROOT / "data" / "raw"
FILES = {
    "Infil1": "Infil1-Wednesday-28-02-2018_TrafficForML_CICFlowMeter.parquet",
    "Infil2": "Infil2-Thursday-01-03-2018_TrafficForML_CICFlowMeter.parquet",
    "Web1": "Web1-Thursday-22-02-2018_TrafficForML_CICFlowMeter.parquet",
    "Web2": "Web2-Friday-23-02-2018_TrafficForML_CICFlowMeter.parquet",
}
COARSE = ["Protocol", "Total Fwd Packets", "Total Backward Packets",
          "Fwd Packets Length Total", "Bwd Packets Length Total"]

# %% Conflict analysis per file
rows = []
for name, fname in FILES.items():
    d = pd.read_parquet(RAW_DIR / fname)
    d["Label"] = d["Label"].astype(str)
    feats = [c for c in d.columns if c != "Label"]

    d["h_exact"] = pd.util.hash_pandas_object(d[feats], index=False).values
    d["h_coarse"] = pd.util.hash_pandas_object(d[COARSE], index=False).values

    benign_exact = set(d.loc[d["Label"] == "Benign", "h_exact"])
    benign_coarse = set(d.loc[d["Label"] == "Benign", "h_coarse"])

    for label in sorted(set(d["Label"]) - {"Benign"}):
        a = d[d["Label"] == label]
        rows.append({
            "file": name,
            "label": label,
            "rows": len(a),
            "exact_benign_twin": a["h_exact"].isin(benign_exact).mean(),
            "coarse_benign_twin": a["h_coarse"].isin(benign_coarse).mean(),
            "unique_attack_vectors": a["h_exact"].nunique(),
        })
    print(f"Done {name}")

res = pd.DataFrame(rows)
print("\nShare of attack rows that have a benign twin:")
print(res.round(4).to_string(index=False))

# %% Overall Infiltration ceiling
inf = res[res["label"] == "Infilteration"]
w = inf["rows"] / inf["rows"].sum()
print(f"\nInfiltration overall: exact twin = {(inf['exact_benign_twin'] * w).sum():.2%}   "
      f"coarse twin = {(inf['coarse_benign_twin'] * w).sum():.2%}")