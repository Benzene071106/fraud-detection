# Precompute assets for the API: Isolation Forest reference scores, benign medians, example flows
# Usage (run once, after the GPU job finishes): python -m src.export_service_assets
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "all_days.parquet"
MODEL_DIR = ROOT / "models"
XGB_TAG = "xgb_v1_nofp_clean_r1500"
SEED = 42

with open(MODEL_DIR / f"{XGB_TAG}_config.json") as f:
    cfg = json.load(f)
with open(MODEL_DIR / "iforest_v2_config.json") as f:
    icfg = json.load(f)
iforest = joblib.load(MODEL_DIR / "iforest_v2.joblib")
rule_feats = list(pd.read_json(MODEL_DIR / "rule_r1_ssh_bounds_v2.json").index)

df = pd.read_parquet(DATA)
df["Label"] = df["Label"].astype(str)
y = df["family"].map({c: i for i, c in enumerate(cfg["classes"])}).values
idx = np.arange(len(df))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=SEED)
va, te = df.iloc[val_idx], df.iloc[test_idx]
vb = va[y[val_idx] == 0]

Xi = vb[icfg["features"]].astype("float64")
if icfg["use_log"]:
    Xi = np.sign(Xi) * np.log1p(np.abs(Xi))
ref = np.sort(-iforest.score_samples(Xi))
np.save(MODEL_DIR / "iforest_v2_val_benign_scores.npy", ref)

feats = sorted(set(cfg["features"]) | set(icfg["features"]) | set(rule_feats) | {"Protocol"})
with open(MODEL_DIR / "benign_medians.json", "w") as f:
    json.dump(vb[feats].median().astype(float).to_dict(), f, indent=2)

examples = []
for label in sorted(te["Label"].unique()):
    rows = te[te["Label"] == label].sample(min(2, int((te["Label"] == label).sum())), random_state=SEED)
    for _, r in rows.iterrows():
        examples.append({"true_label": label, **{c: float(r[c]) for c in feats}})
with open(MODEL_DIR / "example_flows.json", "w") as f:
    json.dump(examples, f)

print(f"Saved IF reference ({len(ref):,} scores), benign medians ({len(feats)} features), "
      f"{len(examples)} example flows -> {MODEL_DIR}")