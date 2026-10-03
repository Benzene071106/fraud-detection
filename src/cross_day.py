# %% Imports + config
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "raw"
with open(MODEL_DIR / "iforest_v1_config.json") as f:
    cfg = json.load(f)
iforest = joblib.load(MODEL_DIR / "iforest_v1.joblib")
bounds = pd.read_json(MODEL_DIR / "rule_r1_ssh_bounds.json")
FEATURES, USE_LOG, SEED = cfg["features"], cfg["use_log"], 42

def to_X(frame):
    X = frame[FEATURES].astype("float64")
    return np.sign(X) * np.log1p(np.abs(X)) if USE_LOG else X

def rule_hit(frame):
    hit = frame["Protocol"] == 6
    for feat in bounds.index:
        hit &= frame[feat].between(bounds.loc[feat, "min"], bounds.loc[feat, "max"])
    return hit.astype(int).values

# %% Rebuild the Wednesday benign validation reference (same split as training)
wed = pd.read_parquet(DATA_DIR / cfg["data"])
is_attack = (wed["Label"].astype(str) != "Benign").values
benign_idx = np.where(~is_attack)[0]
np.random.default_rng(SEED).shuffle(benign_idx)
n = len(benign_idx)
val = wed.iloc[benign_idx[int(0.6 * n): int(0.8 * n)]]
val_scores = np.sort(-iforest.score_samples(to_X(val)))
del wed, val

def iforest_pct(frame):
    s = -iforest.score_samples(to_X(frame))
    return np.searchsorted(val_scores, s, side="right") / len(val_scores)

# %% Run frozen model + rule on every other day
results = []
for path in sorted(DATA_DIR.glob("*.parquet")):
    if path.name == cfg["data"]:
        continue
    print("Scoring", path.name, "...")
    d = pd.read_parquet(path)
    d["Label"] = d["Label"].astype(str)
    d["if_flag"] = (iforest_pct(d) >= 0.99).astype(int)   # same 1% FPR threshold
    d["rule_flag"] = rule_hit(d)
    summary = d.groupby("Label").agg(
        rows=("if_flag", "size"),
        iforest_detect=("if_flag", "mean"),
        rule_detect=("rule_flag", "mean"),
    ).reset_index()
    summary.insert(0, "file", path.name.split("_")[0])
    results.append(summary)

out = pd.concat(results, ignore_index=True)
print("\nCross-day results (for Benign rows, the rate = false positive rate):")
print(out.round(4).to_string(index=False))

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)
out.to_csv(REPORT_DIR / "cross_day_v1.csv", index=False)
print("\nSaved to", REPORT_DIR / "cross_day_v1.csv")