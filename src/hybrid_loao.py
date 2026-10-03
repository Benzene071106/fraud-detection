# Step 5: hybrid (Rule R1 + Isolation Forest v2 + XGBoost) on the unseen-attack test
# Usage: python -m src.hybrid_loao

# %% Imports + config
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

DATA = ROOT / "data" / "processed" / "all_days.parquet"
MODEL_DIR = ROOT / "models"
OUT = ROOT / "reports" / "hybrid_loao.csv"
SEED = 42
ALERT_FPR = 0.01                 # XGBoost alert threshold
REVIEW_BUDGETS = [0.01, 0.02]    # Isolation Forest REVIEW queue size (share of benign)

CLASSES = ["Benign", "BruteForce", "DoS", "DDoS", "Bot", "Web", "Infiltration"]
K = len(CLASSES)
FINGERPRINTS = ["Init Fwd Win Bytes", "Init Bwd Win Bytes", "Fwd Seg Size Min"]

# %% Load + identical split to train_xgb.py / loao_gpu.py
df = pd.read_parquet(DATA)
meta = ["Label", "family", "day"]
xgb_feats = [c for c in df.columns
             if c not in meta and c not in FINGERPRINTS and df[c].nunique() > 1]
X = df[xgb_feats].astype("float32")
y = df["family"].map({c: i for i, c in enumerate(CLASSES)}).values
labels = df["Label"].values

idx = np.arange(len(df))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=SEED)
te = df.iloc[test_idx]
y_te, lab_te = y[test_idx], labels[test_idx]
benign_te = y_te == 0
print(f"Rows: {len(df):,}   Test: {len(test_idx):,}   Benign test: {benign_te.sum():,}")

# %% Isolation Forest v2 (label-free) - REVIEW thresholds from all-day validation benign
with open(MODEL_DIR / "iforest_v2_config.json") as f:
    icfg = json.load(f)
iforest = joblib.load(MODEL_DIR / "iforest_v2.joblib")

def if_score(frame):
    Xi = frame[icfg["features"]].astype("float64")
    if icfg["use_log"]:
        Xi = np.sign(Xi) * np.log1p(np.abs(Xi))
    return -iforest.score_samples(Xi)

print("Scoring Isolation Forest (1-2 min)...")
if_val = if_score(df.iloc[val_idx[y[val_idx] == 0]])
if_thr = {b: np.quantile(if_val, 1 - b) for b in REVIEW_BUDGETS}
if_te = if_score(te)

# %% Rule R1 v2 (SSH session pattern, 4-of-6 match)
bounds = pd.read_json(MODEL_DIR / "rule_r1_ssh_bounds_v2.json")
with open(MODEL_DIR / "hybrid_v2_config.json") as f:
    MIN_MATCH = json.load(f)["min_match"]

def rule_hit(frame):
    count = np.zeros(len(frame), dtype=int)
    for feat in bounds.index:
        count += frame[feat].between(bounds.loc[feat, "min"], bounds.loc[feat, "max"]).values.astype(int)
    count[frame["Protocol"].values != 6] = 0
    return count >= MIN_MATCH

rule_te = rule_hit(te)
print(f"Rule R1 v2 false-positive rate on ALL-DAY benign test: {rule_te[benign_te].mean():.4%}")
for b in REVIEW_BUDGETS:
    print(f"Isolation Forest top-{b:.0%} FPR on benign test: {(if_te >= if_thr[b])[benign_te].mean():.4%}")

# %% Leave-one-attack-out with the full hybrid
rows = []
for held in CLASSES[1:]:
    t0 = time.time()
    h = CLASSES.index(held)
    keep = [i for i in range(K) if i != h]
    lut = np.full(K, -1)
    lut[keep] = np.arange(len(keep))

    tr = train_idx[y[train_idx] != h]
    va = val_idx[y[val_idx] != h]
    y_tr, y_va = lut[y[tr]], lut[y[va]]
    counts = np.bincount(y_tr, minlength=len(keep))
    w = np.sqrt(counts[0] / counts)[y_tr]

    model = XGBClassifier(
        objective="multi:softprob", n_estimators=600, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", device="cuda",
        eval_metric="mlogloss", early_stopping_rounds=30, random_state=SEED, n_jobs=1,
    )
    model.fit(X.iloc[tr], y_tr, sample_weight=w, eval_set=[(X.iloc[va], y_va)], verbose=False)

    s_va = 1 - model.predict_proba(X.iloc[va])[:, 0]
    thr = np.quantile(s_va[y[va] == 0], 1 - ALERT_FPR)
    xgb_flag = (1 - model.predict_proba(X.iloc[test_idx])[:, 0]) >= thr

    alert = xgb_flag | rule_te
    unseen = y_te == h
    row = {
        "held_out": held,
        "unseen_rows": int(unseen.sum()),
        "xgb_only": xgb_flag[unseen].mean(),
        "rule_only": rule_te[unseen].mean(),
        "iforest_only_1pct": (if_te >= if_thr[0.01])[unseen].mean(),
        "ALERT": alert[unseen].mean(),
        "ALERT_fpr": alert[benign_te].mean(),
    }
    for b in REVIEW_BUDGETS:
        flagged = alert | (if_te >= if_thr[b])
        tag = f"{int(b * 100)}pct"
        row[f"ALERT+REVIEW_{tag}"] = flagged[unseen].mean()
        row[f"total_fpr_{tag}"] = flagged[benign_te].mean()
    rows.append(row)

    flagged2 = alert | (if_te >= if_thr[0.02])
    print(f"\n=== Held out: {held} ===  XGB {row['xgb_only']:.2%} | rule {row['rule_only']:.2%} | "
          f"IF {row['iforest_only_1pct']:.2%} | ALERT {row['ALERT']:.2%} | "
          f"ALERT+REVIEW(2%) {row['ALERT+REVIEW_2pct']:.2%}   [{(time.time() - t0) / 60:.1f} min]")
    print("  ALERT+REVIEW(2%) per original label:")
    print(pd.Series(flagged2[unseen]).groupby(lab_te[unseen]).mean().round(4).to_string())

# %% Summary
res = pd.DataFrame(rows)
print("\n\n===== HYBRID ON UNSEEN ATTACKS (detection of the held-out family) =====")
print(res.round(4).to_string(index=False))
res.to_csv(OUT, index=False)
print(f"\nSaved to {OUT}")