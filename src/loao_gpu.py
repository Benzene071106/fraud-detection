# Step 4: leave-one-attack-out (unseen attack test) on GPU
# Usage: python -m src.loao_gpu

# %% Imports + config
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

DATA = ROOT / "data" / "processed" / "all_days.parquet"
OUT = ROOT / "reports" / "loao_xgb.csv"
OUT.parent.mkdir(exist_ok=True)
SEED = 42
TARGET_FPR = 0.01

CLASSES = ["Benign", "BruteForce", "DoS", "DDoS", "Bot", "Web", "Infiltration"]
K = len(CLASSES)
FINGERPRINTS = ["Init Fwd Win Bytes", "Init Bwd Win Bytes", "Fwd Seg Size Min"]

# %% Load + identical split to train_xgb.py
df = pd.read_parquet(DATA)
meta = ["Label", "family", "day"]
features = [c for c in df.columns
            if c not in meta and c not in FINGERPRINTS and df[c].nunique() > 1]
X = df[features].astype("float32")
y = df["family"].map({c: i for i, c in enumerate(CLASSES)}).values
labels = df["Label"].values
print(f"Rows: {len(df):,}   Features: {len(features)} (no fingerprints)")

idx = np.arange(len(df))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=SEED)
y_te, lab_te = y[test_idx], labels[test_idx]

# %% One model per held-out family
rows = []
for held in CLASSES[1:]:
    t0 = time.time()
    h = CLASSES.index(held)
    keep = [i for i in range(K) if i != h]
    lut = np.full(K, -1)
    lut[keep] = np.arange(len(keep))          # remap remaining classes to 0..5 (Benign stays 0)

    tr = train_idx[y[train_idx] != h]          # held-out family removed from TRAINING
    va = val_idx[y[val_idx] != h]              # ...and from VALIDATION (it is "unknown")
    y_tr, y_va = lut[y[tr]], lut[y[va]]

    counts = np.bincount(y_tr, minlength=len(keep))
    w = np.sqrt(counts[0] / counts)[y_tr]

    model = XGBClassifier(
        objective="multi:softprob", n_estimators=600, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", device="cuda",
        eval_metric="mlogloss", early_stopping_rounds=30, random_state=SEED, n_jobs=1,
    )
    model.fit(X.iloc[tr], y_tr, sample_weight=w,
              eval_set=[(X.iloc[va], y_va)], verbose=False)

    # attack score = 1 - P(benign); threshold = 1% FPR on validation benign
    s_va = 1 - model.predict_proba(X.iloc[va])[:, 0]
    thr = np.quantile(s_va[y[va] == 0], 1 - TARGET_FPR)
    P_te = model.predict_proba(X.iloc[test_idx])
    s_te = 1 - P_te[:, 0]
    flag = s_te >= thr

    unseen = y_te == h
    known_attack = (y_te != 0) & ~unseen
    rows.append({
        "held_out": held,
        "unseen_test_rows": int(unseen.sum()),
        "unseen_detected_1pct_fpr": flag[unseen].mean(),
        "known_attacks_detected": flag[known_attack].mean(),
        "benign_fpr": flag[y_te == 0].mean(),
        "rounds": int(model.best_iteration),
        "minutes": round((time.time() - t0) / 60, 1),
    })

    # what did the model think the unseen attack was?
    guessed = pd.Series(np.array(CLASSES)[np.array(keep)[P_te[unseen].argmax(axis=1)]])
    print(f"\n=== Held out: {held} ===  detected {flag[unseen].mean():.2%}  "
          f"(benign FPR {flag[y_te == 0].mean():.2%})  [{rows[-1]['minutes']} min]")
    print("  Classified as:", guessed.value_counts(normalize=True).round(3).to_dict())
    print("  Detection per original label:")
    print(pd.Series(flag[unseen]).groupby(lab_te[unseen]).mean().round(4).to_string())

# %% Summary
res = pd.DataFrame(rows)
print("\n\n===== LEAVE-ONE-ATTACK-OUT SUMMARY =====")
print(res.round(4).to_string(index=False))
res.to_csv(OUT, index=False)
print(f"\nSaved to {OUT}")