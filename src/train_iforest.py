# %% Imports + config
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report)
import joblib

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

DATA_FILE = ROOT / "data" / "raw" / "Bruteforce-Wednesday-14-02-2018_TrafficForML_CICFlowMeter.parquet"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

SEED = 42
USE_LOG = True        # log-transform heavy-tailed features
TARGET_FPR = 0.01     # flag ~1% of normal traffic

# %% Load + binary target
df = pd.read_parquet(DATA_FILE)
df["Label"] = df["Label"].astype(str)
df["is_attack"] = (df["Label"] != "Benign").astype(int)
print(df["is_attack"].value_counts())

# %% Feature selection
feature_cols = [c for c in df.columns if c not in ("Label", "is_attack")]

# 1) constant columns carry no information
constant = [c for c in feature_cols if df[c].nunique() <= 1]
feature_cols = [c for c in feature_cols if c not in constant]

# 2) near-duplicate columns (|corr| > 0.99), computed on benign only
corr = df.loc[df["is_attack"] == 0, feature_cols].astype("float64").corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
redundant = [c for c in upper.columns if (upper[c] > 0.99).any()]
feature_cols = [c for c in feature_cols if c not in redundant]

print("Dropped constant :", constant)
print("Dropped redundant:", redundant)
print("Features kept    :", len(feature_cols))

# %% Transform
X = df[feature_cols].astype("float64")
if USE_LOG:
    X = np.sign(X) * np.log1p(np.abs(X))   # handles the -1 values in Init Win Bytes
y = df["is_attack"].values
labels = df["Label"].values

# %% Split: train/val = benign only, test = held-out benign + all attacks
rng = np.random.default_rng(SEED)
benign_idx = np.where(y == 0)[0]
attack_idx = np.where(y == 1)[0]
rng.shuffle(benign_idx)

n = len(benign_idx)
train_idx = benign_idx[: int(0.6 * n)]
val_idx   = benign_idx[int(0.6 * n): int(0.8 * n)]
test_idx  = np.concatenate([benign_idx[int(0.8 * n):], attack_idx])

X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
y_test, labels_test = y[test_idx], labels[test_idx]
print(f"Train={len(train_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")

# %% Train
iforest = IsolationForest(
    n_estimators=200,
    max_samples=256,
    random_state=SEED,
    n_jobs=-1,
)
iforest.fit(X_train)

# %% Scores + threshold
def anomaly_score(model, X):
    return -model.score_samples(X)   # higher = more anomalous

val_scores = anomaly_score(iforest, X_val)
threshold = np.quantile(val_scores, 1 - TARGET_FPR)
test_scores = anomaly_score(iforest, X_test)
y_pred = (test_scores >= threshold).astype(int)
print("Threshold:", round(float(threshold), 4))

# %% Evaluate
print("ROC-AUC:", round(roc_auc_score(y_test, test_scores), 4))
print("PR-AUC :", round(average_precision_score(y_test, test_scores), 4))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print("FPR:", round(fp / (fp + tn), 4))
print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"], digits=4))

print("Detection rate per label:")
print(pd.Series(y_pred).groupby(labels_test).mean())

# %% Save model + config
joblib.dump(iforest, MODEL_DIR / "iforest_v1.joblib")
with open(MODEL_DIR / "iforest_v1_config.json", "w") as f:
    json.dump({
        "data": DATA_FILE.name,
        "features": feature_cols,
        "use_log": USE_LOG,
        "threshold": float(threshold),
        "target_fpr": TARGET_FPR,
    }, f, indent=2)
print("Saved model + config to", MODEL_DIR)
# %% Diagnostic 1: attack recall at different FPR levels
attack_test_scores = test_scores[y_test == 1]
benign_test_scores = test_scores[y_test == 0]
print("\nRecall vs FPR:")
for fpr in [0.01, 0.05, 0.10, 0.20, 0.30]:
    t = np.quantile(val_scores, 1 - fpr)
    print(f"  FPR {fpr:>4.0%}  threshold={t:.4f}  attack recall={(attack_test_scores >= t).mean():.4f}")

# %% Diagnostic 2: score distributions + histogram for the report
print("\nScore distributions:")
print(pd.DataFrame({
    "benign": pd.Series(benign_test_scores).describe(percentiles=[.5, .9, .99]),
    "attack": pd.Series(attack_test_scores).describe(percentiles=[.5, .9, .99]),
}))

import matplotlib.pyplot as plt
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(9, 4))
plt.hist(benign_test_scores, bins=100, alpha=0.6, density=True, label="Benign")
plt.hist(attack_test_scores, bins=100, alpha=0.6, density=True, label="Attack")
plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold (FPR {TARGET_FPR:.0%})")
plt.xlabel("Anomaly score"); plt.ylabel("Density"); plt.legend()
plt.title("Isolation Forest scores: Benign vs Brute Force")
plt.tight_layout()
plt.savefig(FIG_DIR / "iforest_v1_scores.png", dpi=150)
print("Saved histogram to", FIG_DIR)

# %% Diagnostic 3: how repetitive are the attack flows?
attack_df = df.loc[df["is_attack"] == 1, feature_cols]
print(f"\nAttack rows: {len(attack_df)}  unique attack vectors: {len(attack_df.drop_duplicates())}")

# %% Diagnostic 4: attack rows identical to a benign row (label noise check)
benign_unique = df.loc[df["is_attack"] == 0, feature_cols].drop_duplicates()
overlap = attack_df.merge(benign_unique, how="inner", on=feature_cols)
print("Attack rows identical to some Benign row:", len(overlap))

# %% Diagnostic 5: SSH signature vs benign (medians)
ben = df.loc[df["Label"] == "Benign", feature_cols]
ssh = df.loc[df["Label"] == "SSH-Bruteforce", feature_cols]
profile = pd.DataFrame({
    "benign_median": ben.median(),
    "ssh_median": ssh.median(),
    "ssh_std": ssh.std(),
})
print("\nFeature profile:")
print(profile.round(2).to_string())