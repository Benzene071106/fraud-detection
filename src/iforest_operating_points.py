# %% Imports + same splits as before
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, confusion_matrix

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

MODEL_DIR = ROOT / "models"
with open(MODEL_DIR / "iforest_v2_config.json") as f:
    cfg = json.load(f)
model = joblib.load(MODEL_DIR / "iforest_v2.joblib")
FEATS, USE_LOG, SEED = cfg["features"], cfg["use_log"], 42

df = pd.read_parquet(ROOT / "data" / "raw" / cfg["data"])
y = (df["Label"].astype(str) != "Benign").astype(int).values

rng = np.random.default_rng(SEED)
benign_idx, attack_idx = np.where(y == 0)[0], np.where(y == 1)[0]
rng.shuffle(benign_idx)
n = len(benign_idx)
val_idx = benign_idx[int(0.6 * n): int(0.8 * n)]
test_benign_idx = benign_idx[int(0.8 * n):]
rng2 = np.random.default_rng(SEED + 1)
attack_idx = attack_idx.copy()
rng2.shuffle(attack_idx)
eval_idx = np.concatenate([test_benign_idx, attack_idx[len(attack_idx) // 2:]])

def to_X(frame):
    X = frame[FEATS].astype("float64")
    return np.sign(X) * np.log1p(np.abs(X)) if USE_LOG else X

sv = -model.score_samples(to_X(df.iloc[val_idx]))
se = -model.score_samples(to_X(df.iloc[eval_idx]))
y_ev = y[eval_idx]

# %% Operating points: test-set F1 vs F1 at a realistic 1% attack share
BASE_RATE = 0.01
rows = []
for target in [0.01, 0.02, 0.03, 0.04, 0.05]:
    pred = (se >= np.quantile(sv, 1 - target)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_ev, pred).ravel()
    rec, fpr = tp / (tp + fn), fp / (fp + tn)
    prec_real = rec * BASE_RATE / (rec * BASE_RATE + fpr * (1 - BASE_RATE)) if rec > 0 else 0
    f1_real = 2 * prec_real * rec / (prec_real + rec) if rec > 0 else 0
    rows.append({"target_fpr": target, "recall": rec, "actual_fpr": fpr,
                 "f1_test_set": f1_score(y_ev, pred), "f1_at_1pct_attacks": f1_real,
                 "false_alerts_per_830k_benign": int(fpr * 830_000)})

print(pd.DataFrame(rows).round(4).to_string(index=False))
