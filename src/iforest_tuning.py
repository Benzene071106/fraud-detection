# %% Imports + config
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

MODEL_DIR = ROOT / "models"
with open(MODEL_DIR / "iforest_v1_config.json") as f:
    cfg = json.load(f)
DATA_FILE = ROOT / "data" / "raw" / cfg["data"]
BASE = cfg["features"]          # the 47 features from v1
SEED = 42
TARGET_FPR = 0.01

# %% Load + identical splits to v1
df = pd.read_parquet(DATA_FILE)
df["Label"] = df["Label"].astype(str)
df["is_attack"] = (df["Label"] != "Benign").astype(int)
y = df["is_attack"].values

rng = np.random.default_rng(SEED)
benign_idx = np.where(y == 0)[0]
attack_idx = np.where(y == 1)[0]
rng.shuffle(benign_idx)
n = len(benign_idx)
train_idx = benign_idx[: int(0.6 * n)]
val_idx = benign_idx[int(0.6 * n): int(0.8 * n)]
test_benign_idx = benign_idx[int(0.8 * n):]

rng2 = np.random.default_rng(SEED + 1)
attack_idx = attack_idx.copy()
rng2.shuffle(attack_idx)
design_idx = attack_idx[: len(attack_idx) // 2]
eval_attack_idx = attack_idx[len(attack_idx) // 2:]

train, val = df.iloc[train_idx], df.iloc[val_idx]
design = df.iloc[design_idx]
ev = df.iloc[np.concatenate([test_benign_idx, eval_attack_idx])]
y_ev = ev["is_attack"].values

# %% Feature sets
FLAGS = [c for c in ["Fwd PSH Flags", "FIN Flag Count", "RST Flag Count",
                     "PSH Flag Count", "ACK Flag Count", "URG Flag Count"] if c in BASE]
FINGERPRINT = [c for c in ["Init Fwd Win Bytes", "Init Bwd Win Bytes", "Fwd Seg Size Min"] if c in BASE]
CORE = [c for c in [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packets Length Total", "Fwd Packet Length Max", "Bwd Packet Length Max",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Max",
    "Bwd IAT Total", "Fwd Header Length", "Packet Length Mean",
    "Fwd Act Data Packets", "Idle Min", "Active Max",
] if c in BASE]

FEATURE_SETS = {
    "all_47": BASE,
    "no_flags": [c for c in BASE if c not in FLAGS],
    "no_flags_no_fp": [c for c in BASE if c not in FLAGS + FINGERPRINT],
    "core": CORE,
}
for k, v in FEATURE_SETS.items():
    print(f"{k:<16} {len(v)} features")

def to_X(frame, feats, use_log):
    X = frame[feats].astype("float64")
    return np.sign(X) * np.log1p(np.abs(X)) if use_log else X

def fit_iforest(feats, use_log, max_samples):
    return IsolationForest(n_estimators=200, max_samples=max_samples,
                           random_state=SEED, n_jobs=-1).fit(to_X(train, feats, use_log))

# %% Grid search (validation benign + design attacks ONLY)
rows = []
for fs_name, feats in FEATURE_SETS.items():
    for use_log in [True, False]:
        Xv, Xd = to_X(val, feats, use_log), to_X(design, feats, use_log)
        for ms in [256, 1024, 4096]:
            model = fit_iforest(feats, use_log, ms)
            sv, sd = -model.score_samples(Xv), -model.score_samples(Xd)
            thr = np.quantile(sv, 1 - TARGET_FPR)
            tp = int((sd >= thr).sum())
            fp = int((sv >= thr).sum())
            fn = len(sd) - tp
            rows.append({
                "features": fs_name, "log": use_log, "max_samples": ms,
                "auc": roc_auc_score(np.r_[np.zeros(len(sv)), np.ones(len(sd))], np.r_[sv, sd]),
                "recall_1pct": tp / len(sd),
                "recall_5pct": (sd >= np.quantile(sv, 0.95)).mean(),
                "f1_1pct": 2 * tp / (2 * tp + fp + fn),
            })
            r = rows[-1]
            print(f"  {fs_name:<16} log={str(use_log):<5} ms={ms:<5} "
                  f"AUC={r['auc']:.4f}  recall@1%={r['recall_1pct']:.4f}  F1={r['f1_1pct']:.4f}")

res = pd.DataFrame(rows).sort_values(["f1_1pct", "auc"], ascending=False)
print("\nGrid results (sorted by F1 at 1% FPR):")
print(res.round(4).to_string(index=False))

# %% Final evaluation on eval half: v1 baseline vs best
def evaluate(name, fs_name, use_log, ms):
    feats = FEATURE_SETS[fs_name]
    model = fit_iforest(feats, use_log, ms)
    thr = np.quantile(-model.score_samples(to_X(val, feats, use_log)), 1 - TARGET_FPR)
    s = -model.score_samples(to_X(ev, feats, use_log))
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_ev, pred).ravel()
    print(f"{name:<12} [{fs_name}, log={use_log}, ms={ms}]  AUC={roc_auc_score(y_ev, s):.4f}  "
          f"recall={tp/(tp+fn):.4f}  FPR={fp/(fp+tn):.4f}  F1={f1_score(y_ev, pred):.4f}")
    return model, feats, thr

best = res.iloc[0]
print()
evaluate("v1 baseline", "all_47", True, 256)
model, feats, thr = evaluate("v2 best", best["features"], bool(best["log"]), int(best["max_samples"]))

# %% Save v2
joblib.dump(model, MODEL_DIR / "iforest_v2.joblib")
with open(MODEL_DIR / "iforest_v2_config.json", "w") as f:
    json.dump({"data": DATA_FILE.name, "features": feats, "use_log": bool(best["log"]),
               "max_samples": int(best["max_samples"]), "threshold": float(thr),
               "target_fpr": TARGET_FPR}, f, indent=2)
print("\nSaved iforest_v2 to", MODEL_DIR)