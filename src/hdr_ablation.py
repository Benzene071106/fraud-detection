# Step A: does XGBoost rely on Fwd/Bwd Header Length (TCP-option / OS fingerprint)?
# Usage: python -m src.hdr_ablation

# %% Imports + config
from pathlib import Path
import itertools
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
SEED = 42
TARGET_ALERT_FPR = 0.002
BASE_RATE = 0.01
CLASSES = ["Benign", "BruteForce", "DoS", "DDoS", "Bot", "Web", "Infiltration"]
K = len(CLASSES)
C = {c: i for i, c in enumerate(CLASSES)}
BF, WEB, INF = C["BruteForce"], C["Web"], C["Infiltration"]
FINGERPRINTS = ["Init Fwd Win Bytes", "Init Bwd Win Bytes", "Fwd Seg Size Min"]
HEADER = ["Fwd Header Length", "Bwd Header Length"]

# %% Load + identical split
df = pd.read_parquet(DATA)
meta = ["Label", "family", "day"]
base_feats = [c for c in df.columns if c not in meta and c not in FINGERPRINTS and df[c].nunique() > 1]
y = df["family"].map(C).values
idx = np.arange(len(df))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=SEED)
y_va, y_te = y[val_idx], y[test_idx]
counts = np.bincount(y[train_idx], minlength=K)
w = np.sqrt(counts[0] / counts)[y[train_idx]]

# %% Helpers (same Variant B logic as refine_final, XGBoost only)
def decide(P, thr):
    masked = np.where(P >= thr, P, -1.0)
    masked[:, 0] = -0.5
    return masked.argmax(axis=1)

def macro_f1_ex_inf(y_true, pred):
    cm = np.bincount(y_true * K + pred, minlength=K * K).reshape(K, K)
    cm[INF, :] = 0
    tp = np.diag(cm).astype(float)
    col, row = cm.sum(axis=0), cm.sum(axis=1)
    p = np.divide(tp, col, out=np.zeros_like(tp), where=col > 0)
    r = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return f[[i for i in range(K) if i != INF]].mean()

def variant_b(P, thr):
    cls = decide(P, thr)
    alert = (cls != 0) & (cls != INF)
    return alert, np.where(alert, cls, 0), cls

GRID = {"DDoS": [0.8, 0.9, 0.95, 0.98, 0.99], "DoS": [0.5, 0.7, 0.9, 0.95, 0.98],
        "Bot": [0.7, 0.9, 0.95, 0.98], "BruteForce": [0.5, 0.9, 0.95]}

def tune(P_va):
    base = np.full(K, 0.5)
    base[WEB], base[INF] = 0.95, 0.8
    rows = []
    for combo in itertools.product(*GRID.values()):
        thr = base.copy()
        for c, v in zip(GRID, combo):
            thr[C[c]] = v
        alert, alert_cls, _ = variant_b(P_va, thr)
        rows.append((alert[y_va == 0].mean(), macro_f1_ex_inf(y_va, alert_cls), thr))
    ok = [r for r in rows if r[0] <= TARGET_ALERT_FPR]
    if not ok:
        best_f1 = max(r[1] for r in rows)
        return min((r for r in rows if r[1] >= best_f1 - 0.005), key=lambda r: r[0])[2]
    return max(ok, key=lambda r: (r[1], -r[0]))[2]

def run(name, feats):
    t0 = time.time()
    m = XGBClassifier(objective="multi:softprob", n_estimators=1000, max_depth=8, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8, tree_method="hist", device="cuda",
                      eval_metric="mlogloss", early_stopping_rounds=50, random_state=SEED, n_jobs=2)
    X = df[feats].astype("float32")
    m.fit(X.iloc[train_idx], y[train_idx], sample_weight=w,
          eval_set=[(X.iloc[val_idx], y_va)], verbose=False)
    thr = tune(m.predict_proba(X.iloc[val_idx]))
    alert, alert_cls, cls = variant_b(m.predict_proba(X.iloc[test_idx]), thr)
    fpr, rec = alert[y_te == 0].mean(), alert[y_te != 0].mean()
    row = {"model": name, "features": len(feats), "alert_fpr": fpr,
           "precision@1%": rec * BASE_RATE / (rec * BASE_RATE + fpr * (1 - BASE_RATE)),
           "detection": rec, "macro_f1_ex_inf": macro_f1_ex_inf(y_te, alert_cls)}
    for c in CLASSES[1:]:
        k = C[c]
        row[c] = (cls[y_te == k] == INF).mean() if k == INF else alert[y_te == k].mean()
    imp = pd.Series(m.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    print(f"\n{name}: {(time.time() - t0) / 60:.1f} min, best iteration {m.best_iteration}")
    print("  top 5 features:", list(imp.head(5).index))
    return row

rows = [run("baseline (no fingerprints)", base_feats),
        run("no header lengths", [c for c in base_feats if c not in HEADER])]
res = pd.DataFrame(rows).set_index("model").T
print("\n===== HEADER-LENGTH ABLATION (Variant B, tuned on validation, test results) =====")
print("(Infiltration row = share sent to REVIEW by XGBoost)")
print(res.to_string())