# Stage 2: tune per-class thresholds (Web, Infiltration, DDoS, Bot) so benign FPR <= 1%
# Usage:
#   python -m src.xgb_thresholds --nofp
#   python -m src.xgb_thresholds --tag xgb_v1_nofp_clean_r1500

# %% Imports + config
from pathlib import Path
import argparse
import itertools
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--nofp", action="store_true")
parser.add_argument("--tag", default=None, help="model name, e.g. xgb_v1_nofp_clean_r1500")
args, _ = parser.parse_known_args()

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

DATA = ROOT / "data" / "processed" / "all_days.parquet"
MODEL_DIR = ROOT / "models"
TAG = args.tag or ("xgb_v1_nofp" if args.nofp else "xgb_v1")
SEED = 42
TARGET_FPR = 0.01
print(f"Model: {TAG}")

with open(MODEL_DIR / f"{TAG}_config.json") as f:
    cfg = json.load(f)
CLASSES, FEATURES = cfg["classes"], cfg["features"]
K = len(CLASSES)
model = XGBClassifier()
model.load_model(MODEL_DIR / f"{TAG}.json")

# %% Rebuild the identical split from train_xgb.py
df = pd.read_parquet(DATA, columns=FEATURES + ["family", "Label"])
X = df[FEATURES].astype("float32")
y = df["family"].map({c: i for i, c in enumerate(CLASSES)}).values
idx = np.arange(len(df))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=SEED)

P_val = model.predict_proba(X.iloc[val_idx])
P_test = model.predict_proba(X.iloc[test_idx])
y_val, y_test = y[val_idx], y[test_idx]

WEB, INF = CLASSES.index("Web"), CLASSES.index("Infiltration")
DDOS, BOT = CLASSES.index("DDoS"), CLASSES.index("Bot")

# %% Decision rule + fast metrics
def decide(P, thr):
    masked = np.where(P >= thr, P, -1.0)
    masked[:, 0] = -0.5
    return masked.argmax(axis=1)

def prf(cm):
    tp = np.diag(cm).astype(float)
    col, row = cm.sum(axis=0), cm.sum(axis=1)
    p = np.divide(tp, col, out=np.zeros_like(tp), where=col > 0)
    r = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return p, r, f, row

def metrics(y_true, pred):
    cm = np.bincount(y_true * K + pred, minlength=K * K).reshape(K, K)
    p, r, f, support = prf(cm)
    keep = [i for i in range(K) if i != INF]
    cm_ex = cm.copy()
    cm_ex[INF, :] = 0
    _, r_ex, f_ex, _ = prf(cm_ex)
    return {
        "fpr": 1 - cm[0, 0] / cm[0].sum(),
        "macro_f1": f.mean(),
        "macro_f1_ex_inf": f_ex[keep].mean(),
        "macro_recall": r.mean(),
        "weighted_f1": (f * support).sum() / support.sum(),
        "web_f1": f[WEB], "inf_f1": f[INF],
        "web_recall": r[WEB], "inf_recall": r[INF],
    }

# %% Tune on VALIDATION only
GRID_WEB = [0.8, 0.9, 0.95, 0.98, 0.99]
GRID_INF = [0.6, 0.7, 0.8, 0.9, 1.01]          # 1.01 = Infiltration never alerts
GRID_DDOS = [0.5, 0.6, 0.7, 0.8, 0.9]
GRID_BOT = [0.5, 0.6, 0.7, 0.8, 0.9]

rows = []
for tw, ti, td, tb in itertools.product(GRID_WEB, GRID_INF, GRID_DDOS, GRID_BOT):
    thr = np.full(K, 0.5)
    thr[WEB], thr[INF], thr[DDOS], thr[BOT] = tw, ti, td, tb
    m = metrics(y_val, decide(P_val, thr))
    rows.append({"t_web": tw, "t_inf": ti, "t_ddos": td, "t_bot": tb, **m})

res = pd.DataFrame(rows)
ok = res[res["fpr"] <= TARGET_FPR].sort_values(["macro_f1", "fpr"], ascending=[False, True])
if ok.empty:
    print(res.sort_values("fpr").head(10).round(4).to_string(index=False))
    raise SystemExit("No threshold set reaches the FPR target - send me this table.")
print(f"Tried {len(res)} threshold sets. Top 10 (validation):")
print(ok.head(10).round(4).to_string(index=False))

# If macro-F1 is practically tied (within 0.001), prefer the lowest FPR
near_best = ok[ok["macro_f1"] >= ok["macro_f1"].max() - 0.001]
best = near_best.sort_values("fpr").iloc[0]
print(f"\n{len(near_best)} settings within 0.001 of best macro-F1 -> picked lowest FPR: {best['fpr']:.4f}")

thr = np.full(K, 0.5)
thr[WEB], thr[INF], thr[DDOS], thr[BOT] = best["t_web"], best["t_inf"], best["t_ddos"], best["t_bot"]
print("Chosen thresholds:", dict(zip(CLASSES, thr.round(3).tolist())))

# %% Final test evaluation
for name, pred in [("Argmax (before)", P_test.argmax(axis=1)),
                   ("Tuned thresholds (after)", decide(P_test, thr))]:
    m = metrics(y_test, pred)
    print(f"\n===== {name} =====")
    print(f"Benign FPR: {m['fpr']:.4f}   Macro-F1: {m['macro_f1']:.4f}   "
          f"Macro-F1 excl. Inf: {m['macro_f1_ex_inf']:.4f}   "
          f"Macro recall: {m['macro_recall']:.4f}   Weighted F1: {m['weighted_f1']:.4f}")
    print(classification_report(y_test, pred, target_names=CLASSES, digits=4, zero_division=0))

print("Confusion matrix after tuning (rows = true, cols = predicted):")
print(pd.DataFrame(confusion_matrix(y_test, decide(P_test, thr)),
                   index=CLASSES, columns=CLASSES).to_string())

# %% Save
with open(MODEL_DIR / f"{TAG}_thresholds.json", "w") as f:
    json.dump(dict(zip(CLASSES, thr.tolist())), f, indent=2)
print(f"\nSaved thresholds to {MODEL_DIR / f'{TAG}_thresholds.json'}")