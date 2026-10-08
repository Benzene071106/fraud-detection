# NetFlow track: threshold tuning for the honest model (no ports/TTL/window, duration fixed)
# Usage: python -m src.nf_thresholds

# %% Imports + config
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

DATA = ROOT / "data" / "processed" / "nfuq_sample.parquet"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
TAG = "nfuq_xgb_fixdur_noports_nofp"
SEED = 42
WRAP_MS = 4_294_967.296
BASE_RATE = 0.01
TARGETS = [0.01, 0.005, 0.002]

with open(MODEL_DIR / f"{TAG}_config.json") as f:
    cfg = json.load(f)
FEATS, CLASSES = cfg["features"], cfg["classes"]
model = XGBClassifier()
model.load_model(MODEL_DIR / f"{TAG}.json")

# %% Load, apply the same duration fix, rebuild the identical split
df = pd.read_parquet(DATA)
dur = df["FLOW_DURATION_MILLISECONDS"]
band = dur.between(4_280_000, 4_294_968)
df["FLOW_DURATION_MILLISECONDS"] = np.where(band, WRAP_MS - dur, dur).astype("float32")

y = df["Attack"].map({c: i for i, c in enumerate(CLASSES)}).values
src = df["Dataset"].values
strat = (df["Dataset"] + "|" + df["Attack"]).values
idx = np.arange(len(df))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=strat, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=strat[temp_idx], random_state=SEED)
y_va, y_te = y[val_idx], y[test_idx]
src_va, src_te = src[val_idx], src[test_idx]
X = df[FEATS].astype("float32")

print(f"Model: {TAG}   Features: {len(FEATS)}")
print("Predicting validation + test (2-5 min)...")
P_va = model.predict_proba(X.iloc[val_idx])
P_te = model.predict_proba(X.iloc[test_idx])
s_va, s_te = 1 - P_va[:, 0], 1 - P_te[:, 0]          # attack score
atk_te = P_te[:, 1:].argmax(axis=1) + 1              # most likely attack class
NETS = sorted(np.unique(src_te))
short = lambda s: s.replace("NF-", "").replace("-v2", "")

# %% Evaluation helper
def evaluate(flag, name):
    pred = np.where(flag, atk_te, 0)
    ben = y_te == 0
    fpr, det = flag[ben].mean(), flag[~ben].mean()
    row = {"setting": name, "benign_fpr": fpr, "detection": det,
           "precision@1%": det * BASE_RATE / (det * BASE_RATE + fpr * (1 - BASE_RATE)) if det > 0 else 0,
           "macro_f1": f1_score(y_te, pred, average="macro", zero_division=0),
           "weighted_f1": f1_score(y_te, pred, average="weighted", zero_division=0)}
    for n in NETS:
        row[f"fpr_{short(n)}"] = flag[(src_te == n) & ben].mean()
    for n in NETS:
        row[f"det_{short(n)}"] = flag[(src_te == n) & ~ben].mean()
    return row

rows, flags = [], {}
flags["argmax"] = P_te.argmax(axis=1) != 0
rows.append(evaluate(flags["argmax"], "argmax"))
saved = {"global": {}, "per_network": {}}

for t in TARGETS:
    thr = np.quantile(s_va[y_va == 0], 1 - t)
    saved["global"][str(t)] = float(thr)
    name = f"global {t:.1%}"
    flags[name] = s_te >= thr
    rows.append(evaluate(flags[name], name))

    flag_pn = np.zeros(len(s_te), dtype=bool)
    saved["per_network"][str(t)] = {}
    for n in NETS:
        thr_n = np.quantile(s_va[(y_va == 0) & (src_va == n)], 1 - t)
        saved["per_network"][str(t)][n] = float(thr_n)
        m = src_te == n
        flag_pn[m] = s_te[m] >= thr_n
    name = f"per-network {t:.1%}"
    flags[name] = flag_pn
    rows.append(evaluate(flag_pn, name))

# %% Results
res = pd.DataFrame(rows).set_index("setting")
pd.set_option("display.width", 250)
print("\n===== OPERATING POINTS (test set) =====")
print(res.round(4).to_string())

atk = y_te != 0
names = np.array(CLASSES)[y_te[atk]]
det = pd.DataFrame({k: pd.Series(v[atk]).groupby(names).mean() for k, v in flags.items()})
print("\n===== DETECTION PER ATTACK CLASS (flagged as any attack) =====")
print(det.round(4).to_string())

res.to_csv(REPORT_DIR / "nfuq_thresholds_summary.csv")
det.to_csv(REPORT_DIR / "nfuq_thresholds_per_class.csv")
with open(MODEL_DIR / f"{TAG}_thresholds.json", "w") as f:
    json.dump(saved, f, indent=2)
print(f"\nSaved summary + per-class tables to {REPORT_DIR}, thresholds to {MODEL_DIR}")
