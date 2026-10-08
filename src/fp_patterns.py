# Step B: what do false ALERTs look like, and can a simple suppression rule remove them?
# Usage: python -m src.fp_patterns

# %% Imports + config
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

DATA = ROOT / "data" / "processed" / "all_days.parquet"
MODEL_DIR = ROOT / "models"
XGB_TAG = "xgb_v1_nofp_clean_r1500"
SEED = 42
MAX_TP_LOSS = 0.0005           # suppression may cost at most 0.05% of real alerts (on validation)

with open(MODEL_DIR / f"{XGB_TAG}_config.json") as f:
    cfg = json.load(f)
CLASSES, XFEATS = cfg["classes"], cfg["features"]
C = {c: i for i, c in enumerate(CLASSES)}
BF, INF = C["BruteForce"], C["Infiltration"]
with open(MODEL_DIR / f"{XGB_TAG}_variantB_thresholds.json") as f:
    thr_map = json.load(f)
THR = np.array([thr_map[c] for c in CLASSES])
model = XGBClassifier()
model.load_model(MODEL_DIR / f"{XGB_TAG}.json")

# %% Load + identical split
df = pd.read_parquet(DATA)
y = df["family"].map(C).values
idx = np.arange(len(df))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=SEED)
va, te = df.iloc[val_idx], df.iloc[test_idx]
y_va, y_te = y[val_idx], y[test_idx]

bounds = pd.read_json(MODEL_DIR / "rule_r1_ssh_bounds_v2.json")
with open(MODEL_DIR / "hybrid_v2_config.json") as f:
    MIN_MATCH = json.load(f)["min_match"]

def rule(frame):
    count = np.zeros(len(frame), dtype=int)
    for feat in bounds.index:
        count += frame[feat].between(bounds.loc[feat, "min"], bounds.loc[feat, "max"]).values.astype(int)
    count[frame["Protocol"].values != 6] = 0
    return count >= MIN_MATCH

def alerts(frame):
    P = model.predict_proba(frame[XFEATS].astype("float32"))
    masked = np.where(P >= THR, P, -1.0)
    masked[:, 0] = -0.5
    cls = masked.argmax(axis=1)
    cls = np.where(rule(frame) & (cls == 0), BF, cls)
    return (cls != 0) & (cls != INF), cls

print("Predicting validation + test...")
a_va, cls_va = alerts(va)
a_te, cls_te = alerts(te)

# %% Who are the false ALERTs?
for name, a, cls, yy in [("VALIDATION", a_va, cls_va, y_va), ("TEST", a_te, cls_te, y_te)]:
    fp = a & (yy == 0)
    print(f"\n{name}: {fp.sum():,} false ALERTs = {fp.sum() / (yy == 0).sum():.4%} of benign")
    print("  predicted as:", pd.Series(np.array(CLASSES)[cls[fp]]).value_counts().to_dict())

# %% Candidate suppression rule: zero forward bytes + long idle
def pattern(frame, idle_s):
    zero_fwd = frame["Fwd Packets Length Total"].values == 0
    longest_gap = np.maximum(frame["Idle Max"].values, frame["Flow IAT Max"].values)
    return zero_fwd & (longest_gap >= idle_s * 1e6)        # CIC timings are in microseconds

def effect(frame, a, yy, idle_s):
    pat = pattern(frame, idle_s)
    fp, tp = a & (yy == 0), a & (yy != 0)
    new_fp = (fp & ~pat).sum()
    return {"idle_s": idle_s, "fp_removed": int((fp & pat).sum()), "fp_removed_%": (fp & pat).sum() / max(fp.sum(), 1),
            "tp_lost": int((tp & pat).sum()), "tp_lost_%": (tp & pat).sum() / max(tp.sum(), 1),
            "new_alert_fpr": new_fp / (yy == 0).sum()}

rows_va = pd.DataFrame([effect(va, a_va, y_va, s) for s in [10, 30, 60, 120]])
rows_te = pd.DataFrame([effect(te, a_te, y_te, s) for s in [10, 30, 60, 120]])
print("\nSuppression rule 'zero fwd bytes + idle >= N s' on VALIDATION:")
print(rows_va.round(5).to_string(index=False))

ok = rows_va[rows_va["tp_lost_%"] <= MAX_TP_LOSS]
if ok.empty:
    print("\nNo cutoff keeps real-alert loss under the limit - rule not adopted.")
else:
    best = int(ok.sort_values("fp_removed", ascending=False).iloc[0]["idle_s"])
    print(f"\nChosen on validation: idle >= {best} s.  TEST result:")
    print(rows_te[rows_te["idle_s"] == best].round(5).to_string(index=False))
    pat_te = pattern(te, best)
    lost = a_te & (y_te != 0) & pat_te
    print("  real alerts lost per family:",
          pd.Series(np.array(CLASSES)[y_te[lost]]).value_counts().to_dict())

# %% Data-driven: small tree separating false ALERTs from real ones (validation)
Xa = va[XFEATS][a_va].astype("float32")
ya = (y_va[a_va] == 0).astype(int)                      # 1 = false alert
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50,
                              class_weight="balanced", random_state=SEED).fit(Xa, ya)
print("\nDepth-3 tree on validation alerts (class 1 = false ALERT):")
print(export_text(tree, feature_names=XFEATS, decimals=1))