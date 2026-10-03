# Step 5b: the full hybrid system (XGBoost + Rule R1 + Isolation Forest) on the standard test set
# Usage: python -m src.hybrid_final

# %% Imports + config
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score
from xgboost import XGBClassifier

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

DATA = ROOT / "data" / "processed" / "all_days.parquet"
MODEL_DIR = ROOT / "models"
OUT = ROOT / "reports" / "hybrid_final.csv"
XGB_TAG = "xgb_v1_nofp_clean_r1500"
SEED = 42
REVIEW_BUDGET = 0.01      # Isolation Forest REVIEW queue = top 1% of benign
BASE_RATE = 0.01          # realistic attack share for the precision estimate

with open(MODEL_DIR / f"{XGB_TAG}_config.json") as f:
    xcfg = json.load(f)
CLASSES, XFEATS = xcfg["classes"], xcfg["features"]
K = len(CLASSES)
BF, INF = CLASSES.index("BruteForce"), CLASSES.index("Infiltration")
with open(MODEL_DIR / f"{XGB_TAG}_thresholds.json") as f:
    thr_map = json.load(f)
THR = np.array([thr_map[c] for c in CLASSES])
xgb = XGBClassifier()
xgb.load_model(MODEL_DIR / f"{XGB_TAG}.json")
print(f"XGBoost: {XGB_TAG}   thresholds: {thr_map}")

# %% Load + identical split
df = pd.read_parquet(DATA)
y = df["family"].map({c: i for i, c in enumerate(CLASSES)}).values
labels = df["Label"].values
idx = np.arange(len(df))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=SEED)
te = df.iloc[test_idx]
y_te, lab_te = y[test_idx], labels[test_idx]
benign_te = y_te == 0
fam_te = np.array(CLASSES)[y_te]
print(f"Test rows: {len(test_idx):,}   Benign: {benign_te.sum():,}   Attacks: {(~benign_te).sum():,}")

# %% Component 1: XGBoost with per-class thresholds
P = xgb.predict_proba(te[XFEATS].astype("float32"))
masked = np.where(P >= THR, P, -1.0)
masked[:, 0] = -0.5
xgb_cls = masked.argmax(axis=1)

# %% Component 2: Rule R1 v2
bounds = pd.read_json(MODEL_DIR / "rule_r1_ssh_bounds_v2.json")
with open(MODEL_DIR / "hybrid_v2_config.json") as f:
    MIN_MATCH = json.load(f)["min_match"]
count = np.zeros(len(te), dtype=int)
for feat in bounds.index:
    count += te[feat].between(bounds.loc[feat, "min"], bounds.loc[feat, "max"]).values.astype(int)
count[te["Protocol"].values != 6] = 0
rule_hit = count >= MIN_MATCH

# %% Component 3: Isolation Forest v2 (REVIEW queue)
with open(MODEL_DIR / "iforest_v2_config.json") as f:
    icfg = json.load(f)
iforest = joblib.load(MODEL_DIR / "iforest_v2.joblib")

def if_score(frame):
    Xi = frame[icfg["features"]].astype("float64")
    if icfg["use_log"]:
        Xi = np.sign(Xi) * np.log1p(np.abs(Xi))
    return -iforest.score_samples(Xi)

print("Scoring Isolation Forest (1-2 min)...")
if_thr = np.quantile(if_score(df.iloc[val_idx[y[val_idx] == 0]]), 1 - REVIEW_BUDGET)
if_flag = if_score(te) >= if_thr

# %% Fusion
def run(inf_to_review):
    final_cls = xgb_cls.copy()
    rule_only = rule_hit & (xgb_cls == 0)
    final_cls[rule_only] = BF
    alert = final_cls != 0
    review_extra = np.zeros(len(final_cls), dtype=bool)
    if inf_to_review:
        review_extra = final_cls == INF
        alert = alert & ~review_extra
    review = ~alert & (if_flag | review_extra)
    verdict = np.where(alert, "ALERT", np.where(review, "REVIEW", "NORMAL"))
    alert_cls = np.where(alert, final_cls, 0)
    return verdict, alert_cls, rule_only

def report(name, verdict, alert_cls, rule_only):
    is_alert, is_review = verdict == "ALERT", verdict == "REVIEW"
    a_fpr, r_fpr = is_alert[benign_te].mean(), is_review[benign_te].mean()
    a_rec = is_alert[~benign_te].mean()
    prec_real = a_rec * BASE_RATE / (a_rec * BASE_RATE + a_fpr * (1 - BASE_RATE))

    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"ALERT  FPR: {a_fpr:.4%}  -> {a_fpr * 1e6:,.0f} false ALERTs per 1M benign flows")
    print(f"REVIEW FPR: {r_fpr:.4%}  -> {r_fpr * 1e6:,.0f} benign flows in REVIEW per 1M")
    print(f"Attack detection: ALERT {a_rec:.4f}   ALERT+REVIEW {(is_alert | is_review)[~benign_te].mean():.4f}")
    print(f"ALERT precision at a realistic {BASE_RATE:.0%} attack share: {prec_real:.4f}")

    keep = [i for i in range(K) if i != INF]
    m_ex = y_te != INF
    print(f"System (ALERT-level) Macro-F1: {f1_score(y_te, alert_cls, average='macro', zero_division=0):.4f}   "
          f"Macro-F1 excl. Inf: {f1_score(y_te[m_ex], alert_cls[m_ex], average='macro', labels=keep, zero_division=0):.4f}   "
          f"Macro recall: {recall_score(y_te, alert_cls, average='macro', zero_division=0):.4f}   "
          f"Weighted F1: {f1_score(y_te, alert_cls, average='weighted', zero_division=0):.4f}")
    print(classification_report(y_te, alert_cls, target_names=CLASSES, digits=4, zero_division=0))

    tab = pd.DataFrame({
        "family": fam_te, "label": lab_te,
        "xgb_alone": xgb_cls != 0, "ALERT": is_alert, "REVIEW": is_review,
        "ALERT_by_rule_only": is_alert & rule_only,
    })
    fam = tab.groupby("family").agg(
        rows=("ALERT", "size"), xgb_alone=("xgb_alone", "mean"), ALERT=("ALERT", "mean"),
        REVIEW=("REVIEW", "mean"), ALERT_by_rule_only=("ALERT_by_rule_only", "sum"),
    ).reindex(CLASSES)
    fam["ALERT+REVIEW"] = fam["ALERT"] + fam["REVIEW"]
    print("Per family (for Benign, the rates are false-positive rates):")
    print(fam.round(4).to_string())

    lab = tab[tab["family"] != "Benign"].groupby("label").agg(
        rows=("ALERT", "size"), ALERT=("ALERT", "mean"), REVIEW=("REVIEW", "mean"))
    lab["ALERT+REVIEW"] = lab["ALERT"] + lab["REVIEW"]
    print("\nPer original attack label:")
    print(lab.round(4).to_string())
    fam.insert(0, "variant", name)
    return fam

out = [
    report("VARIANT A: Infiltration -> ALERT", *run(inf_to_review=False)),
    report("VARIANT B: Infiltration -> REVIEW", *run(inf_to_review=True)),
]
pd.concat(out).to_csv(OUT)
print(f"\nSaved per-family results to {OUT}")