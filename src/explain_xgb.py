# Step 6: SHAP explanations for XGBoost alerts (Stage 2 final system, Variant B)
# Usage: python -m src.explain_xgb

# %% Imports + config
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

DATA = ROOT / "data" / "processed" / "all_days.parquet"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
XGB_TAG = "xgb_v1_nofp_clean_r1500"
SEED = 42
REVIEW_BUDGET = 0.01
PER_CLASS = 500                     # rows per class for the global SHAP analysis

with open(MODEL_DIR / f"{XGB_TAG}_config.json") as f:
    cfg = json.load(f)
CLASSES, XFEATS = cfg["classes"], cfg["features"]
K, F = len(CLASSES), len(cfg["features"])
C = {c: i for i, c in enumerate(CLASSES)}
BF, INF = C["BruteForce"], C["Infiltration"]
with open(MODEL_DIR / f"{XGB_TAG}_variantB_thresholds.json") as f:
    thr_map = json.load(f)
THR = np.array([thr_map[c] for c in CLASSES])

model = xgb.XGBClassifier()
model.load_model(MODEL_DIR / f"{XGB_TAG}.json")
booster = model.get_booster()
booster.set_param({"device": "cuda"})

# %% Load + identical split
df = pd.read_parquet(DATA)
y = df["family"].map(C).values
idx = np.arange(len(df))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=SEED)
va, te = df.iloc[val_idx], df.iloc[test_idx]
y_va, y_te = y[val_idx], y[test_idx]
lab_te = df["Label"].values[test_idx]

print("Predicting XGBoost on test...")
P_te = model.predict_proba(te[XFEATS].astype("float32"))

# %% Rule R1 v2 + Isolation Forest v2 (same as the final system)
bounds = pd.read_json(MODEL_DIR / "rule_r1_ssh_bounds_v2.json")
with open(MODEL_DIR / "hybrid_v2_config.json") as f:
    MIN_MATCH = json.load(f)["min_match"]
count = np.zeros(len(te), dtype=int)
for feat in bounds.index:
    count += te[feat].between(bounds.loc[feat, "min"], bounds.loc[feat, "max"]).values.astype(int)
count[te["Protocol"].values != 6] = 0
rule_te = count >= MIN_MATCH

with open(MODEL_DIR / "iforest_v2_config.json") as f:
    icfg = json.load(f)
iforest = joblib.load(MODEL_DIR / "iforest_v2.joblib")

def if_score(frame):
    Xi = frame[icfg["features"]].astype("float64")
    if icfg["use_log"]:
        Xi = np.sign(Xi) * np.log1p(np.abs(Xi))
    return -iforest.score_samples(Xi)

print("Scoring Isolation Forest (1-2 min)...")
if_val = np.sort(if_score(va[y_va == 0]))
if_pct = np.searchsorted(if_val, if_score(te), side="right") / len(if_val)
if_flag = if_pct >= 1 - REVIEW_BUDGET

# %% Final verdicts (Variant B, refined thresholds)
masked = np.where(P_te >= THR, P_te, -1.0)
masked[:, 0] = -0.5
cls = masked.argmax(axis=1)
cls = np.where(rule_te & (cls == 0), BF, cls)
alert = (cls != 0) & (cls != INF)
review = ~alert & (if_flag | (cls == INF))
verdict = np.where(alert, "ALERT", np.where(review, "REVIEW", "NORMAL"))
print("Verdict counts on test:", pd.Series(verdict).value_counts().to_dict())

# %% SHAP via XGBoost's exact TreeSHAP (GPU)
def contribs(frame):
    dm = xgb.DMatrix(frame[XFEATS].astype("float32"))
    c = booster.predict(dm, pred_contribs=True)
    return c.reshape(len(frame), K, F + 1) if c.ndim == 2 else c

# %% (1) Global: top features per attack family
rng = np.random.default_rng(SEED)
samp = np.concatenate([
    rng.choice(np.where(y_te == k)[0], min(PER_CLASS, int((y_te == k).sum())), replace=False)
    for k in range(K)
])
Xs = te.iloc[samp][XFEATS].astype("float32")
cs = contribs(te.iloc[samp])
ys = y_te[samp]

rows = []
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, k in zip(axes.ravel(), range(1, K)):
    imp = pd.Series(np.abs(cs[ys == k, k, :F]).mean(axis=0), index=XFEATS)
    imp = imp.sort_values(ascending=False).head(10)
    for rank, (feat, v) in enumerate(imp.items(), 1):
        rows.append({"class": CLASSES[k], "rank": rank, "feature": feat, "mean_abs_shap": round(float(v), 4)})
    imp[::-1].plot.barh(ax=ax, color="tab:red")
    ax.set_title(f"{CLASSES[k]}: top features (mean |SHAP|)")
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_xgb_per_class.png", dpi=150)
plt.close()

glob = pd.DataFrame(rows)
glob.to_csv(REPORT_DIR / "shap_xgb_top_features.csv", index=False)
print("\n(1) TOP 10 FEATURES PER ATTACK FAMILY (rank x family):")
print(glob.pivot(index="rank", columns="class", values="feature").to_string())

for k in range(1, K):
    shap.summary_plot(cs[:, k, :F], Xs, max_display=12, show=False)
    plt.title(f"SHAP - class {CLASSES[k]}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_xgb_beeswarm_{CLASSES[k]}.png", dpi=130, bbox_inches="tight")
    plt.close()
print(f"Saved per-family charts + beeswarm plots to {FIG_DIR}")

# %% (2) Local: analyst-style explanations
ben_med = va[y_va == 0][XFEATS].median()

def explain(i, title):
    k = int(cls[i]) if cls[i] != 0 else int(P_te[i, 1:].argmax() + 1)
    c = contribs(te.iloc[[i]])[0]
    top = pd.Series(c[k, :F], index=XFEATS).sort_values(ascending=False).head(5)
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"Verdict: {verdict[i]}   True label: {lab_te[i]}   Predicted family: {CLASSES[k]} "
          f"(p={P_te[i, k]:.3f})")
    print(f"Rule R1: {'MATCH' if rule_te[i] else 'no match'}   "
          f"Isolation Forest percentile: {if_pct[i]:.3f}{'  -> anomaly' if if_flag[i] else ''}")
    print(f"Top factors pushing toward {CLASSES[k]} (SHAP, log-odds):")
    for feat, v in top.items():
        print(f"  + {feat:<28} value={te.iloc[i][feat]:>14,.1f}   "
              f"normal median={ben_med[feat]:>12,.1f}   contribution={v:+.3f}")

def pick(mask, title):
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        print(f"\n(no example available for: {title})")
        return
    explain(int(rng.choice(idxs)), title)

print("\n(2) ANALYST EXPLANATIONS")
for fam in ["BruteForce", "DoS", "DDoS", "Bot", "Web"]:
    k = C[fam]
    pick(alert & (cls == k) & (y_te == k), f"Correct ALERT - {fam}")
pick(alert & (y_te == 0), "FALSE ALERT - benign flow flagged")
pick(review & (cls == INF) & (y_te == INF), "REVIEW - Infiltration (low-confidence class)")
pick(review & (cls == 0) & (y_te == C["Web"]), "REVIEW - Web attack caught only by Isolation Forest")