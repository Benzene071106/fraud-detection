# Refinement: (1) Variant-B ALERT FPR < 0.2%, (2) incident-level evaluation, (3) bootstrap 95% CIs
# Usage: python -m src.refine_final

# %% Imports + config
from pathlib import Path
import itertools
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

DATA = ROOT / "data" / "processed" / "all_days.parquet"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
XGB_TAG = "xgb_v1_nofp_clean_r1500"
SEED = 42
REVIEW_BUDGET = 0.01
BASE_RATE = 0.01
TARGET_ALERT_FPR = 0.002
N_BOOT = 1000

with open(MODEL_DIR / f"{XGB_TAG}_config.json") as f:
    xcfg = json.load(f)
CLASSES, XFEATS = xcfg["classes"], xcfg["features"]
K = len(CLASSES)
C = {c: i for i, c in enumerate(CLASSES)}
BF, DOS, DDOS, BOT, WEB, INF = (C[c] for c in ["BruteForce", "DoS", "DDoS", "Bot", "Web", "Infiltration"])
with open(MODEL_DIR / f"{XGB_TAG}_thresholds.json") as f:
    thr_map = json.load(f)
BASE_THR = np.array([thr_map[c] for c in CLASSES])
xgb = XGBClassifier()
xgb.load_model(MODEL_DIR / f"{XGB_TAG}.json")

# %% Load + identical split
df = pd.read_parquet(DATA)
y = df["family"].map(C).values
idx = np.arange(len(df))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=SEED)
va, te = df.iloc[val_idx], df.iloc[test_idx]
y_va, y_te = y[val_idx], y[test_idx]
lab_te, day_te = df["Label"].values[test_idx], df["day"].values[test_idx]
benign_te = y_te == 0

print("Predicting XGBoost on validation + test...")
P_va = xgb.predict_proba(va[XFEATS].astype("float32"))
P_te = xgb.predict_proba(te[XFEATS].astype("float32"))

# %% Rule R1 v2 + Isolation Forest v2
bounds = pd.read_json(MODEL_DIR / "rule_r1_ssh_bounds_v2.json")
with open(MODEL_DIR / "hybrid_v2_config.json") as f:
    MIN_MATCH = json.load(f)["min_match"]

def rule(frame):
    count = np.zeros(len(frame), dtype=int)
    for feat in bounds.index:
        count += frame[feat].between(bounds.loc[feat, "min"], bounds.loc[feat, "max"]).values.astype(int)
    count[frame["Protocol"].values != 6] = 0
    return count >= MIN_MATCH

rule_va, rule_te = rule(va), rule(te)

with open(MODEL_DIR / "iforest_v2_config.json") as f:
    icfg = json.load(f)
iforest = joblib.load(MODEL_DIR / "iforest_v2.joblib")

def if_score(frame):
    Xi = frame[icfg["features"]].astype("float64")
    if icfg["use_log"]:
        Xi = np.sign(Xi) * np.log1p(np.abs(Xi))
    return -iforest.score_samples(Xi)

print("Scoring Isolation Forest (1-2 min)...")
if_thr = np.quantile(if_score(va[y_va == 0]), 1 - REVIEW_BUDGET)
if_flag_te = if_score(te) >= if_thr

# %% Helpers
def decide(P, thr):
    masked = np.where(P >= thr, P, -1.0)
    masked[:, 0] = -0.5
    return masked.argmax(axis=1)

def variant_b(P, thr, rule_arr):
    cls = decide(P, thr)
    cls = np.where(rule_arr & (cls == 0), BF, cls)
    alert = (cls != 0) & (cls != INF)              # Infiltration -> REVIEW, never ALERT
    return alert, np.where(alert, cls, 0), cls

def cm_metrics(y_true, pred):
    cm = np.bincount(y_true * K + pred, minlength=K * K).reshape(K, K)
    cm_ex = cm.copy()
    cm_ex[INF, :] = 0
    tp = np.diag(cm_ex).astype(float)
    col, row = cm_ex.sum(axis=0), cm_ex.sum(axis=1)
    p = np.divide(tp, col, out=np.zeros_like(tp), where=col > 0)
    r = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    keep = [i for i in range(K) if i != INF]
    return f[keep].mean(), r

def prec_at_base(rec, fpr):
    return rec * BASE_RATE / (rec * BASE_RATE + fpr * (1 - BASE_RATE)) if rec > 0 else 0.0

# %% (1) Tune thresholds for low ALERT FPR - VALIDATION only
GRID = {
    "DDoS": [0.8, 0.9, 0.95, 0.98, 0.99],
    "DoS": [0.5, 0.7, 0.9, 0.95, 0.98],
    "Bot": [0.7, 0.9, 0.95, 0.98],
    "BruteForce": [0.5, 0.9, 0.95],
}
rows = []
for combo in itertools.product(*GRID.values()):
    thr = BASE_THR.copy()
    for c, v in zip(GRID, combo):
        thr[C[c]] = v
    alert, alert_cls, _ = variant_b(P_va, thr, rule_va)
    mf1, rec = cm_metrics(y_va, alert_cls)
    rows.append({**dict(zip(GRID, combo)), "val_alert_fpr": alert[y_va == 0].mean(),
                 "macro_f1_ex_inf": mf1, "ddos_rec": rec[DDOS], "dos_rec": rec[DOS],
                 "bot_rec": rec[BOT], "bf_rec": rec[BF]})
res = pd.DataFrame(rows)
ok = res[res["val_alert_fpr"] <= TARGET_ALERT_FPR]
if ok.empty:
    print(f"\nNo setting reaches {TARGET_ALERT_FPR:.2%} on validation - using lowest FPR within 0.005 macro-F1 of best")
    ok = res[res["macro_f1_ex_inf"] >= res["macro_f1_ex_inf"].max() - 0.005]
    best = ok.sort_values("val_alert_fpr").iloc[0]
else:
    best = ok.sort_values(["macro_f1_ex_inf", "val_alert_fpr"], ascending=[False, True]).iloc[0]
print(f"\n(1) Tried {len(res)} threshold sets. Top 10 meeting the FPR target (validation):")
print(ok.sort_values(["macro_f1_ex_inf", "val_alert_fpr"], ascending=[False, True]).head(10).round(4).to_string(index=False))

NEW_THR = BASE_THR.copy()
for c in GRID:
    NEW_THR[C[c]] = best[c]
print("\nOld thresholds:", dict(zip(CLASSES, BASE_THR.round(3).tolist())))
print("New thresholds:", dict(zip(CLASSES, NEW_THR.round(3).tolist())))

# %% Test: before vs after
def evaluate(thr, name):
    alert, alert_cls, cls = variant_b(P_te, thr, rule_te)
    review = ~alert & (if_flag_te | (cls == INF))
    a_fpr, r_fpr = alert[benign_te].mean(), review[benign_te].mean()
    a_rec = alert[~benign_te].mean()
    mf1, _ = cm_metrics(y_te, alert_cls)
    print(f"\n----- {name} -----")
    print(f"ALERT FPR {a_fpr:.4%} ({a_fpr * 1e6:,.0f} per 1M benign)   REVIEW FPR {r_fpr:.4%}")
    print(f"Precision @ {BASE_RATE:.0%} attacks: {prec_at_base(a_rec, a_fpr):.4f}   "
          f"Detection ALERT {a_rec:.4f}   ALERT+REVIEW {(alert | review)[~benign_te].mean():.4f}   "
          f"Macro-F1 excl. Inf {mf1:.4f}")
    fam = pd.DataFrame({"family": np.array(CLASSES)[y_te], "ALERT": alert, "ALERT+REVIEW": alert | review})
    print(fam.groupby("family")[["ALERT", "ALERT+REVIEW"]].mean().reindex(CLASSES).round(4).to_string())
    return alert, review, alert_cls

evaluate(BASE_THR, "BEFORE (Variant B, current thresholds)")
alert, review, alert_cls = evaluate(NEW_THR, "AFTER (Variant B, refined thresholds)")

# %% (2) Incident-level evaluation: one incident = one attack type on one day
flag = alert | review
inc = pd.DataFrame({"day": day_te, "label": lab_te, "alert": alert, "flag": flag})
bg = inc[inc["label"] == "Benign"].groupby("day").agg(
    bg_alert=("alert", "mean"), bg_flag=("flag", "mean"))
att = inc[inc["label"] != "Benign"].groupby(["day", "label"]).agg(
    flows=("alert", "size"), alerts=("alert", "sum"), flags=("flag", "sum"),
    alert_rate=("alert", "mean"), flag_rate=("flag", "mean")).reset_index().merge(bg, on="day")
need = np.minimum(10, np.ceil(0.5 * att["flows"]))
att["lift_alert"] = att["alert_rate"] / att["bg_alert"].clip(lower=1e-6)
att["lift_flag"] = att["flag_rate"] / att["bg_flag"].clip(lower=1e-6)
att["detected_ALERT"] = (att["alerts"] >= need) & (att["lift_alert"] >= 5)
att["detected_ALERT+REVIEW"] = (att["flags"] >= need) & (att["lift_flag"] >= 5)

print("\n(2) INCIDENT-LEVEL DETECTION (incident = attack type x day)")
print(att[["day", "label", "flows", "alert_rate", "flag_rate", "lift_alert", "lift_flag",
           "detected_ALERT", "detected_ALERT+REVIEW"]].round(3).to_string(index=False))
print(f"\nIncidents detected: ALERT {att['detected_ALERT'].sum()}/{len(att)}   "
      f"ALERT+REVIEW {att['detected_ALERT+REVIEW'].sum()}/{len(att)}")
att.to_csv(REPORT_DIR / "incidents_final.csv", index=False)

# %% (3) Bootstrap 95% confidence intervals
rng = np.random.default_rng(SEED)
n = len(y_te)
stats = []
for _ in range(N_BOOT):
    s = rng.integers(0, n, n)
    yt, al, fl, ac = y_te[s], alert[s], flag[s], alert_cls[s]
    b = yt == 0
    a_fpr, a_rec = al[b].mean(), al[~b].mean()
    stats.append({
        "ALERT FPR": a_fpr,
        "Precision @1% attacks": prec_at_base(a_rec, a_fpr),
        "Detection ALERT+REVIEW": fl[~b].mean(),
        "Macro-F1 excl. Inf": cm_metrics(yt, ac)[0],
        "Web detection (A+R)": fl[yt == WEB].mean(),
        "Infiltration detection (A+R)": fl[yt == INF].mean(),
    })
ci = pd.DataFrame(stats).quantile([0.025, 0.5, 0.975]).T
ci.columns = ["low_2.5%", "median", "high_97.5%"]
print(f"\n(3) BOOTSTRAP 95% CONFIDENCE INTERVALS ({N_BOOT} resamples)")
print(ci.round(4).to_string())
ci.to_csv(REPORT_DIR / "bootstrap_ci_final.csv")

# %% Save refined thresholds
with open(MODEL_DIR / f"{XGB_TAG}_variantB_thresholds.json", "w") as f:
    json.dump(dict(zip(CLASSES, NEW_THR.tolist())), f, indent=2)
print(f"\nSaved refined thresholds -> {MODEL_DIR / f'{XGB_TAG}_variantB_thresholds.json'}")