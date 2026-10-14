# Calibrated risk score: make "risk 70" mean "about 70% of flows like this are real attacks" at a realistic
# attack base rate (default 1%), instead of 100 x the raw XGBoost probability (which saturates at 99-100).
#
# Score  : logistic regression on [logit(1 - P_benign), Isolation-Forest percentile, Rule R1 match]
#          fitted on the first half of the VALIDATION split, then isotonic calibration on the second half.
# Prior  : calibrated on the validation class mix, re-weighted at run time to RISK_BASE_RATE (odds shift).
# Check  : TEST split only, re-weighted to the base rate: reliability table, Brier score, and how well the
#          risk ranks real attacks above false alarms AMONG ALERTS (the analyst's actual question).
# Needs  : models/sweep/{aux,base_P}.npz from src.sweep_nohdr (deployed model's validation/test probabilities)
# Usage  : python -m src.calibrate_risk [--base-rate 0.01]
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.detector import XGB_TAG  # noqa: E402
import src.eval_sweep as ES        # noqa: E402  (same decision rules as the API)

ap = argparse.ArgumentParser()
ap.add_argument("--base-rate", type=float, default=0.01)
args = ap.parse_args()
PI = args.base_rate
MOD = ROOT / "models"
A = ES.A
P = np.load(MOD / "sweep" / "base_P.npz")
REF = np.load(MOD / "iforest_v2_val_benign_scores.npy")
dec = json.loads((MOD / f"{XGB_TAG}_decision.json").read_text())
CL = ES.CLASSES
THR = np.array([dec["thresholds"][c] for c in CL] + [dec.get("pool_threshold", 1.01)])
MODE = dec["mode"]


def feats(Pm, if_raw, rule):
    s = np.clip(1 - Pm[:, 0], 1e-6, 1 - 1e-6)
    pct = np.searchsorted(REF, if_raw, side="right") / len(REF)
    return np.column_stack([np.log(s / (1 - s)), pct, rule.astype(float)]), pct


def shift(p, prior_from, prior_to):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    odds = p / (1 - p) * (prior_to / (1 - prior_to)) / (prior_from / (1 - prior_from))
    return odds / (1 + odds)


def weights(y, pi):
    att = y != 0
    return np.where(att, pi / att.mean(), (1 - pi) / (1 - att.mean()))


# ---------------- fit on validation (half for the logistic model, half for isotonic)
Xv, pct_v = feats(P["P_va"], A["if_va"], A["rule_va"])
yv = (A["y_va"] != 0).astype(int)
rng = np.random.default_rng(0)
half = rng.random(len(yv)) < 0.5
lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xv[half], yv[half])
z = lr.decision_function(Xv[~half])
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(z, yv[~half])
val_prior = float(yv[~half].mean())
xs, ys = iso.X_thresholds_, iso.y_thresholds_
if len(xs) > 400:                                  # compress the step function for the JSON file
    keep = np.unique(np.linspace(0, len(xs) - 1, 400).astype(int))
    xs, ys = xs[keep], ys[keep]


def risk(Xm, pi):
    p = np.interp(lr.decision_function(Xm), xs, ys)
    return 100 * shift(p, val_prior, pi)


# ---------------- old risk (current API formula) for comparison
def old_risk(Pm, if_raw, rule, pct):
    alert, _, cls = ES.decide(Pm, THR, rule, MODE)
    if_flag = pct >= 1 - 0.01
    review = ~alert & (if_flag | (cls == ES.INF))
    k = np.where(cls != 0, cls, Pm[:, 1:].argmax(1) + 1)
    p_att = Pm[np.arange(len(Pm)), k]
    r = np.maximum.reduce([np.where(alert | review, p_att, 0), np.where(rule, 0.95, 0), np.where(if_flag, 0.6 * pct, 0)])
    return 100 * r, alert, review


# ---------------- evaluate on TEST at the deployment base rate
Xt, pct_t = feats(P["P_te"], A["if_te"], A["rule_te"])
yt = (A["y_te"] != 0).astype(int)
w = weights(A["y_te"], PI)
new = risk(Xt, PI)
old, alert, review = old_risk(P["P_te"], A["if_te"], A["rule_te"], pct_t)
old_prob = old / 100


def brier(pred, y, w):
    return float(np.average((pred - y) ** 2, weights=w))


def ece(pred, y, w, bins=10):
    idx = np.minimum((pred * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += w[m].sum() * abs(np.average(pred[m], weights=w[m]) - np.average(y[m], weights=w[m]))
    return tot / w.sum()


rows = []
edges = [0, 1, 5, 10, 25, 50, 75, 90, 99, 100.01]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (new >= lo) & (new < hi)
    if m.any():
        rows.append({"risk band": f"{lo:g}-{min(hi, 100):g}", "test flows": int(m.sum()),
                     "mean predicted %": round(float(np.average(new[m], weights=w[m])), 2),
                     "observed attack % (at base rate)": round(100 * float(np.average(yt[m], weights=w[m])), 2)})
rel = pd.DataFrame(rows)
fl = alert | review
sub = alert                      # the analyst's question: which ALERTS are real?
summary = {
    "base_rate": PI,
    "brier_old": brier(old_prob, yt, w), "brier_new": brier(new / 100, yt, w),
    "ece_old": ece(old_prob, yt, w), "ece_new": ece(new / 100, yt, w),
    "alerts": int(sub.sum()),
    "alerts_with_risk>=99_old%": 100 * float((old[sub] >= 99).mean()),
    "alerts_with_risk>=99_new%": 100 * float((new[sub] >= 99).mean()),
    "alert_ranking_AP_old": float(average_precision_score(yt[sub], old[sub] + 1e-9 * new[sub])),
    "alert_ranking_AP_new": float(average_precision_score(yt[sub], new[sub])),
    "alert_ranking_AUC_old": float(roc_auc_score(yt[sub], old[sub])),
    "alert_ranking_AUC_new": float(roc_auc_score(yt[sub], new[sub])),
    "flagged_ranking_AUC_old": float(roc_auc_score(yt[fl], old[fl])),
    "flagged_ranking_AUC_new": float(roc_auc_score(yt[fl], new[fl])),
}
q = np.percentile
print(f"Deployed model {XGB_TAG}/{MODE}; base rate {PI:.1%}; validation prior {val_prior:.3f}")
print(f"Logistic weights [logit(1-P_benign), IF percentile, Rule R1]: {np.round(lr.coef_[0], 3).tolist()}, "
      f"intercept {lr.intercept_[0]:.3f}")
print("\nRELIABILITY on TEST (weighted to the base rate): predicted risk vs how often it really is an attack")
print(rel.to_string(index=False))
print("\nOld vs new risk")
for k, v in summary.items():
    print(f"  {k:<28} {v:.4f}" if isinstance(v, float) else f"  {k:<28} {v}")
print("\nRisk among ALERTS, true attacks vs false alarms (TEST, percentiles 10/50/90):")
for name, r in [("old", old), ("new", new)]:
    print(f"  {name}: attacks {np.round(q(r[sub & (yt == 1)], [10, 50, 90]), 1).tolist()}   "
          f"false alarms {np.round(q(r[sub & (yt == 0)], [10, 50, 90]), 1).tolist()}")

out = {"method": "logistic regression on [logit(1-P_benign), iforest_percentile, rule_r1] + isotonic calibration",
       "model": XGB_TAG, "decision": MODE, "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
       "iso_x": xs.tolist(), "iso_y": ys.tolist(), "val_prior": val_prior, "default_base_rate": PI,
       "test_check": summary}
(MOD / f"{XGB_TAG}_risk.json").write_text(json.dumps(out, indent=1))
rel.to_csv(ROOT / "reports" / "risk_calibration.csv", index=False)
(ROOT / "reports" / "risk_calibration_summary.json").write_text(json.dumps(summary, indent=1))
print(f"\nSaved models/{XGB_TAG}_risk.json, reports/risk_calibration.csv, reports/risk_calibration_summary.json")
