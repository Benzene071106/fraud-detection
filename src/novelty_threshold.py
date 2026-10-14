# Conservative threshold for the novelty detector: instead of the exact 2.00% quantile of validation benign
# (which lands at ~1.96-2.04% on new data by chance), aim the validation FPR below the budget by a two-sample
# binomial margin so that new data stays under the budget with ~95% probability. Re-exports models/novelty/novelty_detector.joblib; test read once.
# Usage: python -m src.novelty_threshold [--budget 0.02]
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=float, default=0.02)
ap.add_argument("--conf", type=float, default=0.95)
args = ap.parse_args()
CL = ["Benign", "BruteForce", "DoS", "DDoS", "Bot", "Web", "Infiltration"]
A = np.load(ROOT / "models/sweep/aux.npz", allow_pickle=True)          # same 70/15/15 split order
va_y, te_y = A["y_va"], A["y_te"]
det_path = ROOT / "models/novelty/novelty_detector.joblib"
det = joblib.load(det_path)
S = {m: np.load(ROOT / f"reports/novelty/{m}_v2.npz") for m in det["members"]}
assert all(len(S[m]["va"]) == len(va_y) for m in S), "score files do not match the validation split"


def pct(s, ref):
    return np.searchsorted(ref, s) / len(ref)


ref = {m: np.sort(S[m]["va"][va_y == 0]) for m in S}
sv = np.max([pct(S[m]["va"], ref[m]) for m in S], axis=0)
st = np.max([pct(S[m]["te"], ref[m]) for m in S], axis=0)
vb = np.sort(sv[va_y == 0])[::-1]                    # benign combined scores, descending
n = len(vb)
upper = lambda k: beta.ppf(args.conf, k + 1, n - k) if k < n else 1.0
# Both the validation estimate AND the new data's FPR are noisy (binomial). For the new-data FPR to stay under the
# budget with probability `conf`, the validation FPR must sit z * sqrt(2) * sigma below it (two-sample margin).
from scipy.stats import norm
sigma = np.sqrt(args.budget * (1 - args.budget) / n)
target = args.budget - norm.ppf(args.conf) * np.sqrt(2) * sigma
k = int(np.floor(target * n))
thr = float(vb[k]) if k < n else 1.0


def rates(flag, yy):
    return {"benign_fpr%": round(100 * flag[yy == 0].mean(), 3),
            **{c: round(100 * flag[yy == i].mean(), 2) for i, c in enumerate(CL[1:], 1)}}


rows = []
for name, t in [("exact quantile (old)", det["threshold"]), (f"two-sample {args.conf:.0%} margin, budget {args.budget:.0%}", thr)]:
    rows.append({"threshold rule": name, "threshold": round(t, 5),
                 "val_fpr%": round(100 * (sv[va_y == 0] > t).mean(), 3), **{f"test_{k}": v for k, v in rates(st > t, te_y).items()}})
tab = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(f"validation benign n={n:,}; target validation FPR {target:.3%} (budget {args.budget:.0%} minus two-sample margin); flags k={k:,}")
print(tab.to_string(index=False))
det.update({"threshold": thr, "threshold_rule": f"validation FPR <= budget - z*sqrt(2)*sigma (one-sided {args.conf:.0%})",
            "budget": args.budget, "ref_sorted": ref})
joblib.dump(det, det_path)
tab.to_csv(ROOT / "reports" / "novelty_threshold.csv", index=False)
print(f"Re-exported {det_path.relative_to(ROOT)} with the conservative threshold")
