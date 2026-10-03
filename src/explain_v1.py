# %% Imports + config
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(MODEL_DIR / "iforest_v1_config.json") as f:
    cfg = json.load(f)
with open(MODEL_DIR / "hybrid_v1_config.json") as f:
    hcfg = json.load(f)
iforest = joblib.load(MODEL_DIR / "iforest_v1.joblib")
bounds = pd.read_json(MODEL_DIR / "rule_r1_ssh_bounds.json")
FEATURES, USE_LOG, SEED = cfg["features"], cfg["use_log"], 42

def to_X(frame):
    X = frame[FEATURES].astype("float64")
    return np.sign(X) * np.log1p(np.abs(X)) if USE_LOG else X

def rule_hit(frame):
    hit = frame["Protocol"] == 6
    for feat in bounds.index:
        hit &= frame[feat].between(bounds.loc[feat, "min"], bounds.loc[feat, "max"])
    return hit.astype(int).values

# %% Load Wednesday + rebuild validation reference (same split as training)
wed = pd.read_parquet(DATA_DIR / cfg["data"])
wed["Label"] = wed["Label"].astype(str)
benign_idx = np.where(wed["Label"].values == "Benign")[0]
np.random.default_rng(SEED).shuffle(benign_idx)
n = len(benign_idx)
val = wed.iloc[benign_idx[int(0.6 * n): int(0.8 * n)]]
test_benign = wed.iloc[benign_idx[int(0.8 * n):]]
val_scores = np.sort(-iforest.score_samples(to_X(val)))
normal_median = val[FEATURES].median()

def iforest_pct(frame):
    s = -iforest.score_samples(to_X(frame))
    return np.searchsorted(val_scores, s, side="right") / len(val_scores)

# %% SHAP explainer + summary plot (1000 benign + 1000 SSH)
explainer = shap.TreeExplainer(iforest)
mix = pd.concat([
    val.sample(1000, random_state=SEED),
    wed[wed["Label"] == "SSH-Bruteforce"].sample(1000, random_state=SEED),
])
X_mix = to_X(mix)
sv = explainer.shap_values(X_mix)

# make sure "+" always means "pushes toward anomaly"
sign = 1 if np.corrcoef(sv.sum(axis=1), -iforest.score_samples(X_mix))[0, 1] > 0 else -1
print("SHAP sign correction:", sign)

shap.summary_plot(sign * sv, X_mix, max_display=15, show=False)
plt.title("Features driving the Isolation Forest anomaly score")
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_iforest_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# %% Analyst-style explanation for a single flow
def explain_flow(flow, title):
    pct = iforest_pct(flow)[0]
    rule = rule_hit(flow)[0]
    risk = 60 * rule + 40 * pct
    verdict = ("ALERT" if risk >= hcfg["alert_at"]
               else "REVIEW" if risk >= hcfg["review_at"] else "NORMAL")
    contrib = pd.Series(sign * explainer.shap_values(to_X(flow))[0], index=FEATURES)

    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    print(f"Verdict: {verdict}   Risk: {risk:.0f}/100   "
          f"IF percentile: {pct:.3f}   Rule R1: {'MATCH' if rule else 'no match'}")
    if rule:
        print("Rule R1 evidence:")
        for feat in bounds.index:
            print(f"  {feat:<26} {flow[feat].iloc[0]:>12,.0f}   "
                  f"in [{bounds.loc[feat, 'min']:,.0f}, {bounds.loc[feat, 'max']:,.0f}]")
    print("Top factors pushing toward anomaly (SHAP):")
    for feat, v in contrib.sort_values(ascending=False).head(5).items():
        print(f"  + {feat:<26} value={flow[feat].iloc[0]:>14,.1f}   "
              f"normal median={normal_median[feat]:>12,.1f}   contribution={v:+.3f}")

# %% Example 1: SSH brute force
explain_flow(wed[wed["Label"] == "SSH-Bruteforce"].sample(1, random_state=SEED),
             "Example 1 - SSH brute force (Wednesday)")

# %% Example 2: Slowloris flagged by Isolation Forest (never seen in training)
dos = pd.read_parquet(DATA_DIR / "DoS1-Thursday-15-02-2018_TrafficForML_CICFlowMeter.parquet")
dos["Label"] = dos["Label"].astype(str)
slow = dos[dos["Label"] == "DoS attacks-Slowloris"]
slow = slow[iforest_pct(slow) >= 0.99]
explain_flow(slow.sample(1, random_state=SEED),
             "Example 2 - Slowloris (unseen attack, flagged by Isolation Forest)")

# %% Example 3: Benign flow in REVIEW (false positive)
fp = test_benign[iforest_pct(test_benign) >= 0.99]
explain_flow(fp.sample(1, random_state=SEED),
             "Example 3 - Benign flow sent to REVIEW (false positive)")

print("\nSaved SHAP summary plot to", FIG_DIR / "shap_iforest_summary.png")