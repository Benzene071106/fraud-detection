# NetFlow: per-site Isolation Forest - a label-free baseline trained on each network's own normal traffic
# Usage: python -u -m src.nf_site_iforest

# %% Imports + config
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

DATA = ROOT / "data" / "processed" / "nfuq_sample.parquet"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
BASE_TAG = "nfuq_xgb_fixdur_noports_nofp"
SEED = 42
WRAP_MS = 4_294_967.296
TARGETS = [0.01, 0.005, 0.002]
TRAIN_SHARE, CAL_SHARE = 0.4, 0.2              # of each network's benign flows

with open(MODEL_DIR / f"{BASE_TAG}_config.json") as f:
    FEATS = json.load(f)["features"]

# %% Load + duration fix
df = pd.read_parquet(DATA)
dur = df["FLOW_DURATION_MILLISECONDS"]
band = dur.between(4_280_000, 4_294_968)
df["FLOW_DURATION_MILLISECONDS"] = np.where(band, WRAP_MS - dur, dur).astype("float32")
src = df["Dataset"].values
is_ben = (df["Attack"] == "Benign").values
NETS = sorted(np.unique(src))
short = lambda s: s.replace("NF-", "").replace("-v2", "")

lono_path = REPORT_DIR / "nfuq_lono_summary.csv"
lono = pd.read_csv(lono_path, index_col=0) if lono_path.exists() else None

def tx(ix):
    Xi = df.iloc[ix][FEATS].astype("float64")
    return np.sign(Xi) * np.log1p(np.abs(Xi))

# %% One Isolation Forest per network, trained on that network's benign traffic only
rng = np.random.default_rng(SEED)
rows, per_class = [], []
for n in NETS:
    t0 = time.time()
    m = src == n
    ben = np.where(m & is_ben)[0]
    rng.shuffle(ben)
    n_tr, n_cal = int(TRAIN_SHARE * len(ben)), int(CAL_SHARE * len(ben))
    tr, cal, ev_b = ben[:n_tr], ben[n_tr:n_tr + n_cal], ben[n_tr + n_cal:]
    atk = np.where(m & ~is_ben)[0]

    ifo = IsolationForest(n_estimators=200, max_samples=4096, random_state=SEED, n_jobs=-1).fit(tx(tr))
    s_cal = -ifo.score_samples(tx(cal))
    s_b = -ifo.score_samples(tx(ev_b))
    s_a = -ifo.score_samples(tx(atk))

    row = {"network": short(n), "benign_train": len(tr), "benign_eval": len(ev_b), "attacks": len(atk),
           "auc": roc_auc_score(np.r_[np.zeros(len(s_b)), np.ones(len(s_a))], np.r_[s_b, s_a])}
    for t in TARGETS:
        thr = np.quantile(s_cal, 1 - t)
        row[f"fpr_{t:.1%}"] = (s_b >= thr).mean()
        row[f"det_{t:.1%}"] = (s_a >= thr).mean()
    if lono is not None and short(n) in lono.index:
        row["supervised_transfer_det_site_0.5%"] = lono.loc[short(n), "det_site_0.5%"]
    rows.append(row)

    thr05 = np.quantile(s_cal, 0.995)
    lab = df["Attack"].values[atk]
    for c in np.unique(lab):
        mc = lab == c
        per_class.append({"network": short(n), "class": c, "rows": int(mc.sum()),
                          "det_0.5%": (s_a[mc] >= thr05).mean()})
    print(f"{short(n)} done in {(time.time() - t0) / 60:.1f} min   AUC={row['auc']:.3f}   "
          f"det@0.5%={row['det_0.5%']:.2%}   fpr@0.5%={row['fpr_0.5%']:.2%}")

# %% Results
pd.set_option("display.width", 250)
res = pd.DataFrame(rows).set_index("network")
pc = pd.DataFrame(per_class)
print("\n===== PER-SITE ISOLATION FOREST (trained on each network's own benign traffic) =====")
print(res.round(4).to_string())
print("\n===== PER ATTACK CLASS (0.5% FPR) =====")
print(pc.round(4).to_string(index=False))
res.to_csv(REPORT_DIR / "nfuq_site_iforest_summary.csv")
pc.to_csv(REPORT_DIR / "nfuq_site_iforest_per_class.csv", index=False)
print(f"\nSaved to {REPORT_DIR}")