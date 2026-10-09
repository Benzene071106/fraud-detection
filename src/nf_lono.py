# NetFlow Step 2: leave-one-network-out ("new dataset" test) + benign/attack duplicate check
# Usage: python -m src.nf_lono

# %% Imports + config
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

DATA = ROOT / "data" / "processed" / "nfuq_sample.parquet"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
BASE_TAG = "nfuq_xgb_fixdur_noports_nofp"        # honest feature set (35 features)
SEED = 42
WRAP_MS = 4_294_967.296
MAX_TRAIN = 3_000_000
TARGETS = [0.005, 0.002]
CALIB_SHARE = 0.2                                 # share of the new network's benign used for calibration

with open(MODEL_DIR / f"{BASE_TAG}_config.json") as f:
    FEATS = json.load(f)["features"]

# %% Load + duration fix
df = pd.read_parquet(DATA)
dur = df["FLOW_DURATION_MILLISECONDS"]
band = dur.between(4_280_000, 4_294_968)
df["FLOW_DURATION_MILLISECONDS"] = np.where(band, WRAP_MS - dur, dur).astype("float32")

ALL = ["Benign"] + sorted(c for c in df["Attack"].unique() if c != "Benign")
y_all = df["Attack"].map({c: i for i, c in enumerate(ALL)}).values
src = df["Dataset"].values
NETS = sorted(np.unique(src))
X = df[FEATS].astype("float32")
short = lambda s: s.replace("NF-", "").replace("-v2", "")
print(f"Rows: {len(df):,}   Features: {len(FEATS)}   Networks: {[short(n) for n in NETS]}")

# %% (0) Duplicate check on the 35 honest features
print("\n(0) BENIGN FLOWS THAT ARE EXACT COPIES OF ATTACK FLOWS (35 features)")
h = pd.util.hash_pandas_object(df[FEATS], index=False).values
is_ben = y_all == 0
is_recon = df["Attack"].values == "Reconnaissance"
atk_h, recon_h, ben_h = np.unique(h[~is_ben]), np.unique(h[is_recon]), np.unique(h[is_ben])
for n in NETS:
    m = is_ben & (src == n)
    print(f"  {short(n):<16} benign rows: {m.sum():>9,}   identical to any attack: "
          f"{np.isin(h[m], atk_h).mean():.2%}   identical to Reconnaissance: {np.isin(h[m], recon_h).mean():.2%}")
print(f"  Reconnaissance rows identical to some benign flow: {np.isin(h[is_recon], ben_h).mean():.2%}")

# %% (1) Leave-one-network-out
rng = np.random.default_rng(SEED)
rows, per_class = [], []
for held in NETS:
    t0 = time.time()
    pool = np.where(src != held)[0]
    te = np.where(src == held)[0]
    present = sorted(set(y_all[pool].tolist()))          # classes seen in training
    lut = np.full(len(ALL), -1)
    lut[present] = np.arange(len(present))

    strat = df["Dataset"].values[pool] + "|" + df["Attack"].values[pool]
    tr, va = train_test_split(pool, test_size=0.15, stratify=strat, random_state=SEED)
    if len(tr) > MAX_TRAIN:
        tr = rng.choice(tr, MAX_TRAIN, replace=False)
    y_tr, y_va = lut[y_all[tr]], lut[y_all[va]]
    counts = np.bincount(y_tr, minlength=len(present))
    w = np.minimum(np.sqrt(counts[0] / counts), 50)[y_tr]

    model = XGBClassifier(
        objective="multi:softprob", n_estimators=600, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", device="cuda",
        eval_metric="mlogloss", early_stopping_rounds=30, random_state=SEED, n_jobs=2,
    )
    model.fit(X.iloc[tr], y_tr, sample_weight=w, eval_set=[(X.iloc[va], y_va)], verbose=False)

    s_va = 1 - model.predict_proba(X.iloc[va])[:, 0]
    P = model.predict_proba(X.iloc[te])
    s = 1 - P[:, 0]
    pred = np.array(present)[P.argmax(axis=1)]
    yt = y_all[te]
    ben = yt == 0

    ben_idx = np.where(ben)[0]
    is_cal = np.zeros(len(yt), dtype=bool)
    is_cal[rng.choice(ben_idx, int(CALIB_SHARE * len(ben_idx)), replace=False)] = True
    ev_ben = ben & ~is_cal

    row = {"held_out": short(held), "train_rows": len(tr), "test_rows": len(te),
           "classes_seen": len(present) - 1, "best_iter": int(model.best_iteration),
           "argmax_fpr": (pred[ev_ben] != 0).mean(), "argmax_det": (pred[~ben] != 0).mean()}
    for t in TARGETS:
        for name, thr in [("transfer", np.quantile(s_va[y_va == 0], 1 - t)),
                          ("site", np.quantile(s[is_cal], 1 - t))]:
            flag = s >= thr
            row[f"fpr_{name}_{t:.1%}"] = flag[ev_ben].mean()
            row[f"det_{name}_{t:.1%}"] = flag[~ben].mean()
    rows.append(row)

    flag = s >= np.quantile(s[is_cal], 1 - 0.005)         # site-calibrated, 0.5%
    for k in np.unique(yt[~ben]):
        mk = yt == k
        per_class.append({"held_out": short(held), "class": ALL[k], "rows": int(mk.sum()),
                          "seen_in_training": k in present,
                          "detected_site_0.5%": flag[mk].mean(),
                          "correct_class_argmax": (pred[mk] == k).mean() if k in present else np.nan})
    print(f"\n{short(held)} done in {(time.time() - t0) / 60:.1f} min  "
          f"(trained on {len(tr):,} rows, {len(present) - 1} attack classes seen)")

# %% Results
pd.set_option("display.width", 250)
res = pd.DataFrame(rows).set_index("held_out")
pc = pd.DataFrame(per_class)
print("\n===== LEAVE-ONE-NETWORK-OUT: OPERATING POINTS ON THE UNSEEN NETWORK =====")
print(res.round(4).to_string())
print("\n===== PER ATTACK CLASS ON THE UNSEEN NETWORK (site-calibrated 0.5% FPR) =====")
print(pc.round(4).to_string(index=False))
res.to_csv(REPORT_DIR / "nfuq_lono_summary.csv")
pc.to_csv(REPORT_DIR / "nfuq_lono_per_class.csv", index=False)
print(f"\nSaved to {REPORT_DIR}")