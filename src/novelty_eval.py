# Unknown-attack detection: models trained on BENIGN traffic only (they never see any attack, so every attack
# family is "unknown" to them). Proper protocol: fit on TRAIN-split benign, threshold at 1% (and 0.5%) FPR on
# VALIDATION benign, report per-family detection on the TEST split.
# Checkpointed: each model's scores are saved to reports/novelty/<name>_v2.npz and skipped next time.
# Combination of detectors is CHOSEN ON VALIDATION (attacks + benign), test is read once at the end.
# The chosen detector is exported to models/novelty/ for the API (src/novelty_export happens here too).
# Usage: python -m src.novelty_eval                 (all models, uses all CPU cores)
#        python -m src.novelty_eval --fit-rows 60000 --quick
import argparse
import itertools
import json
import os
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, MOD = ROOT / "data/processed/all_days.parquet", ROOT / "reports/novelty", ROOT / "models/novelty"
CLASSES = ["Benign", "BruteForce", "DoS", "DDoS", "Bot", "Web", "Infiltration"]
ap = argparse.ArgumentParser()
ap.add_argument("--fit-rows", type=int, default=200000, help="benign TRAIN rows used to fit each model")
ap.add_argument("--knn-ref", type=int, default=100000, help="reference benign rows for kNN")
ap.add_argument("--quick", action="store_true", help="only 3 small models (smoke test)")
args = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True); MOD.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


log("Loading data + identical 70/15/15 split...")
cfg = json.loads((ROOT / "models/xgb_v1_nofp_nohdr_clean_r1500_config.json").read_text())
F = cfg["features"]
df = pd.read_parquet(DATA, columns=F + ["family"])
y = df["family"].map({c: i for i, c in enumerate(CLASSES)}).values
idx = np.arange(len(df))
tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=42)
X = df[F].to_numpy(dtype="float64")
del df
X = np.sign(X) * np.log1p(np.abs(X))                          # heavy-tailed counts/rates -> log scale
rng = np.random.default_rng(0)
tr_b = rng.permutation(tr[y[tr] == 0])[:args.fit_rows]
sc = StandardScaler().fit(X[tr_b])
Z = np.clip(sc.transform(X), -20, 20).astype("float32")
va_b = va[y[va] == 0]
va_y, te_y = y[va], y[te]
log(f"benign fit rows {len(tr_b):,}; validation benign {len(va_b):,}; test rows {len(te):,}; cores {os.cpu_count()}")

MODELS = {
    "iforest_300": lambda: IsolationForest(n_estimators=300, random_state=0, n_jobs=-1),
    "gmm_10": lambda: GaussianMixture(10, covariance_type="full", reg_covar=1e-3, random_state=0, max_iter=300),
    "gmm_20": lambda: GaussianMixture(20, covariance_type="full", reg_covar=1e-3, random_state=0, max_iter=300),
    "gmm_40": lambda: GaussianMixture(40, covariance_type="full", reg_covar=1e-3, random_state=0, max_iter=300),
    "gmm_80": lambda: GaussianMixture(80, covariance_type="full", reg_covar=1e-3, random_state=0, max_iter=300),
    "knn_1": lambda: NearestNeighbors(n_neighbors=1, n_jobs=-1),
    "knn_5": lambda: NearestNeighbors(n_neighbors=5, n_jobs=-1),
}
if args.quick:
    MODELS = {k: MODELS[k] for k in ["iforest_300", "gmm_10", "knn_1"]}


def scorer(name, m):
    if name.startswith("iforest"):
        return lambda z: -m.score_samples(z)
    if name.startswith("gmm"):
        return lambda z: -m.score_samples(z)
    return lambda z: np.concatenate([m.kneighbors(z[i:i + 20000])[0].mean(1) for i in range(0, len(z), 20000)])


def rates(flag, yy):
    out = {"benign_fpr%": round(100 * flag[yy == 0].mean(), 3)}
    for k, c in enumerate(CLASSES[1:], 1):
        out[c] = round(100 * flag[yy == k].mean(), 2)
    return out


def pct(s, ref):
    r = np.sort(ref)
    return np.searchsorted(r, s) / len(r)


for name, make in MODELS.items():
    ck = OUT / f"{name}_v2.npz"
    if ck.exists():
        log(f"[skip] {name}")
        continue
    t0 = time.time()
    log(f"[fit ] {name}")
    m = make()
    m.fit(Z[tr_b][:args.knn_ref] if name.startswith("knn") else Z[tr_b])
    sc_fn = scorer(name, m)
    np.savez_compressed(ck, va=sc_fn(Z[va]).astype("float32"), te=sc_fn(Z[te]).astype("float32"))
    joblib.dump({"scaler": sc, "features": F, "kind": name.split("_")[0],
                 "model": m if not name.startswith("knn") else None,
                 "reference": Z[tr_b][:args.knn_ref] if name.startswith("knn") else None,
                 "k": m.n_neighbors if name.startswith("knn") else None}, MOD / f"{name}.joblib")
    log(f"[done] {name} in {(time.time() - t0) / 60:.1f} min")

# ---- choose the combination on VALIDATION: max of per-detector percentiles (each calibrated on benign)
S = {n: np.load(OUT / f"{n}_v2.npz") for n in MODELS if (OUT / f"{n}_v2.npz").exists()}
P = {n: (pct(v["va"], v["va"][va_y == 0]), pct(v["te"], v["va"][va_y == 0])) for n, v in S.items()}
rows, best = [], {}
for budget in (0.01, 0.02):
    for k in (1, 2, 3):
        for combo in itertools.combinations(sorted(P), k):
            sv = np.max([P[c][0] for c in combo], axis=0)
            thr = float(np.quantile(sv[va_y == 0], 1 - budget))
            r = rates(sv > thr, va_y)
            r5 = [r[c] for c in CLASSES[1:6]]
            row = {"combo": "+".join(combo), "budget": budget, "threshold": thr, "val_mean5": np.mean(r5),
                   "val_min5": min(r5), **{f"val_{c}": r[c] for c in CLASSES[1:]}}
            rows.append(row)
            if budget not in best or row["val_mean5"] > best[budget]["val_mean5"]:
                best[budget] = row
res = []
for budget, b in best.items():                          # TEST read once, for the chosen combos only
    combo = b["combo"].split("+")
    st = np.max([P[c][1] for c in combo], axis=0)
    res.append({"budget": f"{budget:.0%}", "combo": b["combo"], **rates(st > b["threshold"], te_y)})
pd.DataFrame(rows).to_csv(ROOT / "reports" / "novelty_validation.csv", index=False)
tab = pd.DataFrame(res)
tab.to_csv(ROOT / "reports" / "novelty_eval.csv", index=False)
pd.set_option("display.width", 250)
print("\nChosen on VALIDATION (best mean detection over BruteForce/DoS/DDoS/Bot/Web), TEST results:")
print(tab.to_string(index=False))

# ---- export the 2%-budget detector for the API (REVIEW 'Unknown anomaly')
b = best[0.02]
combo = b["combo"].split("+")
export = {"members": combo, "threshold": b["threshold"], "budget": 0.02,
          "benign_val_scores": {c: np.sort(S[c]["va"][va_y == 0]).astype("float32") for c in combo}}
joblib.dump(export, MOD / "novelty_detector.joblib")
log(f"Exported models/novelty/novelty_detector.joblib  ({b['combo']}, 2% budget)")
