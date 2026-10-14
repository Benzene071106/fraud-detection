# Improve the robust (no header length) model: train a few variants on the GPU, save validation/test
# probabilities for each, so the decision layer can be compared on VALIDATION (test is only read at the end).
# Checkpointed: a variant whose probabilities already exist is skipped. Ctrl+C / crash -> just rerun.
# Usage: python -m src.sweep_nohdr
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.features import RATIO_FEATURES, add_ratio_features  # noqa: E402

DATA, RAW_DIR, MODELS = ROOT / "data/processed/all_days.parquet", ROOT / "data/raw", ROOT / "models"
OUT = MODELS / "sweep"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
CLASSES = ["Benign", "BruteForce", "DoS", "DDoS", "Bot", "Web", "Infiltration"]
CMAP = {c: i for i, c in enumerate(CLASSES)}
INF = CMAP["Infiltration"]
DROP = ["Init Fwd Win Bytes", "Init Bwd Win Bytes", "Fwd Seg Size Min", "Fwd Header Length", "Bwd Header Length"]
BASE_TAG = "xgb_v1_nofp_nohdr_clean_r1500"

# name: (max_depth, learning_rate, max_rounds, use_ratio_features)   -- run in this order
VARIANTS = {
    "base":        None,                      # existing model, predict only
    "ratio_d8":    (8, 0.1, 2000, True),
    "d10":         (10, 0.1, 2000, False),
    "ratio_d10":   (10, 0.1, 2000, True),
    "ratio_d8_lr05": (8, 0.05, 4000, True),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


t_all = time.time()
log("Loading data...")
df = pd.read_parquet(DATA)
meta = ["Label", "family", "day"]
raw_feats = [c for c in df.columns if c not in meta]
feats = [c for c in raw_feats if df[c].nunique() > 1 and c not in DROP]
y = df["family"].map(CMAP).values
idx = np.arange(len(df))
tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)

# identical cleaning to train_xgb --clean
def row_hash(frame):
    return pd.util.hash_pandas_object(frame[raw_feats].astype("float64"), index=False).values
bh = np.unique(np.concatenate([row_hash(r[r["Label"].astype(str) == "Benign"])
                               for r in (pd.read_parquet(p) for p in sorted(RAW_DIR.glob("Infil*.parquet")))]))
inf_tr = tr[y[tr] == INF]
tr = np.setdiff1d(tr, inf_tr[np.isin(row_hash(df.iloc[inf_tr]), bh)])
log(f"Train={len(tr):,} Val={len(va):,} Test={len(te):,}  base features={len(feats)}")

# --- shared arrays for the evaluator (rule R1, Isolation Forest scores, labels) - once
aux_path = OUT / "aux.npz"
if not aux_path.exists():
    log("Computing Rule R1 + Isolation Forest scores for val/test (1-2 min)...")
    bounds = pd.read_json(MODELS / "rule_r1_ssh_bounds_v2.json")
    min_match = json.load(open(MODELS / "hybrid_v2_config.json"))["min_match"]
    icfg = json.load(open(MODELS / "iforest_v2_config.json"))
    iforest = joblib.load(MODELS / "iforest_v2.joblib")

    def rule(fr):
        cnt = np.zeros(len(fr), dtype=int)
        for f in bounds.index:
            cnt += fr[f].between(bounds.loc[f, "min"], bounds.loc[f, "max"]).values.astype(int)
        cnt[fr["Protocol"].values != 6] = 0
        return cnt >= min_match

    def ifs(fr):
        Xi = fr[icfg["features"]].astype("float64")
        if icfg["use_log"]:
            Xi = np.sign(Xi) * np.log1p(np.abs(Xi))
        return -iforest.score_samples(Xi)

    V, T = df.iloc[va], df.iloc[te]
    np.savez(aux_path, y_va=y[va], y_te=y[te], rule_va=rule(V), rule_te=rule(T), if_va=ifs(V), if_te=ifs(T),
             lab_te=df["Label"].astype(str).values[te], day_te=df["day"].astype(str).values[te])
    del V, T
    log("Saved aux.npz")

X = df[feats].astype("float32")
del df
Xr = None
counts = np.bincount(y[tr], minlength=len(CLASSES))
w = np.sqrt(counts[0] / counts)[y[tr]]

for name, cfg in VARIANTS.items():
    pfile = OUT / f"{name}_P.npz"
    if pfile.exists():
        log(f"[skip] {name}: already done")
        continue
    t0 = time.time()
    if cfg is None:
        m = XGBClassifier()
        m.load_model(MODELS / f"{BASE_TAG}.json")
        cols, Xuse = json.load(open(MODELS / f"{BASE_TAG}_config.json"))["features"], X
        info = {"source": BASE_TAG}
    else:
        depth, lr, rounds, ratio = cfg
        if ratio and Xr is None:
            Xr = add_ratio_features(X)
        Xuse = Xr if ratio else X
        cols = feats + (RATIO_FEATURES if ratio else [])
        log(f"[run ] {name}: depth={depth} lr={lr} max_rounds={rounds} features={len(cols)}")
        m = XGBClassifier(objective="multi:softprob", n_estimators=rounds, max_depth=depth, learning_rate=lr,
                          subsample=0.8, colsample_bytree=0.8, tree_method="hist", device="cuda",
                          eval_metric="mlogloss", early_stopping_rounds=50, random_state=SEED, n_jobs=-1)
        m.fit(Xuse[cols].iloc[tr], y[tr], sample_weight=w, eval_set=[(Xuse[cols].iloc[va], y[va])], verbose=200)
        m.set_params(device="cpu")
        m.save_model(OUT / f"{name}.json")
        info = {"depth": depth, "lr": lr, "best_iteration": int(m.best_iteration), "ratio_features": ratio}
        json.dump({"features": cols, "classes": CLASSES, "ratio_features": ratio, **info},
                  open(OUT / f"{name}_config.json", "w"), indent=2)
    P_va = m.predict_proba(Xuse[cols].iloc[va]).astype("float32")
    P_te = m.predict_proba(Xuse[cols].iloc[te]).astype("float32")
    np.savez(pfile, P_va=P_va, P_te=P_te)
    log(f"[done] {name} in {(time.time() - t0) / 60:.1f} min  {info}")

log(f"All variants finished in {(time.time() - t_all) / 60:.1f} min")
