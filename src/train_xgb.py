# Stage 2: train XGBoost on all 10 days (7 attack families)
# Examples:
#   python -m src.train_xgb --nofp --clean --noproto --rounds 1000    (baseline without Protocol)
#   python -m src.train_xgb --nofp --clean --with2017 --rounds 1000   (+ CIC-IDS2017 web day, training only)

# %% Imports + config
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--nofp", action="store_true", help="exclude TCP-stack fingerprint features")
parser.add_argument("--clean", action="store_true",
                    help="drop Infiltration TRAINING rows that have an exact Benign twin")
parser.add_argument("--noproto", action="store_true", help="drop the Protocol feature")
parser.add_argument("--with2017", action="store_true",
                    help="add CIC-IDS2017 web-attack day to TRAINING only (implies --noproto)")
parser.add_argument("--rounds", type=int, default=600, help="max boosting rounds")
args, _ = parser.parse_known_args()
if args.with2017:
    args.noproto = True

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

DATA = ROOT / "data" / "processed" / "all_days.parquet"
WEB2017 = ROOT / "data" / "processed" / "web2017.parquet"
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
SEED = 42

TAG = "xgb_v1"
if args.nofp:
    TAG += "_nofp"
if args.clean:
    TAG += "_clean"
if args.with2017:
    TAG += "_w2017"
elif args.noproto:
    TAG += "_noproto"
if args.rounds != 600:
    TAG += f"_r{args.rounds}"

CLASSES = ["Benign", "BruteForce", "DoS", "DDoS", "Bot", "Web", "Infiltration"]
CMAP = {c: i for i, c in enumerate(CLASSES)}
FINGERPRINTS = ["Init Fwd Win Bytes", "Init Bwd Win Bytes", "Fwd Seg Size Min"]
INF = CLASSES.index("Infiltration")

# %% Load + features
df = pd.read_parquet(DATA)
meta = ["Label", "family", "day"]
raw_feats = [c for c in df.columns if c not in meta]
features = [c for c in raw_feats if df[c].nunique() > 1]
if args.nofp:
    features = [c for c in features if c not in FINGERPRINTS]
if args.noproto:
    features = [c for c in features if c != "Protocol"]
print(f"Model: {TAG}   Rows: {len(df):,}   Features: {len(features)}   "
      f"nofp={args.nofp} clean={args.clean} noproto={args.noproto} with2017={args.with2017} rounds={args.rounds}")

X = df[features].astype("float32")
y = df["family"].map(CMAP).values

# %% Stratified split 70 / 15 / 15 (identical in every run)
idx = np.arange(len(df))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=SEED)

# %% Optional: remove Infiltration training rows that are exact copies of Benign flows
if args.clean:
    def row_hash(frame):
        return pd.util.hash_pandas_object(frame[raw_feats].astype("float64"), index=False).values

    benign_hashes = []
    for p in sorted(RAW_DIR.glob("Infil*.parquet")):
        r = pd.read_parquet(p)
        benign_hashes.append(row_hash(r[r["Label"].astype(str) == "Benign"]))
    benign_hashes = np.unique(np.concatenate(benign_hashes))

    inf_train = train_idx[y[train_idx] == INF]
    is_twin = np.isin(row_hash(df.iloc[inf_train]), benign_hashes)
    train_idx = np.setdiff1d(train_idx, inf_train[is_twin])
    print(f"Clean: removed {is_twin.sum():,} of {len(inf_train):,} Infiltration training rows "
          f"({is_twin.mean():.1%}). Test set untouched.")

X_train, y_train = X.iloc[train_idx], y[train_idx]

# %% Optional: add CIC-IDS2017 web-attack day to TRAINING only
if args.with2017:
    w = pd.read_parquet(WEB2017)
    X_train = pd.concat([X_train, w[features].astype("float32")], ignore_index=True)
    y_train = np.concatenate([y_train, w["family"].map(CMAP).values])
    print(f"Added 2017 rows to TRAINING: {len(w):,} "
          f"({(w['family'] == 'Web').sum():,} Web, {(w['family'] == 'Benign').sum():,} Benign)")

print(f"Train={len(y_train):,}  Val={len(val_idx):,}  Test={len(test_idx):,}  (val/test are 2018 only)")

# %% Class weights: sqrt(benign / class count)
counts = np.bincount(y_train, minlength=len(CLASSES))
class_w = np.sqrt(counts[0] / counts)
print("Training rows per class:", dict(zip(CLASSES, counts.tolist())))
print("Class weights:", {c: round(float(wt), 2) for c, wt in zip(CLASSES, class_w)})
sample_w = class_w[y_train]

# %% Train
model = XGBClassifier(
    objective="multi:softprob",
    n_estimators=args.rounds,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="mlogloss",
    early_stopping_rounds=50 if args.rounds > 600 else 30,
    n_jobs=19,
    random_state=SEED,
)
model.fit(
    X_train, y_train,
    sample_weight=sample_w,
    eval_set=[(X.iloc[val_idx], y[val_idx])],
    verbose=100,
)
print("Best iteration:", model.best_iteration)

# %% Quick argmax check on 2018 test (full evaluation in xgb_thresholds.py)
pred = model.predict(X.iloc[test_idx])
y_test = y[test_idx]
print("\nPer-class report (argmax, 2018 test):")
print(classification_report(y_test, pred, target_names=CLASSES, digits=4, zero_division=0))
print(f"Benign FPR (argmax): {(pred[y_test == 0] != 0).mean():.4f}")

imp = pd.Series(model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
print("\nTop 15 features by gain:")
print(imp.head(15).round(1).to_string())

# %% Save
model.save_model(MODEL_DIR / f"{TAG}.json")
with open(MODEL_DIR / f"{TAG}_config.json", "w") as f:
    json.dump({"features": features, "classes": CLASSES, "exclude_fingerprints": args.nofp,
               "clean": args.clean, "noproto": args.noproto, "with2017": args.with2017,
               "rounds": args.rounds, "best_iteration": int(model.best_iteration)}, f, indent=2)
print(f"\nSaved {TAG} to {MODEL_DIR}")
print(f"Next: python -m src.xgb_thresholds --tag {TAG}")