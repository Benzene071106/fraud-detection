# NetFlow track: XGBoost on the NF-UQ-NIDS-v2 sample (21 classes, GPU)
# Usage:
#   python -m src.nf_train_xgb --fixdur                      (duration bug fixed, all features)
#   python -m src.nf_train_xgb --fixdur --noports --nofp     (+ shortcut ablation)

# %% Imports + config
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from xgboost import XGBClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--fixdur", action="store_true", help="fix FLOW_DURATION 2^32-microsecond wraparound")
parser.add_argument("--noports", action="store_true", help="drop L4_SRC_PORT / L4_DST_PORT")
parser.add_argument("--nofp", action="store_true", help="drop TTL / TCP window fingerprints")
parser.add_argument("--rounds", type=int, default=1000)
args, _ = parser.parse_known_args()

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

DATA = ROOT / "data" / "processed" / "nfuq_sample.parquet"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)
SEED = 42
WEIGHT_CAP = 50
WRAP_MS = 4_294_967.296          # 2^32 microseconds, in milliseconds

with open(MODEL_DIR / "nfuq_sample_config.json") as f:
    meta = json.load(f)
TAG = ("nfuq_xgb" + ("_fixdur" if args.fixdur else "")
       + ("_noports" if args.noports else "") + ("_nofp" if args.nofp else ""))

# %% Load + duration fix
df = pd.read_parquet(DATA)
dur = df["FLOW_DURATION_MILLISECONDS"]
band = dur.between(4_280_000, 4_294_968)
print("FLOW_DURATION near 2^32 microseconds (wraparound), by source:")
print(band.groupby(df["Dataset"]).mean().round(4).to_string())
if args.fixdur:
    df["FLOW_DURATION_MILLISECONDS"] = np.where(band, WRAP_MS - dur, dur).astype("float32")
    print(f"Fixed FLOW_DURATION for {band.sum():,} rows (true duration = 2^32 us - recorded value)")

# %% Features
features = list(meta["features"])
if args.noports:
    features = [c for c in features if c not in meta["ports"]]
if args.nofp:
    features = [c for c in features if c not in meta["nf_fingerprints"]]
features = [c for c in features if df[c].nunique() > 1]

CLASSES = ["Benign"] + sorted(c for c in df["Attack"].unique() if c != "Benign")
K = len(CLASSES)
y = df["Attack"].map({c: i for i, c in enumerate(CLASSES)}).values
src = df["Dataset"].values
X = df[features].astype("float32")
print(f"\nModel: {TAG}   Rows: {len(df):,}   Features: {len(features)}   Classes: {K}")

# %% Split 70/15/15 stratified by (source x class) - identical in every run
strat = (df["Dataset"] + "|" + df["Attack"]).values
idx = np.arange(len(df))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=strat, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=strat[temp_idx], random_state=SEED)
print(f"Train={len(train_idx):,}  Val={len(val_idx):,}  Test={len(test_idx):,}")

counts = np.bincount(y[train_idx], minlength=K)
w_class = np.minimum(np.sqrt(counts[0] / counts), WEIGHT_CAP)

# %% Train on GPU
model = XGBClassifier(
    objective="multi:softprob", n_estimators=args.rounds, max_depth=8, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist", device="cuda",
    eval_metric="mlogloss", early_stopping_rounds=50, random_state=SEED, n_jobs=2,
)
model.fit(X.iloc[train_idx], y[train_idx], sample_weight=w_class[y[train_idx]],
          eval_set=[(X.iloc[val_idx], y[val_idx])], verbose=100)
print("Best iteration:", model.best_iteration)

# %% Evaluate on test
pred = model.predict(X.iloc[test_idx])
y_te, src_te = y[test_idx], src[test_idx]
print("\nPer-class report (argmax):")
print(classification_report(y_te, pred, target_names=CLASSES, digits=4, zero_division=0))
print(f"Macro-F1: {f1_score(y_te, pred, average='macro'):.4f}   "
      f"Weighted F1: {f1_score(y_te, pred, average='weighted'):.4f}")

ben = y_te == 0
print(f"\nBenign FPR overall: {(pred[ben] != 0).mean():.4%}")
print("Benign FPR per source network:")
print(pd.Series(pred[ben] != 0).groupby(src_te[ben]).mean().round(4).to_string())
print("Attack detection per source network:")
print(pd.Series(pred[~ben] != 0).groupby(src_te[~ben]).mean().round(4).to_string())

cm = confusion_matrix(y_te, pred, labels=list(range(K)))
pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(REPORT_DIR / f"{TAG}_confusion.csv")
off = [(cm[i, j], CLASSES[i], CLASSES[j]) for i in range(K) for j in range(K) if i != j and cm[i, j] > 0]
print("\nTop 15 confusions (count, true -> predicted):")
for n, t, p in sorted(off, reverse=True)[:15]:
    print(f"  {n:>7,}  {t} -> {p}")

imp = pd.Series(model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
print("\nTop 15 features by gain:")
print(imp.head(15).round(1).to_string())

# %% Save
model.save_model(MODEL_DIR / f"{TAG}.json")
with open(MODEL_DIR / f"{TAG}_config.json", "w") as f:
    json.dump({"features": features, "classes": CLASSES, "fixdur": args.fixdur,
               "noports": args.noports, "nofp": args.nofp,
               "best_iteration": int(model.best_iteration)}, f, indent=2)
print(f"\nSaved {TAG} -> {MODEL_DIR}")