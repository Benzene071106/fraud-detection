# %% Imports + config
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

MODEL_DIR = ROOT / "models"
with open(MODEL_DIR / "iforest_v1_config.json") as f:
    cfg = json.load(f)
iforest = joblib.load(MODEL_DIR / "iforest_v1.joblib")

DATA_FILE = ROOT / "data" / "raw" / cfg["data"]
FEATURES = cfg["features"]
USE_LOG = cfg["use_log"]
SEED = 42
TARGET_FPR = 0.01
REVIEW_AT = 39.6   # Isolation Forest alone in top 1% of normal traffic

# %% Load + identical benign split to train_iforest.py
df = pd.read_parquet(DATA_FILE)
df["Label"] = df["Label"].astype(str)
df["is_attack"] = (df["Label"] != "Benign").astype(int)
y = df["is_attack"].values

rng = np.random.default_rng(SEED)
benign_idx = np.where(y == 0)[0]
attack_idx = np.where(y == 1)[0]
rng.shuffle(benign_idx)
n = len(benign_idx)
val_idx = benign_idx[int(0.6 * n): int(0.8 * n)]
test_benign_idx = benign_idx[int(0.8 * n):]

rng2 = np.random.default_rng(SEED + 1)
attack_idx = attack_idx.copy()
rng2.shuffle(attack_idx)
design_idx = attack_idx[: len(attack_idx) // 2]
eval_attack_idx = attack_idx[len(attack_idx) // 2:]
eval_idx = np.concatenate([test_benign_idx, eval_attack_idx])

# %% Rule + Isolation Forest helpers
RULE_FEATURES = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packets Length Total",
    "Bwd Packet Length Max",
    "Fwd Act Data Packets",
    "Flow Duration",
]
# Deliberately excluded: Init Fwd/Bwd Win Bytes, Fwd Seg Size Min (attacker TCP-stack fingerprints)

def make_bounds(attacks, q):
    return pd.DataFrame({
        "min": attacks[RULE_FEATURES].quantile(q),
        "max": attacks[RULE_FEATURES].quantile(1 - q),
    })

def rule_hit(frame, bounds):
    hit = frame["Protocol"] == 6
    for feat in RULE_FEATURES:
        hit &= frame[feat].between(bounds.loc[feat, "min"], bounds.loc[feat, "max"])
    return hit.astype(int).values

def to_X(frame):
    X = frame[FEATURES].astype("float64")
    return np.sign(X) * np.log1p(np.abs(X)) if USE_LOG else X

val = df.iloc[val_idx]
val_scores = np.sort(-iforest.score_samples(to_X(val)))

def iforest_pct(frame):
    s = -iforest.score_samples(to_X(frame))
    return np.searchsorted(val_scores, s, side="right") / len(val_scores)

design = df.iloc[design_idx]
design_ssh = design[design["Label"] == "SSH-Bruteforce"]
design_pct = iforest_pct(design_ssh)
val_pct = iforest_pct(val)

# %% Tune rule tightness (q) + IF gate together - design attacks + validation benign ONLY
Q_GRID = [0.001, 0.005, 0.01, 0.02, 0.05]
GATE_GRID = [0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8]

rows = []
for q in Q_GRID:
    b = make_bounds(design_ssh, q)
    d_hit = rule_hit(design_ssh, b)
    v_hit = rule_hit(val, b)
    for g in GATE_GRID:
        rows.append({
            "q": q,
            "gate": g,
            "design_recall": ((d_hit == 1) & (design_pct >= g)).mean(),
            "val_fpr": ((v_hit == 1) & (val_pct >= g)).mean(),
        })
res = pd.DataFrame(rows)
print("Tuning grid (design attacks / validation benign):")
print(res.round(4).to_string(index=False))

ok = res[res["val_fpr"] <= TARGET_FPR]
if ok.empty:
    raise SystemExit("\nNo setting meets the FPR target - send me the table above.")
best = ok.sort_values(["design_recall", "val_fpr"], ascending=[False, True]).iloc[0]
Q, GATE = float(best["q"]), float(best["gate"])
ALERT_AT = 60 + 40 * GATE
bounds = make_bounds(design_ssh, Q)

print(f"\nChosen: q={Q}  IF gate={GATE}  -> ALERT_AT={ALERT_AT:.1f}")
print("RULE R1 - SSH brute-force session pattern (TCP and all of):")
print(bounds.to_string())

# %% Final evaluation - eval set used ONCE
ev = df.iloc[eval_idx].copy()
ev["rule_hit"] = rule_hit(ev, bounds)
ev["if_pct"] = iforest_pct(ev)
ev["risk"] = 60 * ev["rule_hit"] + 40 * ev["if_pct"]
ev["verdict"] = np.select(
    [ev["risk"] >= ALERT_AT, ev["risk"] >= REVIEW_AT],
    ["ALERT", "REVIEW"],
    default="NORMAL",
)

y_ev = ev["is_attack"].values

def report(name, pred):
    pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_ev, pred).ravel()
    print(f"{name:<24} recall={tp/(tp+fn):.4f}  FPR={fp/(fp+tn):.4f}  "
          f"F1={f1_score(y_ev, pred):.4f}  TP={tp}  FP={fp}")

print()
report("IForest only (1% FPR)", ev["if_pct"] >= 0.99)
report("Rule R1 only", ev["rule_hit"])
report("Hybrid ALERT", ev["verdict"] == "ALERT")
report("Hybrid ALERT+REVIEW", ev["verdict"] != "NORMAL")
print("Risk score ROC-AUC:", round(roc_auc_score(y_ev, ev["risk"]), 4))

print("\nVerdict by label:")
print(pd.crosstab(ev["Label"], ev["verdict"]))

# %% Save
bounds.to_json(MODEL_DIR / "rule_r1_ssh_bounds.json", indent=2)
with open(MODEL_DIR / "hybrid_v1_config.json", "w") as f:
    json.dump({"q": Q, "if_gate": GATE, "alert_at": ALERT_AT,
               "review_at": REVIEW_AT, "rule_features": RULE_FEATURES}, f, indent=2)
print("\nSaved rule bounds + hybrid config to", MODEL_DIR)