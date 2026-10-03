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
with open(MODEL_DIR / "hybrid_v1_config.json") as f:
    v1_cfg = json.load(f)
iforest = joblib.load(MODEL_DIR / "iforest_v1.joblib")
bounds_v1 = pd.read_json(MODEL_DIR / "rule_r1_ssh_bounds.json")

DATA_FILE = ROOT / "data" / "raw" / cfg["data"]
FEATURES = cfg["features"]
USE_LOG = cfg["use_log"]
SEED = 42
TARGET_FPR = 0.01
REVIEW_AT = 39.6

# %% Load + identical splits to v1
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

# %% Helpers
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

def rule_matches(frame, bounds):
    """How many of the 6 rule conditions a flow satisfies (0 if not TCP)."""
    count = np.zeros(len(frame), dtype=int)
    for feat in RULE_FEATURES:
        count += frame[feat].between(bounds.loc[feat, "min"],
                                     bounds.loc[feat, "max"]).values.astype(int)
    count[frame["Protocol"].values != 6] = 0
    return count

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

# %% Tune q + min_match + IF gate for best F1 (design attacks + validation benign ONLY)
Q_GRID = [0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01]
MATCH_GRID = [6, 5, 4]
GATE_GRID = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8]

n_design = len(design_ssh)
rows = []
for q in Q_GRID:
    b = make_bounds(design_ssh, q)
    d_m = rule_matches(design_ssh, b)
    v_m = rule_matches(val, b)
    for m in MATCH_GRID:
        for g in GATE_GRID:
            tp = int(((d_m >= m) & (design_pct >= g)).sum())
            fp = int(((v_m >= m) & (val_pct >= g)).sum())
            fn = n_design - tp
            rows.append({
                "q": q, "min_match": m, "gate": g,
                "recall": tp / n_design,
                "val_fpr": fp / len(val),
                "f1": 2 * tp / (2 * tp + fp + fn),
            })

res = pd.DataFrame(rows)
top = res[res["val_fpr"] <= TARGET_FPR].sort_values(["f1", "val_fpr"], ascending=[False, True])
print("Top 10 settings (design attacks + validation benign):")
print(top.head(10).round(4).to_string(index=False))

best = top.iloc[0]
Q, MIN_MATCH, GATE = float(best["q"]), int(best["min_match"]), float(best["gate"])
ALERT_AT = 60 + 40 * GATE
bounds = make_bounds(design_ssh, Q)
print(f"\nChosen: q={Q}  min_match={MIN_MATCH}/6  IF gate={GATE}  -> ALERT_AT={ALERT_AT:.1f}")
print(bounds.to_string())

# %% Why are design SSH flows still missed?
d_m = rule_matches(design_ssh, bounds)
missed = design_ssh[(d_m < MIN_MATCH) | (design_pct < GATE)]
print(f"\nMissed design SSH flows: {len(missed)} of {n_design}")
print("Conditions they fail:")
for feat in RULE_FEATURES:
    fails = int((~missed[feat].between(bounds.loc[feat, "min"], bounds.loc[feat, "max"])).sum())
    print(f"  {feat:<26} {fails}")
print("Blocked by IF gate:", int(((d_m >= MIN_MATCH) & (design_pct < GATE)).sum()))

# %% Final evaluation - v1 vs v2 on the same eval set
ev = df.iloc[eval_idx].copy()
ev["if_pct"] = iforest_pct(ev)
ev["rule_hit"] = (rule_matches(ev, bounds) >= MIN_MATCH).astype(int)
ev["risk"] = 60 * ev["rule_hit"] + 40 * ev["if_pct"]
ev["verdict"] = np.select(
    [ev["risk"] >= ALERT_AT, ev["risk"] >= REVIEW_AT],
    ["ALERT", "REVIEW"],
    default="NORMAL",
)
v1_alert = ((rule_matches(ev, bounds_v1) == 6) & (ev["if_pct"] >= v1_cfg["if_gate"])).astype(int)

y_ev = ev["is_attack"].values

def report(name, pred):
    pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_ev, pred).ravel()
    print(f"{name:<22} recall={tp/(tp+fn):.4f}  FPR={fp/(fp+tn):.4f}  "
          f"F1={f1_score(y_ev, pred):.4f}  TP={tp}  FP={fp}  FN={fn}")

print()
report("v1 Hybrid ALERT", v1_alert)
report("v2 Hybrid ALERT", ev["verdict"] == "ALERT")
report("v2 ALERT+REVIEW", ev["verdict"] != "NORMAL")
print("v2 risk ROC-AUC:", round(roc_auc_score(y_ev, ev["risk"]), 4))

ssh_only = ev["Label"] != "FTP-BruteForce"
f1_ssh = f1_score(y_ev[ssh_only], (ev["verdict"] == "ALERT").astype(int)[ssh_only])
print(f"v2 F1 excluding the 21 FTP rows: {f1_ssh:.4f}")

print("\nVerdict by label:")
print(pd.crosstab(ev["Label"], ev["verdict"]))

# %% Save
bounds.to_json(MODEL_DIR / "rule_r1_ssh_bounds_v2.json", indent=2)
with open(MODEL_DIR / "hybrid_v2_config.json", "w") as f:
    json.dump({"q": Q, "min_match": MIN_MATCH, "if_gate": GATE, "alert_at": ALERT_AT,
               "review_at": REVIEW_AT, "rule_features": RULE_FEATURES}, f, indent=2)
print("\nSaved v2 bounds + config to", MODEL_DIR)
