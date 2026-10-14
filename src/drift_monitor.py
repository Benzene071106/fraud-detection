# Drift monitoring demo: does the traffic the detector sees still look like what it was trained on,
# and when it does not, does detection actually get worse?
#
# Reference : 2018 validation split (what the model was tuned on)          -> sample of REF_ROWS flows
# Windows   : 2018 test days (same network, expect little drift)
#             CIC-IDS2017 Thursday web attacks + Wednesday DoS (different network/year, expect drift)
# Per window: feature drift on the traffic the model calls NORMAL (label-free, works in production: the attack
#             mix changes every day, so drift on ALL traffic mostly measures the attack mix, not the network),
#             the same on all traffic for comparison, prediction drift
#             (ALERT/REVIEW rates, Isolation-Forest score PSI), and real performance (labels available).
# Outputs   : reports/drift/summary.csv, top_features.csv, summary.json (read by the API/dashboard),
#             and an Evidently HTML report per window if evidently is installed.
# Usage     : python -m src.drift_monitor            (checkpointed: reference + 2017 windows cached in data/drift/)
#             python -m src.drift_monitor --no-html
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.detector import HybridDetector  # noqa: E402

DATA, CACHE, OUT = ROOT / "data/processed/all_days.parquet", ROOT / "data/drift", ROOT / "reports/drift"
RAW17 = ROOT / "data/raw_2017"
CLASSES = ["Benign", "BruteForce", "DoS", "DDoS", "Bot", "Web", "Infiltration"]
REF_ROWS, WIN_ROWS = 30000, 20000
PSI_DRIFT, SHARE_DRIFT = 0.2, 0.5      # feature drifts if PSI > 0.2; dataset drifts if >= 50% of features drift
RENAME_2017 = {
    "Total Length of Fwd Packets": "Fwd Packets Length Total", "Total Length of Bwd Packets": "Bwd Packets Length Total",
    "Min Packet Length": "Packet Length Min", "Max Packet Length": "Packet Length Max",
    "Average Packet Size": "Avg Packet Size", "Init_Win_bytes_forward": "Init Fwd Win Bytes",
    "Init_Win_bytes_backward": "Init Bwd Win Bytes", "act_data_pkt_fwd": "Fwd Act Data Packets",
    "min_seg_size_forward": "Fwd Seg Size Min",
}

ap = argparse.ArgumentParser()
ap.add_argument("--no-html", action="store_true")
args = ap.parse_args()
det = HybridDetector()
REQ = det.required
FEATS = [f for f in det.xfeats if f in REQ]          # model features that clients send
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ data
def reference():
    p = CACHE / "reference_2018_val.parquet"
    if p.exists():
        return pd.read_parquet(p)
    log("Building reference sample from the 2018 validation split (once)...")
    meta = pd.read_parquet(DATA, columns=["family", "day", "Label"])
    y = meta["family"].map({c: i for i, c in enumerate(CLASSES)}).values
    idx = np.arange(len(meta))
    _, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    va, _ = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=42)
    pick = np.sort(np.random.default_rng(0).choice(va, REF_ROWS, replace=False))
    pf, cols = pq.ParquetFile(DATA), {}
    for c in REQ:                                    # column by column: low memory
        cols[c] = pf.read(columns=[c]).column(0).to_numpy()[pick]
    ref = pd.DataFrame(cols).assign(Label=meta["Label"].values[pick], family=meta["family"].values[pick])
    ref.to_parquet(p, index=False)
    return ref


def protocol_2017(d):
    """CIC-IDS2017 CSVs have no Protocol column. TCP flows carry flags / a TCP window; others treated as UDP (17).
    This is an approximation and is reported as such."""
    tcp = ((d[["SYN Flag Count", "ACK Flag Count", "FIN Flag Count", "RST Flag Count", "PSH Flag Count"]].sum(axis=1) > 0)
           | (d["Init Fwd Win Bytes"] > 0) | (d["Init Bwd Win Bytes"] > 0))
    return np.where(tcp, 6, 17)


def load_2017(csv, label_map, name):
    p = CACHE / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    log(f"Preparing {name} from {csv} (once)...")
    d = pd.read_csv(RAW17 / csv, encoding="latin1", low_memory=False)
    d.columns = d.columns.str.strip()
    d = d.rename(columns=RENAME_2017)
    d["Protocol"] = protocol_2017(d)
    num = [c for c in REQ if c != "Protocol"]
    d[num] = d[num].apply(pd.to_numeric, errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=num).drop_duplicates(subset=num + ["Label"])
    lab = d["Label"].astype(str).str.strip()
    d["family"] = "Other"
    for key, fam in label_map.items():
        d.loc[lab.str.contains(key, case=False, regex=False), "family"] = fam
    d = d[d["family"] != "Other"].assign(Label=lab)
    # keep the day's attack share but cap the window size
    d = d.sample(min(WIN_ROWS, len(d)), random_state=0)[REQ + ["Label", "family"]]
    d.to_parquet(p, index=False)
    return d


def windows():
    w = {}
    days = sorted((ROOT / "data/eval_days").glob("*.parquet"))
    for name in ["Web2-Friday-23-02-2018", "DoS1-Thursday-15-02-2018", "Botnet-Friday-02-03-2018"]:
        f = ROOT / "data/eval_days" / f"{name}.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            w[f"2018 {name.split('-')[0]} (same network)"] = d.sample(min(WIN_ROWS, len(d)), random_state=0)
    if not days:
        log("data/eval_days missing: run python -m src.eval_incidents once to build it (2018 windows skipped)")
    w["2017 Thu WebAttacks (new network)"] = load_2017(
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        {"BENIGN": "Benign", "Brute Force": "Web", "XSS": "Web", "Sql": "Web"}, "cicids2017_thu_web")
    w["2017 Wed DoS (new network)"] = load_2017(
        "Wednesday-workingHours.pcap_ISCX.csv",
        {"BENIGN": "Benign", "Hulk": "DoS", "GoldenEye": "DoS", "slowloris": "DoS", "Slowhttptest": "DoS"},
        "cicids2017_wed_dos")
    return w


# ------------------------------------------------------------------ drift statistics
def psi(ref, cur, bins=10):
    """Population Stability Index with reference-quantile bins (ties collapse)."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:                                         # constant in reference
        return 0.0 if np.all(cur == ref[0]) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def score(df):
    res = []
    for s in range(0, len(df), 5000):
        res += det.score(df[REQ].iloc[s:s + 5000].astype(float).to_dict("records"), explain=False)
    return pd.DataFrame({"verdict": [r["verdict"] for r in res], "pred": [r["family"] for r in res],
                         "if_pct": [r["components"].get("iforest_percentile", 0.0) for r in res]})


def performance(df, pr):
    fam = df["family"].values
    flagged = pr["verdict"].isin(["ALERT", "REVIEW"]).values
    benign = fam == "Benign"
    out = {"benign_alert_rate%": 100 * (pr["verdict"].values[benign] == "ALERT").mean() if benign.any() else np.nan,
           "attack_detect%": 100 * flagged[~benign].mean() if (~benign).any() else np.nan}
    for f in sorted(set(fam) - {"Benign"}):
        out[f"{f}_detect%"] = 100 * flagged[fam == f].mean()
    return out


# ------------------------------------------------------------------ run
ref = reference()
log(f"Reference: {len(ref):,} flows from the 2018 validation split; model {det.xgb_tag}/{det.mode}")
ref_pred = score(ref)
rows, tops, html = [], [], 0
ref_normal = ref[ref_pred["verdict"].values == "NORMAL"]


def feature_drift(r, c, name):
    per = []
    for f in FEATS:
        a, b = r[f].astype(float).values, c[f].astype(float).values
        per.append({"window": name, "feature": f, "psi": psi(a, b), "ks": ks_2samp(a, b).statistic,
                    "ref_median": float(np.median(a)), "cur_median": float(np.median(b))})
    return pd.DataFrame(per)


for name, cur in windows().items():
    t0 = time.time()
    pr = score(cur)
    per = feature_drift(ref_normal, cur[pr["verdict"].values == "NORMAL"], name)     # label-free monitor
    per_all = feature_drift(ref, cur, name)
    share = (per["psi"] > PSI_DRIFT).mean()
    row = {"window": name, "flows": len(cur), "attack_share%": 100 * (cur["family"] != "Benign").mean(),
           "drifted_features": int((per["psi"] > PSI_DRIFT).sum()), "features": len(per),
           "share_drifted": round(share, 3), "dataset_drift": bool(share >= SHARE_DRIFT),
           "median_psi": round(per["psi"].median(), 3),
           "drifted_features_all_traffic": int((per_all["psi"] > PSI_DRIFT).sum()),
           "alert_rate%": 100 * (pr["verdict"] == "ALERT").mean(),
           "review_rate%": 100 * (pr["verdict"] == "REVIEW").mean(),
           "iforest_pct_psi": round(psi(ref_pred["if_pct"].values, pr["if_pct"].values), 3),
           **performance(cur, pr)}
    rows.append(row)
    tops.append(per.sort_values("psi", ascending=False).head(10))
    log(f"{name}: {row['drifted_features']}/{row['features']} features drifted "
        f"(dataset drift={row['dataset_drift']}), attack detection {row['attack_detect%']:.1f}%, "
        f"benign alert rate {row['benign_alert_rate%']:.2f}%  [{time.time() - t0:.0f}s]")
    if not args.no_html:
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset
            slug = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
            Report([DataDriftPreset()]).run(reference_data=ref_normal[FEATS],
                                            current_data=cur[pr["verdict"].values == "NORMAL"][FEATS]) \
                .save_html(str(OUT / f"evidently_{slug}.html"))
            html += 1
        except ImportError:
            pass

base = performance(ref, ref_pred)
summary = pd.DataFrame([{"window": "REFERENCE 2018 validation", "flows": len(ref),
                         "attack_share%": 100 * (ref["family"] != "Benign").mean(), "drifted_features": 0,
                         "features": len(FEATS), "share_drifted": 0.0, "dataset_drift": False, "median_psi": 0.0,
                         "drifted_features_all_traffic": 0,
                         "alert_rate%": 100 * (ref_pred["verdict"] == "ALERT").mean(),
                         "review_rate%": 100 * (ref_pred["verdict"] == "REVIEW").mean(), "iforest_pct_psi": 0.0,
                         **base}] + rows)
summary.round(3).to_csv(OUT / "summary.csv", index=False)
pd.concat(tops).round(4).to_csv(OUT / "top_features.csv", index=False)
(OUT / "summary.json").write_text(json.dumps({
    "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "model": f"{det.xgb_tag}/{det.mode}",
    "rules": {"feature_drift": f"PSI > {PSI_DRIFT}", "dataset_drift": f">= {SHARE_DRIFT:.0%} of features drifted"},
    "windows": json.loads(summary.round(3).to_json(orient="records")),
    "top_features": json.loads(pd.concat(tops).round(4).to_json(orient="records"))}, indent=1))
pd.set_option("display.width", 250)
print("\nFeature drift measured on flows the model calls NORMAL (label-free); all-traffic count for comparison")
print(summary[["window", "flows", "attack_share%", "drifted_features", "drifted_features_all_traffic",
                      "dataset_drift", "median_psi",
                      "alert_rate%", "benign_alert_rate%", "attack_detect%"]].round(2).to_string(index=False))
log(f"Saved {OUT / 'summary.csv'}, top_features.csv, summary.json"
    + (f", {html} Evidently HTML reports" if html else
       ("" if args.no_html else "  (no Evidently HTML: python -m pip install evidently)")))
