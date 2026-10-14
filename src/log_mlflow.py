# Log the project's experiments to MLflow (local: mlflow.db + mlruns/ in the project folder).
#   - one run per (model variant x decision rule) from reports/sweep_validation.csv (+ sweep_test.csv)
#   - the deployed model: parameters, test metrics, files, and the XGBoost booster registered as
#     "threat-detector-xgb" in the model registry
#   - drift run (reports/drift/summary.csv) and incident-level eval (reports/incident_eval.csv)
# Idempotent: a run whose name already exists is skipped (use --force to log again).
# Usage:  python -m src.log_mlflow            then   mlflow ui --backend-store-uri sqlite:///mlflow.db
import argparse
import json
import re
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true")
ap.add_argument("--store", default=str(ROOT), help="folder for mlflow.db and mlruns/ (default: project root)")
args = ap.parse_args()

import mlflow  # noqa: E402

STORE = Path(args.store).resolve()
mlflow.set_tracking_uri(f"sqlite:///{(STORE / 'mlflow.db').as_posix()}")
EXP = "threat-detection"
if mlflow.get_experiment_by_name(EXP) is None:
    mlflow.create_experiment(EXP, artifact_location=(STORE / "mlruns").as_uri())
mlflow.set_experiment(EXP)
REP, MOD = ROOT / "reports", ROOT / "models"
from api.detector import XGB_TAG  # noqa: E402  (deployed model tag)

existing = set(mlflow.search_runs(experiment_names=[EXP], output_format="pandas").get("tags.mlflow.runName", []))


def clean(name):
    return re.sub(r"[^0-9A-Za-z_\-. /]", "", name.replace("%", "_pct")).strip("_ ")


def metrics(row, prefix):
    return {clean(f"{prefix}{k}"): float(v) for k, v in row.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and pd.notna(v)}


def start(name):
    if name in existing and not args.force:
        print(f"[skip] {name}")
        return None
    print(f"[log ] {name}")
    return mlflow.start_run(run_name=name)


# 1) sweep: model variants x decision rules
val = REP / "sweep_validation.csv"
test = pd.read_csv(REP / "sweep_test.csv") if (REP / "sweep_test.csv").exists() else pd.DataFrame()
if val.exists():
    for _, r in pd.read_csv(val).iterrows():
        run = start(f"sweep/{r['variant']}/{r['mode']}")
        if run is None:
            continue
        with run:
            cfgf = MOD / "sweep" / f"{r['variant']}_config.json"
            cfg = json.loads(cfgf.read_text()) if cfgf.exists() else {"source": "existing base model"}
            mlflow.set_tags({"stage": "sweep", "variant": r["variant"], "decision": r["mode"],
                             "selection": "validation only; FPR <= 0.2%, max macro-F1 excl. Infiltration"})
            mlflow.log_params({"variant": r["variant"], "decision": r["mode"], "thresholds": r["thr"],
                               **{k: v for k, v in cfg.items() if k in ("depth", "lr", "best_iteration",
                                                                       "ratio_features", "source")}})
            mlflow.log_metrics(metrics(r.drop(["variant", "mode", "thr"]), "val_"))
            if not test.empty:
                t = test[(test["variant"] == r["variant"]) & (test["mode"] == r["mode"])]
                if len(t):
                    mlflow.log_metrics(metrics(t.iloc[0].drop(["variant", "mode", "thr"]), "test_"))

# 2) deployed model + registry
run = start(f"deployed/{XGB_TAG}")
if run is not None:
    with run:
        cfg = json.loads((MOD / f"{XGB_TAG}_config.json").read_text())
        dec_f = MOD / f"{XGB_TAG}_decision.json"
        dec = json.loads(dec_f.read_text()) if dec_f.exists() else {"mode": "perclass"}
        mlflow.set_tags({"stage": "deployed", "dataset": "CSE-CIC-IDS2018 (10 days, stratified 70/15/15)"})
        mlflow.log_params({"model": XGB_TAG, "n_features": len(cfg["features"]), "classes": ",".join(cfg["classes"]),
                           "best_iteration": cfg.get("best_iteration"), "exclude_fingerprints": cfg.get("exclude_fingerprints"),
                           "no_header_lengths": cfg.get("nohdr"), "clean_infiltration": cfg.get("clean"),
                           "decision": dec["mode"], "thresholds": json.dumps(dec.get("thresholds")),
                           "pool_threshold": dec.get("pool_threshold")})
        if not test.empty:
            t = test[(test["variant"] == "base") & (test["mode"] == dec["mode"])]
            if len(t):
                mlflow.log_metrics(metrics(t.iloc[0].drop(["variant", "mode", "thr"]), "test_"))
        ci = REP / f"bootstrap_ci_final_{XGB_TAG}.csv"
        for f in [MOD / f"{XGB_TAG}_config.json", dec_f, MOD / f"{XGB_TAG}_variantB_thresholds.json", ci,
                  REP / f"incidents_final_{XGB_TAG}.csv", REP / "sweep_nohdr_validation.txt"]:
            if f.exists():
                mlflow.log_artifact(str(f), artifact_path="files")
        booster = MOD / f"{XGB_TAG}.json"
        if booster.exists():
            import mlflow.xgboost
            import xgboost as xgb
            b = xgb.Booster()
            b.load_model(str(booster))
            try:
                mlflow.xgboost.log_model(b, name="model", registered_model_name="threat-detector-xgb")
            except TypeError:   # MLflow 2.x
                mlflow.xgboost.log_model(b, artifact_path="model", registered_model_name="threat-detector-xgb")
        else:
            print(f"  (model file {booster.name} not found; parameters and metrics logged without the model)")

# 3) drift
drift = REP / "drift" / "summary.csv"
if drift.exists():
    run = start("monitoring/drift")
    if run is not None:
        with run:
            d = pd.read_csv(drift)
            mlflow.set_tags({"stage": "monitoring", "method": "PSI>0.2 per feature on model-NORMAL flows; "
                                                              "window drift if >=50% features drift"})
            for _, r in d.iterrows():
                key = clean(r["window"].split(" (")[0]).replace(" ", "_")
                mlflow.log_metrics({f"{key}_drifted_features": float(r["drifted_features"]),
                                    f"{key}_attack_detect_pct": float(r["attack_detect%"]),
                                    f"{key}_dataset_drift": float(bool(r["dataset_drift"]))})
            for f in ["summary.csv", "top_features.csv", "summary.json"]:
                mlflow.log_artifact(str(REP / "drift" / f), artifact_path="drift")

# 4) incident-level evaluation
inc = REP / "incident_eval.csv"
if inc.exists():
    run = start("evaluation/incidents")
    if run is not None:
        with run:
            e = pd.read_csv(inc)
            mlflow.set_tags({"stage": "evaluation", "note": "replay with synthetic IPs: mechanism demo"})
            mlflow.log_metrics({"days": float(len(e)), "noise_in_queue_total": float(e["noise_in_queue"].sum()),
                                "real_in_queue_total": float(e["real_in_queue"].sum()),
                                "queue_precision_mean": float(e["precision"].mean())})
            mlflow.log_artifact(str(inc), artifact_path="incident_eval")

print(f"\nDone. View: mlflow ui --backend-store-uri sqlite:///{(STORE / 'mlflow.db').as_posix()}")
