"""Hybrid detector: XGBoost (Variant B thresholds) + Rule R1 v2 + Isolation Forest v2, with SHAP reasons."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
XGB_TAG = "xgb_v1_nofp_clean_r1500"
REVIEW_BUDGET = 0.01


class HybridDetector:
    def __init__(self, xgb_tag: str = XGB_TAG):
        self.xgb_tag = xgb_tag
        with open(MODEL_DIR / f"{xgb_tag}_config.json") as f:
            cfg = json.load(f)
        self.classes = cfg["classes"]
        self.xfeats = cfg["features"]
        self.K, self.F = len(self.classes), len(self.xfeats)
        self.BF = self.classes.index("BruteForce")
        self.INF = self.classes.index("Infiltration")

        with open(MODEL_DIR / f"{xgb_tag}_variantB_thresholds.json") as f:
            tm = json.load(f)
        self.thr = np.array([tm[c] for c in self.classes])
        self.model = xgb.XGBClassifier()
        self.model.load_model(MODEL_DIR / f"{xgb_tag}.json")
        self.booster = self.model.get_booster()

        self.bounds = pd.read_json(MODEL_DIR / "rule_r1_ssh_bounds_v2.json")
        with open(MODEL_DIR / "hybrid_v2_config.json") as f:
            self.min_match = json.load(f)["min_match"]

        with open(MODEL_DIR / "iforest_v2_config.json") as f:
            icfg = json.load(f)
        self.if_feats, self.if_log = icfg["features"], icfg["use_log"]
        self.iforest = joblib.load(MODEL_DIR / "iforest_v2.joblib")
        self.if_ref = np.load(MODEL_DIR / "iforest_v2_val_benign_scores.npy")
        with open(MODEL_DIR / "benign_medians.json") as f:
            self.medians = json.load(f)

        self.required = sorted(set(self.xfeats) | set(self.if_feats) | set(self.bounds.index) | {"Protocol"})

    def missing(self, flows):
        req, miss = set(self.required), set()
        for fl in flows:
            miss |= req - fl.keys()
        return sorted(miss)

    def score(self, flows, explain=True, top_k=5):
        df = pd.DataFrame(flows)[self.required].astype("float64")
        X = df[self.xfeats].astype("float32")

        # 1) XGBoost with per-class thresholds
        P = self.model.predict_proba(X)
        masked = np.where(P >= self.thr, P, -1.0)
        masked[:, 0] = -0.5
        cls = masked.argmax(axis=1)

        # 2) Rule R1 v2 (SSH session pattern)
        count = np.zeros(len(df), dtype=int)
        for feat in self.bounds.index:
            count += df[feat].between(self.bounds.loc[feat, "min"], self.bounds.loc[feat, "max"]).values.astype(int)
        count[df["Protocol"].values != 6] = 0
        rule = count >= self.min_match
        cls = np.where(rule & (cls == 0), self.BF, cls)

        # 3) Isolation Forest v2 percentile vs normal traffic
        Xi = df[self.if_feats]
        if self.if_log:
            Xi = np.sign(Xi) * np.log1p(np.abs(Xi))
        s = -self.iforest.score_samples(Xi)
        pct = np.searchsorted(self.if_ref, s, side="right") / len(self.if_ref)
        if_flag = pct >= 1 - REVIEW_BUDGET

        alert = (cls != 0) & (cls != self.INF)
        review = ~alert & (if_flag | (cls == self.INF))

        contrib = None
        if explain:
            c = self.booster.predict(xgb.DMatrix(X), pred_contribs=True)
            contrib = c.reshape(len(df), self.K, self.F + 1) if c.ndim == 2 else c

        results = []
        for i in range(len(df)):
            k = int(cls[i]) if cls[i] != 0 else int(P[i, 1:].argmax() + 1)
            if alert[i]:
                verdict, family = "ALERT", self.classes[k]
            elif review[i]:
                verdict = "REVIEW"
                family = self.classes[k] if cls[i] == self.INF else "Unknown anomaly"
            else:
                verdict, family = "NORMAL", "Benign"
            p_att = float(P[i, k])
            risk = 100 * max(p_att if verdict != "NORMAL" else 0.0,
                             0.95 if rule[i] else 0.0,
                             0.6 * float(pct[i]) if if_flag[i] else 0.0)
            out = {
                "verdict": verdict,
                "family": family,
                "risk_score": int(round(risk)),
                "components": {
                    "xgb_top_attack": self.classes[k],
                    "xgb_probability": round(p_att, 4),
                    "rule_r1_match": bool(rule[i]),
                    "iforest_percentile": round(float(pct[i]), 4),
                },
            }
            if explain:
                top = np.argsort(contrib[i, k, :self.F])[::-1][:top_k]
                out["reasons"] = [{
                    "feature": self.xfeats[j],
                    "value": float(X.iloc[i, j]),
                    "normal_median": float(self.medians.get(self.xfeats[j], float("nan"))),
                    "contribution": round(float(contrib[i, k, j]), 3),
                } for j in top]
            results.append(out)
        return results