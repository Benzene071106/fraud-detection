"""Hybrid detector: XGBoost (Variant B thresholds) + Rule R1 v2 + Isolation Forest v2, with SHAP reasons."""
from pathlib import Path
import json
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from api.features import RATIO_FEATURES, RATIO_INPUTS, add_ratio_features
from api.mitre import map_flow

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
# Override with the XGB_TAG env var, e.g. XGB_TAG=xgb_v1_nofp_nohdr_clean_r1500
XGB_TAG = os.getenv("XGB_TAG", "xgb_v1_nofp_nohdr_clean_r1500")
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

        self.ratio = bool(cfg.get("ratio_features", False))   # engineered ratios computed here, not sent by clients
        self.DOS, self.DDOS = self.classes.index("DoS"), self.classes.index("DDoS")
        # Decision rule: "<tag>_decision.json" ({"mode": "flood", "thresholds": {...}}) if present, else Variant B.
        # flood = DoS and DDoS probabilities are pooled before thresholding (the model often splits a flood
        # between the two classes, so neither alone crosses its threshold).
        dfile = MODEL_DIR / f"{xgb_tag}_decision.json"
        if dfile.exists():
            with open(dfile) as f:
                dec = json.load(f)
            self.mode, tm = dec["mode"], dec["thresholds"]
        else:
            with open(MODEL_DIR / f"{xgb_tag}_variantB_thresholds.json") as f:
                self.mode, tm = "perclass", json.load(f)
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

        model_inputs = (set(self.xfeats) - set(RATIO_FEATURES)) | (set(RATIO_INPUTS) if self.ratio else set())
        self.required = sorted(model_inputs | set(self.if_feats) | set(self.bounds.index) | {"Protocol"})

    def missing(self, flows):
        req, miss = set(self.required), set()
        for fl in flows:
            miss |= req - fl.keys()
        return sorted(miss)

    def score(self, flows, explain=True, top_k=5):
        df = pd.DataFrame(flows)[self.required].astype("float64")
        X = (add_ratio_features(df) if self.ratio else df)[self.xfeats].astype("float32")

        # 1) XGBoost with per-class thresholds (flood mode: DoS+DDoS pooled, assigned to the larger of the two)
        P = self.model.predict_proba(X)
        Q = P.copy()
        if self.mode == "flood":
            pool = P[:, self.DOS] + P[:, self.DDOS]
            dos_top = P[:, self.DOS] >= P[:, self.DDOS]
            Q[:, self.DOS] = np.where(dos_top, pool, 0.0)
            Q[:, self.DDOS] = np.where(dos_top, 0.0, pool)
        masked = np.where(Q >= self.thr, Q, -1.0)
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
            p_att = float(Q[i, k]) if Q[i, k] > 0 else float(P[i, k])
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
            out["mitre"] = (map_flow(family, verdict, df.iloc[i].to_dict(), p_att, bool(rule[i]))
                            if verdict != "NORMAL" else [])
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