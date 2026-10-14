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
            self.pool_thr = float(dec.get("pool_threshold", 1.01))
        else:
            with open(MODEL_DIR / f"{xgb_tag}_variantB_thresholds.json") as f:
                self.mode, tm = "perclass", json.load(f)
            self.pool_thr = 1.01
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
        # Calibrated risk (src.calibrate_risk): "risk 70" ~ 70% of such flows are attacks at RISK_BASE_RATE.
        rfile = MODEL_DIR / f"{xgb_tag}_risk.json"
        self.risk_cal = json.loads(rfile.read_text()) if rfile.exists() else None
        if self.risk_cal:
            self.risk_cal["iso_x"] = np.asarray(self.risk_cal["iso_x"])
            self.risk_cal["iso_y"] = np.asarray(self.risk_cal["iso_y"])
        self.base_rate = float(os.getenv("RISK_BASE_RATE", (self.risk_cal or {}).get("default_base_rate", 0.01)))

        # Unknown-attack detector (src.novelty_eval + src.novelty_threshold): GMM + kNN fitted on benign traffic
        # only; a flow scoring above the conservative threshold becomes REVIEW "Unknown anomaly".
        # It replaces the Isolation Forest as the REVIEW trigger; NOVELTY=0 switches back.
        self.nov = None
        nfile = MODEL_DIR / "novelty" / "novelty_detector.joblib"
        if nfile.exists() and os.getenv("NOVELTY", "1") != "0":
            from sklearn.neighbors import NearestNeighbors
            det = joblib.load(nfile)
            members = {}
            for m in det["members"]:
                b = joblib.load(MODEL_DIR / "novelty" / f"{m}.joblib")
                if b["kind"] == "knn":
                    b["model"] = NearestNeighbors(n_neighbors=b["k"]).fit(b["reference"])
                members[m] = b
            first = members[det["members"][0]]
            self.nov = {"members": members, "threshold": det["threshold"], "ref": det["ref_sorted"],
                        "scaler": first["scaler"], "features": first["features"], "rule": det.get("threshold_rule"),
                        "budget": det.get("budget")}

        self.required = sorted(model_inputs | set(self.if_feats) | set(self.bounds.index) | {"Protocol"})

    def missing(self, flows):
        req, miss = set(self.required), set()
        for fl in flows:
            miss |= req - fl.keys()
        return sorted(miss)

    def novelty(self, df):
        n = self.nov
        X = df[n["features"]].to_numpy(dtype="float64")
        Z = np.clip(n["scaler"].transform(np.sign(X) * np.log1p(np.abs(X))), -20, 20).astype("float32")
        pcts = []
        for name, b in n["members"].items():
            s = -b["model"].score_samples(Z) if b["kind"] == "gmm" else b["model"].kneighbors(Z)[0].mean(1)
            ref = n["ref"][name]
            pcts.append(np.searchsorted(ref, s) / len(ref))
        return np.max(pcts, axis=0)

    def calibrated_risk(self, P, pct, rule):
        c = self.risk_cal
        s = np.clip(1 - P[:, 0], 1e-6, 1 - 1e-6)
        z = (np.column_stack([np.log(s / (1 - s)), pct, rule.astype(float)]) @ np.asarray(c["coef"])
             + c["intercept"])
        p = np.clip(np.interp(z, c["iso_x"], c["iso_y"]), 1e-9, 1 - 1e-9)
        prior = c["val_prior"]
        odds = p / (1 - p) * (self.base_rate / (1 - self.base_rate)) / (prior / (1 - prior))
        return 100 * odds / (1 + odds)

    def score(self, flows, explain=True, top_k=5):
        df = pd.DataFrame(flows)[self.required].astype("float64")
        X = (add_ratio_features(df) if self.ratio else df)[self.xfeats].astype("float32")

        # 1) XGBoost with per-class thresholds.
        #    flood    : DoS+DDoS probabilities pooled, assigned to the larger of the two, then thresholded.
        #    flood_or : normal per-class decision, PLUS any still-benign flow whose pooled DoS+DDoS
        #               probability >= pool_threshold is assigned to the larger of the two.
        P = self.model.predict_proba(X)
        Q = P.copy()
        pool = P[:, self.DOS] + P[:, self.DDOS]
        top = np.where(P[:, self.DOS] >= P[:, self.DDOS], self.DOS, self.DDOS)
        if self.mode == "flood":
            Q[:, self.DOS] = np.where(top == self.DOS, pool, 0.0)
            Q[:, self.DDOS] = np.where(top == self.DDOS, pool, 0.0)
        masked = np.where(Q >= self.thr, Q, -1.0)
        masked[:, 0] = -0.5
        cls = masked.argmax(axis=1)
        if self.mode == "flood_or":
            add = (cls == 0) & (pool >= self.pool_thr)
            cls = np.where(add, top, cls)
            Q[add, top[add]] = pool[add]

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

        # 4) Unknown-attack detector (benign-only GMM + kNN), else the Isolation Forest flag
        nov_pct = self.novelty(df) if self.nov else None
        unknown = nov_pct > self.nov["threshold"] if self.nov else if_flag

        alert = (cls != 0) & (cls != self.INF)
        review = ~alert & (unknown | (cls == self.INF))
        cal_risk = self.calibrated_risk(P, pct, rule) if self.risk_cal else None

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
            if cal_risk is not None:
                risk = float(cal_risk[i])
            else:                                       # legacy heuristic (no calibration file)
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
                    "novelty_percentile": round(float(nov_pct[i]), 5) if nov_pct is not None else None,
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