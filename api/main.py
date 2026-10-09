"""Threat Detection API - scores network flows with the hybrid detector."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.detector import HybridDetector, REVIEW_BUDGET

app = FastAPI(
    title="Threat Detection API",
    version="0.1.0",
    description="Hybrid network-flow detector: XGBoost + Rule R1 + Isolation Forest, with SHAP explanations.",
)
detector = HybridDetector()


class ScoreRequest(BaseModel):
    flows: list[dict[str, float]] = Field(..., min_length=1, max_length=5000)
    explain: bool = True
    top_k: int = Field(5, ge=1, le=20)


@app.get("/health")
def health():
    return {"status": "ok", "model": detector.xgb_tag}


@app.get("/model-info")
def model_info():
    return {
        "model": detector.xgb_tag,
        "classes": detector.classes,
        "thresholds": dict(zip(detector.classes, detector.thr.tolist())),
        "review_budget": REVIEW_BUDGET,
        "required_features": detector.required,
    }


@app.post("/score")
def score(req: ScoreRequest):
    missing = detector.missing(req.flows)
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})
    return {"model": detector.xgb_tag, "results": detector.score(req.flows, req.explain, req.top_k)}