"""Threat Detection API - scores flows, stores flagged events, groups them into incidents."""
import json
import os
from datetime import datetime
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, func, case

from api.detector import HybridDetector, REVIEW_BUDGET
from api.db import SessionLocal, init_db, Incident, Event
from api.incidents import store_flagged
from api import alerts
from api.mitre import coverage, TECHNIQUES

app = FastAPI(
    title="Threat Detection API",
    version="0.2.0",
    description="Hybrid flow detector (XGBoost + Rule R1 + Isolation Forest) with SHAP reasons and incident aggregation.",
)
detector = HybridDetector()
init_db()


class ScoreRequest(BaseModel):
    flows: list[dict[str, float]] = Field(..., min_length=1, max_length=5000)
    explain: bool = True
    top_k: int = Field(5, ge=1, le=20)


class IngestFlow(BaseModel):
    features: dict[str, float]
    src_ip: str = "unknown"
    dst_ip: str = "unknown"
    dst_port: int | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class IngestRequest(BaseModel):
    flows: list[IngestFlow] = Field(..., min_length=1, max_length=5000)


class Feedback(BaseModel):
    verdict: Literal["TP", "FP"]
    note: str | None = None


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model": detector.xgb_tag}


@app.get("/model-info")
def model_info():
    return {"model": detector.xgb_tag, "classes": detector.classes,
            "thresholds": dict(zip(detector.classes, detector.thr.tolist())),
            "review_budget": REVIEW_BUDGET, "required_features": detector.required}


@app.post("/score")
def score(req: ScoreRequest):
    missing = detector.missing(req.flows)
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})
    return {"model": detector.xgb_tag, "results": detector.score(req.flows, req.explain, req.top_k)}


@app.post("/ingest")
def ingest(req: IngestRequest, background: BackgroundTasks):
    feats = [f.features for f in req.flows]
    missing = detector.missing(feats)
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})
    results = detector.score(feats, explain=False)
    flagged = [i for i, r in enumerate(results) if r["verdict"] != "NORMAL"]
    if flagged:                                      # explain only flagged flows (faster)
        for i, r in zip(flagged, detector.score([feats[i] for i in flagged], explain=True, top_k=3)):
            results[i] = r
    with SessionLocal() as s:
        touched = store_flagged(s, req.flows, results)
        s.commit()
        to_notify = claim_notifications(s, touched)
    if to_notify:
        background.add_task(send_notifications, to_notify)     # after the response: ingestion never waits on Slack/SMTP
    return {"scored": len(results),
            "notifications_queued": len(to_notify),
            "alerts": sum(r["verdict"] == "ALERT" for r in results),
            "reviews": sum(r["verdict"] == "REVIEW" for r in results),
            "incidents_touched": touched}


# Noise = LOW incidents with too little evidence to be worth an analyst's time:
#   (a) at most NOISE_MAX_EVENTS flagged flows in total, or
#   (b) REVIEW-only (no ALERT from the classifier/rule) with fewer than NOISE_MIN_REVIEW_EVENTS flows, or
#   (c) scattered: every flow comes from a different source and there are fewer than NOISE_SCATTER_ALERTS
#       alerts -- no attacker is repeating anything, which is what isolated false positives look like.
# Exception: a Web or Bot ALERT is never noise. Those attacks are low-volume by nature (one SQL injection
# request is an incident) and the classifier almost never fires them on benign traffic.
# Noise stays in the database but is hidden from the analyst queue and from summaries by default.
NOISE_MAX_EVENTS = int(os.getenv("NOISE_MAX_EVENTS", "1"))
NOISE_MIN_REVIEW_EVENTS = int(os.getenv("NOISE_MIN_REVIEW_EVENTS", "3"))
NOISE_SCATTER_ALERTS = int(os.getenv("NOISE_SCATTER_ALERTS", "3"))
NEVER_NOISE_FAMILIES = ("Web", "Bot")


def noise_clause(max_events: int | None = None):
    n = NOISE_MAX_EVENTS if max_events is None else max_events
    sources = (select(func.count(func.distinct(Event.src_ip))).where(Event.incident_id == Incident.id)
               .correlate(Incident).scalar_subquery())
    tiny = Incident.event_count <= n
    weak_review = (Incident.alert_count == 0) & (Incident.event_count < NOISE_MIN_REVIEW_EVENTS)
    scattered = (sources >= Incident.event_count) & (Incident.alert_count < NOISE_SCATTER_ALERTS)
    protected = Incident.family.in_(NEVER_NOISE_FAMILIES) & (Incident.alert_count > 0)
    return (Incident.severity == "LOW") & ~protected & (tiny | weak_review | scattered)


SEV_RANK = case({"HIGH": 0, "MEDIUM": 1, "LOW": 2}, value=Incident.severity, else_=3)


def incident_techniques(inc, sources):
    """Roll per-flow technique hits up to the incident, adding incident-level evidence."""
    techs = json.loads(inc.techniques or "{}")
    out = []
    for tid, t in techs.items():
        share_high = t["HIGH"] / max(t["flows"], 1)
        conf = "HIGH" if share_high >= 0.5 else "MEDIUM" if (t["HIGH"] + t["MEDIUM"]) / max(t["flows"], 1) >= 0.5 else "LOW"
        evidence = [f"{t['flows']} flagged flows support this technique "
                    f"({t['HIGH']} high / {t['MEDIUM']} medium / {t['LOW']} low confidence)"]
        if tid in ("T1110", "T1110.001") and inc.alert_count >= 10:
            evidence.append(f"repeated attempts: {inc.alert_count} alerts against one target")
            conf = "HIGH" if conf != "LOW" else "MEDIUM"
        out.append({"technique_id": tid, "technique": t["technique"], "tactic": t["tactic"],
                    "confidence": conf, "flows": t["flows"], "evidence": evidence,
                    "url": f"https://attack.mitre.org/techniques/{tid.replace('.', '/')}/"})
    if inc.family == "DDoS" and sources is not None and sources >= 3:   # distribution is only visible per incident
        name, tactic, _ = TECHNIQUES["T1498"]
        out.append({"technique_id": "T1498", "technique": name, "tactic": tactic,
                    "confidence": "HIGH" if inc.alert_count >= 100 else "MEDIUM", "flows": inc.alert_count,
                    "evidence": [f"distributed: {sources} distinct sources attacking one target",
                                 f"{inc.alert_count} flood alerts in the incident"],
                    "url": "https://attack.mitre.org/techniques/T1498/"})
    return sorted(out, key=lambda x: -x["flows"])


def incident_dict(inc, sources=None):
    return {"id": inc.id, "family": inc.family, "target": inc.dst_ip, "severity": inc.severity,
            "status": inc.status, "first_seen": inc.first_seen.isoformat(), "last_seen": inc.last_seen.isoformat(),
            "events": inc.event_count, "alerts": inc.alert_count, "reviews": inc.review_count,
            "max_risk": inc.max_risk, "sources": sources, "analyst_verdict": inc.analyst_verdict,
            "notified_at": inc.notified_at.isoformat() if inc.notified_at else None,
            "mitre": incident_techniques(inc, sources)}



@app.get("/incidents")
def list_incidents(status: str | None = None, severity: str | None = None, limit: int = 50,
                   include_noise: bool = False, noise_max_events: int | None = None):
    with SessionLocal() as s:
        q = select(Incident)
        if not include_noise:
            q = q.where(~noise_clause(noise_max_events))
        if status:
            q = q.where(Incident.status == status.upper())
        if severity:
            q = q.where(Incident.severity == severity.upper())
        incs = s.scalars(q.order_by(SEV_RANK, Incident.last_seen.desc()).limit(limit)).all()
        srcs = dict(s.execute(
            select(Event.incident_id, func.count(func.distinct(Event.src_ip)))
            .where(Event.incident_id.in_([i.id for i in incs])).group_by(Event.incident_id)).all())
        by_sev = dict(s.execute(select(Incident.severity, func.count()).group_by(Incident.severity)).all())
        noise = s.scalar(select(func.count()).select_from(Incident)
                         .where(noise_clause(noise_max_events)))
        return {"total_incidents": sum(by_sev.values()), "by_severity": by_sev,
                "hidden_noise_incidents": 0 if include_noise else noise,
                "incidents": [incident_dict(i, srcs.get(i.id, 0)) for i in incs]}


@app.get("/incidents/{incident_id}")
def get_incident(incident_id: int, events: int = 20):
    with SessionLocal() as s:
        inc = s.get(Incident, incident_id)
        if inc is None:
            raise HTTPException(status_code=404, detail="incident not found")
        evs = s.scalars(select(Event).where(Event.incident_id == incident_id)
                        .order_by(Event.risk.desc(), Event.ts).limit(events)).all()
        sources = s.scalar(select(func.count(func.distinct(Event.src_ip))).where(Event.incident_id == incident_id))
        out = incident_dict(inc, sources)
        out["analyst_note"] = inc.analyst_note
        out["sample_events"] = [{"ts": e.ts.isoformat(), "src_ip": e.src_ip, "dst_port": e.dst_port,
                                 "verdict": e.verdict, "risk": e.risk, "xgb_prob": e.xgb_prob,
                                 "rule_match": e.rule_match, "iforest_percentile": e.if_pct,
                                 **_split_reasons(e.reasons)} for e in evs]
        return out


@app.post("/incidents/{incident_id}/feedback")
def feedback(incident_id: int, fb: Feedback):
    with SessionLocal() as s:
        inc = s.get(Incident, incident_id)
        if inc is None:
            raise HTTPException(status_code=404, detail="incident not found")
        inc.analyst_verdict, inc.analyst_note, inc.status = fb.verdict, fb.note, "CLOSED"
        s.commit()
        return incident_dict(inc)


def _split_reasons(raw):
    r = json.loads(raw)
    if isinstance(r, dict):
        return {"reasons": r.get("shap", []), "mitre": r.get("mitre", [])}
    return {"reasons": r, "mitre": []}                     # events stored before MITRE mapping


def claim_notifications(s, incident_ids):
    """Mark newly eligible incidents as notified (so each one is announced once) and return their payloads."""
    if not alerts.enabled() or not incident_ids:
        return []
    incs = s.scalars(select(Incident).where(Incident.id.in_(list(incident_ids)), Incident.notified_at.is_(None),
                                            Incident.status == "OPEN",
                                            Incident.severity.in_(alerts.eligible_severities()))).all()
    if not incs:
        return []
    now = datetime.now()
    srcs = dict(s.execute(select(Event.incident_id, func.count(func.distinct(Event.src_ip)))
                          .where(Event.incident_id.in_([i.id for i in incs])).group_by(Event.incident_id)).all())
    payloads = [incident_dict(i, srcs.get(i.id, 0)) for i in incs]
    for i in incs:
        i.notified_at = now
    s.commit()
    return payloads


def send_notifications(payloads):
    for inc in payloads:
        res = alerts.notify(inc)
        if res and all(v != "ok" for v in res.values()):          # every channel failed: allow a retry later
            with SessionLocal() as s:
                row = s.get(Incident, inc["id"])
                if row is not None:
                    row.notified_at = None
                    s.commit()


@app.get("/alerts/config")
def alerts_config():
    """Which notification channels are configured (no secrets returned)."""
    return alerts.config()


@app.post("/alerts/test")
def alerts_test():
    """Send a sample notification on every configured channel and report the result."""
    if not alerts.enabled():
        raise HTTPException(status_code=400, detail="no channel configured: set SLACK_WEBHOOK_URL and/or "
                                                    "SMTP_HOST + ALERT_EMAIL_TO")
    sample = {"id": 0, "severity": "HIGH", "family": "BruteForce", "target": "192.0.2.10", "sources": 1,
              "events": 120, "alerts": 120, "reviews": 0, "max_risk": 100,
              "first_seen": datetime.now().isoformat(timespec="seconds"),
              "last_seen": datetime.now().isoformat(timespec="seconds"),
              "mitre": [{"technique_id": "T1110.001", "technique": "Brute Force: Password Guessing",
                         "confidence": "HIGH"}]}
    return {"test": True, "results": alerts.notify(sample)}


@app.post("/incidents/{incident_id}/notify")
def notify_incident(incident_id: int):
    """Manually (re)send the notification for one incident."""
    if not alerts.enabled():
        raise HTTPException(status_code=400, detail="no notification channel configured")
    with SessionLocal() as s:
        inc = s.get(Incident, incident_id)
        if inc is None:
            raise HTTPException(status_code=404, detail="incident not found")
        sources = s.scalar(select(func.count(func.distinct(Event.src_ip))).where(Event.incident_id == incident_id))
        res = alerts.notify(incident_dict(inc, sources))
        if any(v == "ok" for v in res.values()):
            inc.notified_at = datetime.now()
            s.commit()
        return {"incident": incident_id, "results": res}


@app.get("/mitre/coverage")
def mitre_coverage():
    """Which detector families map to which ATT&CK techniques."""
    return coverage()


@app.get("/mitre/summary")
def mitre_summary(include_closed: bool = True, include_noise: bool = False, noise_max_events: int | None = None):
    """Technique counts across incidents (for dashboards / Power BI)."""
    with SessionLocal() as s:
        q = select(Incident)
        if not include_closed:
            q = q.where(Incident.status != "CLOSED")
        if not include_noise:
            q = q.where(~noise_clause(noise_max_events))
        agg = {}
        for inc in s.scalars(q).all():
            for tid, t in json.loads(inc.techniques or "{}").items():
                a = agg.setdefault(tid, {"technique_id": tid, "technique": t["technique"], "tactic": t["tactic"],
                                         "incidents": 0, "flows": 0})
                a["incidents"] += 1
                a["flows"] += t["flows"]
        return {"techniques": sorted(agg.values(), key=lambda x: -x["incidents"])}
