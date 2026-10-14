"""Group flagged flows into incidents: same family + same target, close in time (window depends on family)."""
import json
from datetime import timedelta
from sqlalchemy import select
from api.db import Event, Incident

# How long a target can be quiet before a new incident starts.
# Floods are dense (seconds between flows); credential, web, bot and infiltration activity is
# sparse and can pause for many minutes between attempts, so it gets a longer window.
WINDOWS = {
    "DDoS": timedelta(minutes=15), "Unknown anomaly": timedelta(minutes=15),
    "DoS": timedelta(minutes=60),          # slow DoS (Slowloris, SlowHTTPTest) and DoS-like slow brute force are sparse
    "BruteForce": timedelta(minutes=60), "Web": timedelta(minutes=60),
    "Bot": timedelta(minutes=60), "Infiltration": timedelta(minutes=60),
}
DEFAULT_WINDOW = timedelta(minutes=15)
HIGH_FAMILIES = {"BruteForce", "Bot", "Web"}


REVIEW_MEDIUM = 100   # sustained anomaly: this many REVIEW flows on one target is worth a look even with no ALERT


def severity(inc):
    if inc.alert_count == 0:
        return "MEDIUM" if inc.review_count >= REVIEW_MEDIUM else "LOW"
    if inc.alert_count >= 100 or (inc.family in HIGH_FAMILIES and inc.alert_count >= 10):
        return "HIGH"
    if inc.alert_count >= 10:
        return "MEDIUM"
    return "LOW"


def store_flagged(session, flows, results):
    """Store ALERT/REVIEW flows as events and attach each to an open incident (or a new one)."""
    cache, touched = {}, set()
    for i in sorted(range(len(flows)), key=lambda j: flows[j].timestamp):
        r = results[i]
        if r["verdict"] == "NORMAL":
            continue
        fl = flows[i]
        key = (r["family"], fl.dst_ip)
        inc = cache.get(key)
        window = WINDOWS.get(key[0], DEFAULT_WINDOW)
        if inc is None or abs(fl.timestamp - inc.last_seen) > window:
            inc = session.scalars(
                select(Incident)
                .where(Incident.family == key[0], Incident.dst_ip == key[1],
                       Incident.status != "CLOSED",
                       Incident.last_seen >= fl.timestamp - window,     # target active recently...
                       Incident.first_seen <= fl.timestamp + window)    # ...and not an incident from the "future"
                .order_by(Incident.last_seen.desc()).limit(1)
            ).first()
            if inc is None:
                inc = Incident(family=key[0], dst_ip=key[1], first_seen=fl.timestamp, last_seen=fl.timestamp,
                               event_count=0, alert_count=0, review_count=0, max_risk=0,
                               severity="LOW", status="OPEN")
                session.add(inc)
                session.flush()
            cache[key] = inc

        inc.first_seen = min(inc.first_seen, fl.timestamp)
        inc.last_seen = max(inc.last_seen, fl.timestamp)
        inc.event_count += 1
        if r["verdict"] == "ALERT":
            inc.alert_count += 1
        else:
            inc.review_count += 1
        inc.max_risk = max(inc.max_risk, r["risk_score"])
        inc.severity = severity(inc)

        techs = json.loads(inc.techniques or "{}")
        for t in r.get("mitre", []):
            e = techs.setdefault(t["technique_id"], {"technique": t["technique"], "tactic": t["tactic"],
                                                     "flows": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0})
            e["flows"] += 1
            e[t["confidence"]] += 1
        inc.techniques = json.dumps(techs)

        c = r["components"]
        session.add(Event(
            ts=fl.timestamp, src_ip=fl.src_ip, dst_ip=fl.dst_ip, dst_port=fl.dst_port,
            verdict=r["verdict"], family=r["family"], risk=r["risk_score"],
            xgb_prob=c["xgb_probability"], rule_match=c["rule_r1_match"], if_pct=c["iforest_percentile"],
            reasons=json.dumps({"shap": r.get("reasons", []), "mitre": r.get("mitre", [])}), features=json.dumps(fl.features), incident_id=inc.id,
        ))
        touched.add(inc.id)
    return sorted(touched)
