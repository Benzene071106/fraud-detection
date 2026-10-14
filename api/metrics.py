"""Prometheus metrics for the API (scraped at GET /metrics).

Counters/histograms are updated in the request handlers; database-backed gauges (open incidents, noise,
analyst verdicts) and the latest drift report are computed at scrape time by SocCollector.
"""
import json
from pathlib import Path

try:
    from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, Counter, Gauge, Histogram, generate_latest
    from prometheus_client.core import GaugeMetricFamily
    AVAILABLE = True
except ImportError:          # metrics are optional: the API runs without prometheus-client installed
    AVAILABLE = False

    class _NoOp:
        def __init__(self, *a, **k): pass
        def labels(self, *a, **k): return self
        def inc(self, *a, **k): pass
        def observe(self, *a, **k): pass
        def set(self, *a, **k): pass

    Counter = Gauge = Histogram = _NoOp
    GaugeMetricFamily = None

FLOWS = Counter("soc_flows_scored_total", "Flows scored by the hybrid detector", ["endpoint", "verdict"])
SCORE_SECONDS = Histogram("soc_scoring_seconds", "Time to score one request (all flows in it)", ["endpoint"],
                          buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30))
BATCH = Histogram("soc_batch_flows", "Flows per request", ["endpoint"], buckets=(1, 10, 100, 500, 1000, 2000, 5000))
NOTIFICATIONS = Counter("soc_notifications_total", "Slack/email notifications", ["channel", "result"])
MODEL = Gauge("soc_model_info", "Deployed model and decision rule (value is always 1)", ["model", "decision"])

DRIFT_FILE = Path(__file__).resolve().parents[1] / "reports" / "drift" / "summary.json"


def record(endpoint, results, seconds):
    SCORE_SECONDS.labels(endpoint).observe(seconds)
    BATCH.labels(endpoint).observe(len(results))
    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    for verdict, n in counts.items():
        FLOWS.labels(endpoint, verdict).inc(n)


def record_notification(results):
    for channel, res in results.items():
        NOTIFICATIONS.labels(channel, "ok" if res == "ok" else "error").inc()


class SocCollector:
    """Gauges read from the database / drift report at scrape time (cheap aggregate queries)."""

    def __init__(self, session_factory, incident_model, noise_clause):
        self.sf, self.Incident, self.noise = session_factory, incident_model, noise_clause

    def collect(self):
        from sqlalchemy import func, select
        I = self.Incident
        open_g = GaugeMetricFamily("soc_open_incidents", "Open incidents in the analyst queue (noise hidden)",
                                   labels=["severity"])
        noise_g = GaugeMetricFamily("soc_noise_incidents", "Incidents hidden as noise")
        total_g = GaugeMetricFamily("soc_incidents", "All incidents by status", labels=["status"])
        verdict_g = GaugeMetricFamily("soc_analyst_verdicts", "Closed incidents by analyst verdict", labels=["verdict"])
        try:
            with self.sf() as s:
                sev = dict(s.execute(select(I.severity, func.count()).where(I.status == "OPEN", ~self.noise())
                                     .group_by(I.severity)).all())
                for level in ("HIGH", "MEDIUM", "LOW"):
                    open_g.add_metric([level], sev.get(level, 0))
                noise_g.add_metric([], s.scalar(select(func.count()).select_from(I).where(self.noise())) or 0)
                for st, n in s.execute(select(I.status, func.count()).group_by(I.status)).all():
                    total_g.add_metric([st], n)
                for v, n in s.execute(select(I.analyst_verdict, func.count())
                                      .where(I.analyst_verdict.is_not(None)).group_by(I.analyst_verdict)).all():
                    verdict_g.add_metric([v], n)
        except Exception:   # noqa: BLE001 - a DB hiccup must not break the scrape
            pass
        yield from (open_g, noise_g, total_g, verdict_g)

        if DRIFT_FILE.exists():
            try:
                d = json.loads(DRIFT_FILE.read_text())
                feat = GaugeMetricFamily("soc_drift_drifted_features", "Features drifted (PSI > 0.2) per window",
                                         labels=["window"])
                flag = GaugeMetricFamily("soc_drift_alarm", "1 if the window is flagged as drifted", labels=["window"])
                det = GaugeMetricFamily("soc_drift_attack_detect_percent", "Attack detection on that window (%)",
                                        labels=["window"])
                for w in d["windows"][1:]:
                    feat.add_metric([w["window"]], w["drifted_features"])
                    flag.add_metric([w["window"]], 1.0 if w["dataset_drift"] else 0.0)
                    det.add_metric([w["window"]], w.get("attack_detect%") or 0.0)
                yield from (feat, flag, det)
            except Exception:  # noqa: BLE001
                pass


_registered = False


def setup(session_factory, incident_model, noise_clause, model_tag, decision):
    global _registered
    if not AVAILABLE:
        return
    MODEL.labels(model_tag, decision).set(1)
    if not _registered:                      # importing the app twice (tests, reload) must not double-register
        REGISTRY.register(SocCollector(session_factory, incident_model, noise_clause))
        _registered = True


def render():
    if not AVAILABLE:
        return b"# prometheus-client not installed: python -m pip install prometheus-client\n", "text/plain"
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
