"""Notify analysts once when an incident reaches ALERT_MIN_SEVERITY (default HIGH): Slack and/or email.

Configuration (environment variables, all optional; an unset channel is simply skipped):
  SLACK_WEBHOOK_URL   Slack incoming-webhook URL
  SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, SMTP_STARTTLS (1)   mail server
  ALERT_EMAIL_FROM, ALERT_EMAIL_TO (comma-separated)                       sender / recipients
  ALERT_MIN_SEVERITY  HIGH (default) or MEDIUM
  DASHBOARD_URL       link placed in the message (default http://localhost:8501)
"""
import json
import logging
import os
import smtplib
import ssl
import urllib.request
from email.message import EmailMessage

log = logging.getLogger("soc.alerts")
SEV_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
SEV_EMOJI = {"HIGH": ":red_circle:", "MEDIUM": ":large_orange_circle:", "LOW": ":white_circle:"}


def config():
    return {
        "slack": bool(os.getenv("SLACK_WEBHOOK_URL")),
        "email": bool(os.getenv("SMTP_HOST") and os.getenv("ALERT_EMAIL_TO")),
        "min_severity": os.getenv("ALERT_MIN_SEVERITY", "HIGH").upper(),
        "dashboard_url": os.getenv("DASHBOARD_URL", "http://localhost:8501"),
    }


def enabled():
    c = config()
    return c["slack"] or c["email"]


def eligible_severities():
    floor = SEV_RANK.get(config()["min_severity"], 2)
    return [s for s, r in SEV_RANK.items() if r >= floor]


def _summary(inc):
    techs = ", ".join(f"{t['technique_id']} {t['technique']} ({t['confidence']})" for t in inc.get("mitre", [])[:3])
    return [
        f"Family: {inc['family']}   Target: {inc['target']}   Sources: {inc.get('sources', '?')}",
        f"Flows: {inc['events']} (alerts {inc['alerts']}, reviews {inc['reviews']})   Max risk: {inc['max_risk']}",
        f"Window: {inc['first_seen']} -> {inc['last_seen']}",
        f"MITRE ATT&CK: {techs or 'none claimed'}",
    ]


def slack_payload(inc):
    head = f"{SEV_EMOJI.get(inc['severity'], '')} *{inc['severity']} incident #{inc['id']}*: {inc['family']} on `{inc['target']}`"
    lines = _summary(inc)
    return {
        "text": f"{inc['severity']} incident #{inc['id']}: {inc['family']} on {inc['target']}",   # notification preview
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": head}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}},
            {"type": "context", "elements": [{"type": "mrkdwn",
                                              "text": f"Open the dashboard: {config()['dashboard_url']}"}]},
        ],
    }


def send_slack(inc):
    req = urllib.request.Request(os.environ["SLACK_WEBHOOK_URL"], data=json.dumps(slack_payload(inc)).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status


def send_email(inc):
    msg = EmailMessage()
    msg["Subject"] = f"[SOC {inc['severity']}] Incident #{inc['id']}: {inc['family']} on {inc['target']}"
    msg["From"] = os.getenv("ALERT_EMAIL_FROM", os.getenv("SMTP_USER", "soc-alerts@localhost"))
    msg["To"] = os.environ["ALERT_EMAIL_TO"]
    msg.set_content("\n".join(_summary(inc) + ["", f"Dashboard: {config()['dashboard_url']}"]))
    host, port = os.environ["SMTP_HOST"], int(os.getenv("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=15) as smtp:
        if os.getenv("SMTP_STARTTLS", "1") == "1":
            smtp.starttls(context=ssl.create_default_context())
        if os.getenv("SMTP_USER"):
            smtp.login(os.environ["SMTP_USER"], os.getenv("SMTP_PASSWORD", ""))
        smtp.send_message(msg)
    return "sent"


def notify(inc):
    """Send on every configured channel. Returns {channel: 'ok' | 'error: ...'}; never raises."""
    out, c = {}, config()
    for name, on, fn in [("slack", c["slack"], send_slack), ("email", c["email"], send_email)]:
        if not on:
            continue
        try:
            fn(inc)
            out[name] = "ok"
        except Exception as exc:  # noqa: BLE001 - a broken channel must not break ingestion
            out[name] = f"error: {type(exc).__name__}: {exc}"
            log.warning("alert via %s failed for incident %s: %s", name, inc.get("id"), exc)
    return out
