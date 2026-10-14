# PostgreSQL, Power BI and alerts: setup

Everything below is optional: with no environment variables the API uses SQLite (`data/soc.db`) and sends no alerts.

## 1. PostgreSQL (Windows)

1. Install PostgreSQL 16 from https://www.postgresql.org/download/windows/ (keep port 5432, remember the `postgres` password).
2. Create a database and user. Open **SQL Shell (psql)** and run:
   ```sql
   CREATE USER soc WITH PASSWORD 'soc';
   CREATE DATABASE soc OWNER soc;
   ```
3. Install the driver and point the API at it (same PowerShell window that starts the API):
   ```powershell
   pip install "psycopg[binary]"
   $env:DATABASE_URL = "postgresql+psycopg://soc:soc@localhost:5432/soc"
   uvicorn api.main:app --port 8000
   ```
   Tables are created automatically on startup. Older databases are migrated (new columns added).
4. Fill it: `python -m src.replay_ingest` (and/or other days with `--day ...`).

Tested on PostgreSQL 16 with SQLAlchemy 2 + psycopg 3: ingest, incident grouping, feedback, MITRE summary, export.

## 2. Power BI

1. Build the reporting tables (same `DATABASE_URL` as the API):
   ```powershell
   python -m src.export_powerbi --to-db
   ```
   This writes `pbi_incidents`, `pbi_incident_techniques`, `pbi_events` into the database and the same data as
   CSV files in `reports/powerbi/` (use the CSVs if you stay on SQLite).
2. Power BI Desktop → **Get data → PostgreSQL database** → server `localhost`, database `soc` → user `soc` / `soc`
   → select the three `pbi_*` tables → **Load**. (CSV route: **Get data → Text/CSV** for each file.)
3. **Model view**: relate `pbi_incidents[incident_id]` 1 → * `pbi_incident_techniques[incident_id]` and
   1 → * `pbi_events[incident_id]`.
4. Measures (Modeling → New measure, on `pbi_incidents`):
   ```DAX
   Incidents      = COUNTROWS ( FILTER ( pbi_incidents, pbi_incidents[is_noise] = FALSE () ) )
   Open HIGH      = CALCULATE ( [Incidents], pbi_incidents[status] = "OPEN", pbi_incidents[severity] = "HIGH" )
   Noise hidden   = CALCULATE ( COUNTROWS ( pbi_incidents ), pbi_incidents[is_noise] = TRUE () )
   Flows flagged  = SUM ( pbi_incidents[event_count] )
   Analyst FP rate =
       DIVIDE ( CALCULATE ( COUNTROWS ( pbi_incidents ), pbi_incidents[is_false_positive] = TRUE () ),
                CALCULATE ( COUNTROWS ( pbi_incidents ), NOT ISBLANK ( pbi_incidents[analyst_verdict] ) ) )
   Avg duration (min) = AVERAGE ( pbi_incidents[duration_min] )
   ```
5. Suggested page (one screen):
   - Cards: Incidents, Open HIGH, Noise hidden, Analyst FP rate
   - Stacked column: `first_seen` (by hour) × Incidents, legend `severity`
   - Bar: `pbi_incident_techniques[technique]` × count of `incident_id`, legend `tactic` (MITRE view)
   - Bar: `family` × Incidents
   - Table: incident_id, severity, family, target, sources, event_count, max_risk, status
   - Slicers: severity, family, status, `is_noise` (default FALSE)
6. Refresh after new data: rerun `python -m src.export_powerbi --to-db`, then **Refresh** in Power BI.

## 3. Slack / email alerts for HIGH incidents

Each incident is announced **once**, when it first reaches the minimum severity (default HIGH). Sending happens
after the `/ingest` response, so a slow or broken channel never blocks ingestion; if every channel fails, the
incident is released and retried on its next update.

Set these in the PowerShell window that starts the API, then start it:
```powershell
# Slack: create an Incoming Webhook (api.slack.com/apps → your app → Incoming Webhooks) and paste its URL
$env:SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/XXX/YYY/ZZZ"

# Email (Gmail example: needs 2-step verification + an App Password, not your normal password)
$env:SMTP_HOST = "smtp.gmail.com"; $env:SMTP_PORT = "587"
$env:SMTP_USER = "you@gmail.com";  $env:SMTP_PASSWORD = "<16-char app password>"
$env:ALERT_EMAIL_TO = "you@gmail.com"

$env:ALERT_MIN_SEVERITY = "HIGH"          # or MEDIUM
$env:DASHBOARD_URL = "http://localhost:8501"
uvicorn api.main:app --port 8000
```
Check and test:
```powershell
curl.exe http://127.0.0.1:8000/alerts/config          # which channels are on (no secrets shown)
curl.exe -X POST http://127.0.0.1:8000/alerts/test    # sends a sample HIGH alert
curl.exe -X POST http://127.0.0.1:8000/incidents/1/notify   # resend for one incident
```
Never commit webhook URLs or passwords; keep them in environment variables.
