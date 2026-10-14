# Running the full stack with Docker Compose

Services: PostgreSQL, API (FastAPI), dashboard (Streamlit), Prometheus, Grafana, plus an optional one-shot
`replay` job that feeds demo traffic.

## Prerequisites (Windows)
1. Install **Docker Desktop** (WSL 2 backend), start it, wait until it says "Engine running".
2. The model files must exist locally (they are not in git): `models/xgb_v1_nofp_nohdr_clean_r1500.json`,
   `models/iforest_v2.joblib`, etc. If missing: `python -m src.run_nohdr`.
3. Ports 5432, 8000, 8501, 9090 and 3000 must be free. Stop a local `uvicorn` / `streamlit` / PostgreSQL first.

## Start
```powershell
cd "D:\projects\fraud detection"
docker compose up -d --build          # first build ~5-10 min (Python packages), later starts take seconds
docker compose ps                     # api and dashboard should become "healthy"
docker compose --profile demo run --rm replay     # feed one test day (needs data/processed/all_days.parquet)
```

| What | URL |
|---|---|
| API docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| Prometheus (targets, alerts) | http://localhost:9090/targets , http://localhost:9090/alerts |
| Grafana (admin / admin) | http://localhost:3000 → dashboard "SOC Threat Detection - Overview" |
| PostgreSQL for Power BI | localhost:5432, database `soc`, user `soc`, password `soc` |

Other demo days: `docker compose --profile demo run --rm replay python -m src.replay_ingest --url http://api:8000 --day DDoS2-Wednesday-21-02-2018`

## Monitoring
`GET /metrics` exposes:
- `soc_flows_scored_total{endpoint,verdict}`, `soc_scoring_seconds` (latency histogram), `soc_batch_flows`
- `soc_open_incidents{severity}`, `soc_noise_incidents`, `soc_incidents{status}`, `soc_analyst_verdicts{verdict}`
- `soc_notifications_total{channel,result}`, `soc_model_info{model,decision}`
- `soc_drift_drifted_features{window}`, `soc_drift_alarm{window}`, `soc_drift_attack_detect_percent{window}`
  (from `reports/drift/summary.json`, mounted read-only into the API; refresh with `python -m src.drift_monitor`)

Prometheus alert rules (`monitoring/alert_rules.yml`): HighSeverityIncidentOpen, ModelDriftDetected, ScoringSlow, ApiDown.
Validated with `promtool check config` and against a live API scrape (all dashboard queries return data).

## Alerts inside Docker
Create a `.env` file next to `docker-compose.yml` (it is git-ignored):
```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
SMTP_HOST=smtp.gmail.com
SMTP_USER=you@gmail.com
SMTP_PASSWORD=<app password>
ALERT_EMAIL_TO=you@gmail.com
```
then `docker compose up -d` again.

## Stop / reset
```powershell
docker compose down          # stop, keep the PostgreSQL data
docker compose down -v       # stop and delete the database volume (fresh start)
docker compose logs -f api   # follow API logs
```
