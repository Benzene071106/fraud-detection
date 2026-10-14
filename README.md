# Security Log Analysis and Threat Detection Dashboard using ML

B.Tech CSE Sem V project (Ramdeobaba University). Network flows (CICFlowMeter features, CSE-CIC-IDS2018) are
scored by a hybrid detector, grouped into incidents, mapped to MITRE ATT&CK with evidence, and shown to an analyst.

```
flows ──► FastAPI /ingest ──► Hybrid detector ──────────────► incidents (SQLite / PostgreSQL)
            │                  • XGBoost, 7 families           │  grouped by family + target + time window
            │                  • Rule R1 (SSH brute force)     │  noise filter, severity, MITRE roll-up
            │                  • Isolation Forest (unknowns)   ├──► Streamlit dashboard (analyst queue, TP/FP)
            │                  • SHAP reasons, MITRE evidence  ├──► Power BI (pbi_* tables / CSV)
            └──────────────────────────────────────────────────┴──► Slack / email for HIGH incidents
```

## Results (held-out test split, 424,579 flows)

Model `xgb_v1_nofp_nohdr_clean_r1500`, decision rule `flood_or` (all thresholds chosen on the validation split).

| Metric | Value |
|---|---|
| ALERT false-positive rate (benign) | 0.174 % |
| Precision if 1 % of traffic is attack | 0.836 |
| Detection (ALERT + REVIEW) | 88.4 % |
| Macro-F1 (excl. Infiltration) | 0.923 |
| BruteForce / Bot / DDoS / DoS | 99.8 % / 99.1 % / 96.7 % / 91.0 % |
| Web / Infiltration (ALERT + REVIEW) | 71.4 % / 7.7 % |

Design choices that cost accuracy on purpose (robustness over testbed score):
- **No TCP fingerprint features** (`Init Win Bytes`, `Fwd Seg Size Min`) and **no header lengths**: header size per
  packet reveals the attacker machine's OS/TCP options, i.e. the lab's attack box, not the attack. Keeping it gave
  macro-F1 0.948 / DoS 99.9 %, but that would not transfer to a differently configured attacker.
- **Data ceiling**: 19.6 % of Web and 8.4 % of Infiltration test flows have an exact benign twin (identical 64
  features), so no flow-level model can separate them. Infiltration goes to REVIEW, never ALERT.

Incident level: every test day replayed in full through the API (synthetic IPs and timestamps, attacks as
60-minute bursts; `python -m src.eval_incidents`, results in `reports/incident_eval.csv`). This demonstrates the
mechanism; it is not a real-world precision figure.

| | Value |
|---|---|
| Days / flows replayed | 10 / 424,579 |
| Analyst queue | 31 incidents = 30 real + 1 noise (queue precision 97 %) |
| Attack types found | 17 of 18 (missed: DoS-SlowHTTPTest, 5 test flows) |

## Drift monitoring (`python -m src.drift_monitor`)

Feature drift (PSI > 0.2 per feature) is measured on the flows the model calls NORMAL, so it needs no labels.
Reference = 2018 validation split. CIC-IDS2017 (different network, year and CICFlowMeter version) stands in for
"the model was moved to a new network".

| Window | Features drifted | Drift alarm | Attack detection |
|---|---|---|---|
| 2018 Web2 / DoS1 / Botnet (same network) | 12 / 13 / 11 of 64 | no | 66.7 % / 99.4 % / 99.2 % |
| 2017 Thursday web attacks (new network) | 43 of 64 | **yes** | **5.8 %** |
| 2017 Wednesday DoS (new network) | 43 of 64 | **yes** | **4.6 %** |

The alarm fires exactly where detection collapses, while the benign alert rate stays low (≈0.35 %), so the
failure would be silent without drift monitoring. Caveats: 2017 has no Protocol column (inferred from TCP
flags/window), and part of the drift is the older CICFlowMeter version, itself a realistic pipeline change.
Per-window Evidently HTML reports are written to `reports/drift/` (not in git, ~7 MB each).

## Experiment tracking (`python -m src.log_mlflow`, then `mlflow ui --backend-store-uri sqlite:///mlflow.db`)

Logs every model variant × decision rule (validation and test metrics), the deployed model (registered as
`threat-detector-xgb`), the drift run and the incident evaluation. Re-running skips runs that already exist.

Details: `reports/sweep_nohdr_validation.txt` (model sweep, decision rules, test numbers), `reports/*.csv`.

## Run everything with Docker

```powershell
docker compose up -d --build                      # PostgreSQL + API + dashboard + Prometheus + Grafana
docker compose --profile demo run --rm replay     # demo traffic
```
API http://localhost:8000/docs · dashboard http://localhost:8501 · Grafana http://localhost:3000 (admin/admin) ·
Prometheus http://localhost:9090. Details: `docs/DOCKER.md`.

## Run it without Docker (Windows PowerShell)

```powershell
pip install -r requirements-app.txt
uvicorn api.main:app --port 8000                 # terminal 1: API (docs at http://127.0.0.1:8000/docs)
python -m src.replay_ingest                      # terminal 2: replay one test day into the API
streamlit run dashboard/app.py                   # terminal 2: dashboard at http://localhost:8501
```
- Other days: `python -m src.replay_ingest --day DDoS2-Wednesday-21-02-2018`
- PostgreSQL, Power BI, Slack/email alerts: `docs/SETUP_POSTGRES_POWERBI_ALERTS.md`
- Drift report: `python -m src.drift_monitor` (dashboard tab 🌊 Drift, API `GET /drift`)
- MLflow: `python -m src.log_mlflow` then `mlflow ui --backend-store-uri sqlite:///mlflow.db` (http://127.0.0.1:5000)
- Model files (`models/*.json` boosters, `*.joblib`) are not in git; retrain with `python -m src.run_nohdr`.

## API

| Endpoint | Purpose |
|---|---|
| `POST /score` | score up to 5,000 flows, verdict + family + risk + SHAP reasons + MITRE |
| `POST /ingest` | score and store flows with src/dst IP and time; updates incidents; queues alerts |
| `GET /incidents`, `GET /incidents/{id}` | analyst queue (noise hidden by default), detail with sample events |
| `POST /incidents/{id}/feedback` | analyst TP / FP, closes the incident |
| `GET /mitre/summary`, `GET /mitre/coverage` | technique counts, family → technique map |
| `GET /drift` | latest drift report |
| `GET /metrics` | Prometheus metrics (throughput, latency, open incidents, drift, notifications) |
| `GET /alerts/config`, `POST /alerts/test`, `POST /incidents/{id}/notify` | Slack / email notifications |

## Repository map

| Path | What |
|---|---|
| `api/` | FastAPI app, detector, incident grouping, MITRE mapping, alerts, database models |
| `dashboard/app.py` | Streamlit analyst dashboard (+ `sample_flows.csv` demo file) |
| `src/train_xgb.py`, `src/xgb_thresholds.py`, `src/refine_final.py`, `src/run_nohdr.py` | training + thresholds (checkpointed) |
| `src/sweep_nohdr.py`, `src/eval_sweep.py` | model/decision sweep (selection on validation, test read once) |
| `src/eval_incidents.py`, `src/replay_ingest.py` | incident-level evaluation and live replay |
| `src/export_powerbi.py` | reporting tables for Power BI |
| `src/drift_monitor.py` | drift monitoring (PSI/KS, Evidently HTML) |
| `src/log_mlflow.py` | MLflow experiment tracking + model registry |
| `Dockerfile`, `docker-compose.yml`, `monitoring/` | container stack, Prometheus config + alert rules, Grafana dashboard |
