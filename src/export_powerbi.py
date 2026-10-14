# Export the SOC database into flat tables for Power BI (works with SQLite or PostgreSQL via DATABASE_URL).
# Writes CSVs to reports/powerbi/ and, with --to-db, the same tables as pbi_* tables inside the database
# (so Power BI can read everything from PostgreSQL with one connection and refresh it).
# Usage:  python -m src.export_powerbi            (CSV only)
#         python -m src.export_powerbi --to-db    (CSV + pbi_* tables in the database)
import argparse
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.db import DB_URL, engine, init_db  # noqa: E402
from api.mitre import TECHNIQUES  # noqa: E402

OUT = ROOT / "reports" / "powerbi"
SEV_ORDER = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}

ap = argparse.ArgumentParser()
ap.add_argument("--to-db", action="store_true", help="also write pbi_* tables into the database")
ap.add_argument("--max-events", type=int, default=200000, help="cap on event rows exported (newest first)")
args = ap.parse_args()

init_db()
inc = pd.read_sql("SELECT * FROM incidents", engine, parse_dates=["first_seen", "last_seen", "notified_at"])
src = pd.read_sql("SELECT incident_id, COUNT(DISTINCT src_ip) AS sources FROM events GROUP BY incident_id", engine)
ev = pd.read_sql(f"SELECT id, ts, src_ip, dst_ip, dst_port, verdict, family, risk, xgb_prob, rule_match, if_pct, "
                 f"incident_id FROM events ORDER BY ts DESC LIMIT {int(args.max_events)}", engine, parse_dates=["ts"])

# noise flag, same rule as the API (api.main.noise_clause) so Power BI can hide noise with one slicer
inc = inc.merge(src, left_on="id", right_on="incident_id", how="left").drop(columns="incident_id")
inc["sources"] = inc["sources"].fillna(0).astype(int)
low = inc["severity"] == "LOW"
web_bot_alert = inc["family"].isin(["Web", "Bot"]) & (inc["alert_count"] > 0)
inc["is_noise"] = low & ~web_bot_alert & ((inc["event_count"] <= 1)
                                          | ((inc["alert_count"] == 0) & (inc["event_count"] < 3))
                                          | ((inc["sources"] >= inc["event_count"]) & (inc["alert_count"] < 3)))
inc["duration_min"] = ((inc["last_seen"] - inc["first_seen"]).dt.total_seconds() / 60).round(1)
inc["date"] = inc["first_seen"].dt.date
inc["hour"] = inc["first_seen"].dt.hour
inc["severity_order"] = inc["severity"].map(SEV_ORDER)
inc["is_false_positive"] = inc["analyst_verdict"].eq("FP")
inc["is_true_positive"] = inc["analyst_verdict"].eq("TP")

rows = []
for _, r in inc.iterrows():
    for tid, t in json.loads(r["techniques"] or "{}").items():
        rows.append({"incident_id": r["id"], "technique_id": tid, "technique": t["technique"],
                     "tactic": t["tactic"], "tactic_id": TECHNIQUES.get(tid, (None, None, None))[2],
                     "flows": t["flows"], "high": t["HIGH"], "medium": t["MEDIUM"], "low": t["LOW"],
                     "url": f"https://attack.mitre.org/techniques/{tid.replace('.', '/')}/"})
tech = pd.DataFrame(rows, columns=["incident_id", "technique_id", "technique", "tactic", "tactic_id", "flows",
                                   "high", "medium", "low", "url"])

inc_out = inc.drop(columns=["techniques"]).rename(columns={"id": "incident_id", "dst_ip": "target"})
tables = {"pbi_incidents": inc_out, "pbi_incident_techniques": tech, "pbi_events": ev.rename(columns={"id": "event_id"})}

OUT.mkdir(parents=True, exist_ok=True)
for name, df in tables.items():
    df.to_csv(OUT / f"{name.removeprefix('pbi_')}.csv", index=False)
    print(f"{name:<26} {len(df):>8,} rows -> {OUT / (name.removeprefix('pbi_') + '.csv')}")
if args.to_db:
    for name, df in tables.items():
        df.to_sql(name, engine, if_exists="replace", index=False)
    print(f"Wrote pbi_* tables into {DB_URL.split('@')[-1]}")
