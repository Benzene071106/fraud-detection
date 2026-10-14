# Incident-pipeline evaluation on your laptop: replay each day's TEST flows (synthetic IPs/timestamps,
# attacks as bursts) through the real FastAPI app (in-process, no server needed) with a fresh SQLite DB
# per day, then score the analyst queue. Demonstrates the mechanism; it is not a real-world precision number.
# Usage:  python -m src.eval_incidents                      (all 10 days, full test rows)
#         python -m src.eval_incidents DDoS1 Bruteforce     (days whose name starts with these)
#         python -m src.eval_incidents --max-rows 20000 --quiet
# Checkpoints: each finished day is saved to reports/incident_eval/<day>.json and skipped next time
# (--force reruns). Two terminals can run different days at the same time (separate DB per process).
import argparse, importlib, json, os, sys, tempfile, time, zlib
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data/processed/all_days.parquet"
CACHE = ROOT / "data/eval_days"          # per-day test rows, built once (checkpoint)
DONE = ROOT / "reports/incident_eval"    # per-day results (checkpoint)
CLASSES = ["Benign", "BruteForce", "DoS", "DDoS", "Bot", "Web", "Infiltration"]

ap = argparse.ArgumentParser()
ap.add_argument("days", nargs="*")
ap.add_argument("--max-rows", type=int, default=0, help="0 = all test rows of the day")
ap.add_argument("--attack-minutes", type=int, default=60)
ap.add_argument("--quiet", action="store_true")
ap.add_argument("--force", action="store_true", help="rerun days that already have a checkpoint")
args = ap.parse_args()


def build_cache(required):
    if CACHE.exists() and any(CACHE.glob("*.parquet")):
        return
    print("Building per-day test cache (once)...", flush=True)
    meta = pd.read_parquet(DATA, columns=["family", "day", "Label"])
    y = meta["family"].map({c: i for i, c in enumerate(CLASSES)}).values
    idx = np.arange(len(meta))
    _, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    _, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=42)
    feats = pd.read_parquet(DATA, columns=required)
    te_df = pd.concat([feats.iloc[te].reset_index(drop=True), meta.iloc[te].reset_index(drop=True)], axis=1)
    CACHE.mkdir(parents=True, exist_ok=True)
    for d, g in te_df.groupby("day"):
        g.to_parquet(CACHE / f"{d}.parquet", index=False)
        print(f"  {d}: {len(g):,} rows")


def run_day(path):
    day = path.stem
    db = Path(tempfile.gettempdir()) / f"fd_eval_{day}_{os.getpid()}.db"
    db.unlink(missing_ok=True)
    os.environ["DATABASE_URL"] = f"sqlite:///{db.as_posix()}"
    for m in [m for m in list(sys.modules) if m.startswith("api.")]:
        del sys.modules[m]
    import api.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    c = TestClient(main.app)
    te = pd.read_parquet(path)
    if args.max_rows:
        te = te.sample(min(args.max_rows, len(te)), random_state=42)
    REQ = main.detector.required
    labels = te["Label"].astype(str).values
    start = datetime.strptime(day[-10:], "%d-%m-%Y").replace(hour=9)
    DAY = timedelta(hours=8)
    rng = np.random.default_rng(7)
    hosts = [f"172.31.{rng.integers(1, 64)}.{rng.integers(2, 250)}" for _ in range(300)]
    victim = {l: f"172.31.69.{zlib.crc32(l.encode()) % 200 + 10}" for l in set(labels) if l != "Benign"}
    attackers = [f"203.0.113.{k}" for k in range(1, 60)]
    burst = timedelta(minutes=args.attack_minutes)
    bstart = {l: start + (DAY - burst) * rng.random() for l in victim} if args.attack_minutes else {}
    flows = []
    for lab, row in zip(labels, te[REQ].astype(float).to_dict("records")):
        if lab == "Benign":
            src, dst, ts = hosts[rng.integers(300)], hosts[rng.integers(300)], start + DAY * rng.random()
        else:
            dst = victim[lab]
            src = attackers[rng.integers(59)] if "DDOS" in lab.upper() else attackers[0]
            ts = bstart[lab] + burst * rng.random() if lab in bstart else start + DAY * rng.random()
        flows.append({"features": row, "src_ip": src, "dst_ip": dst, "timestamp": ts.isoformat()})
    flows.sort(key=lambda f: f["timestamp"])
    t0, tot = time.time(), Counter()
    for b in range(0, len(flows), 2000):
        r = c.post("/ingest", json={"flows": flows[b:b + 2000]}).json()
        tot.update({"alerts": r["alerts"], "reviews": r["reviews"]})
    every = c.get("/incidents", params={"limit": 100000, "include_noise": True}).json()["incidents"]
    queue = c.get("/incidents", params={"limit": 100000}).json()["incidents"]
    qids, tgt = {i["id"] for i in queue}, set(victim.values())
    real = [i for i in every if i["target"] in tgt]
    shown_real = [i for i in queue if i["target"] in tgt]
    shown_noise = [i for i in queue if i["target"] not in tgt]
    hidden_real = [i for i in real if i["id"] not in qids]
    found = {l: any(i["target"] == v for i in queue) for l, v in victim.items()}
    res = dict(day=day, setup=SETUP, rows=len(te), flagged=tot["alerts"] + tot["reviews"], incidents=len(every),
               queue=len(queue), real_in_queue=len(shown_real), noise_in_queue=len(shown_noise),
               real_hidden=len(hidden_real), precision=round(len(shown_real) / max(len(queue), 1), 3),
               attacks_found=f"{sum(found.values())}/{len(found)}", missed=[l for l, f in found.items() if not f],
               secs=round(time.time() - t0))
    print(json.dumps(res), flush=True)
    DONE.mkdir(parents=True, exist_ok=True)
    (DONE / f"{day}.json").write_text(json.dumps(res))
    if not args.quiet:
        for name, g in [("REAL", shown_real), ("NOISE", shown_noise), ("REAL HIDDEN", hidden_real)]:
            for i in sorted(g, key=lambda x: -x["events"])[:10]:
                print(f"  {name:<11} #{i['id']:<4} {i['severity']:<6} {i['family']:<15} tgt={i['target']:<15} "
                      f"ev={i['events']:<6} al={i['alerts']:<6} rv={i['reviews']:<4} src={i['sources']:<3} "
                      f"mitre={[(t['technique_id'], t['confidence']) for t in i.get('mitre', [])][:2]}")
    return res


from api.detector import HybridDetector  # noqa: E402
_det = HybridDetector()
SETUP = f"{_det.xgb_tag}/{_det.mode}" + ("+novelty" if _det.nov else "")   # checkpoint valid only for same setup
print("Setup:", SETUP, flush=True)
build_cache(_det.required)
paths = sorted(CACHE.glob("*.parquet"))
if args.days:
    paths = [p for p in paths if any(p.stem.startswith(d) for d in args.days)]
for p in paths:
    ck = DONE / f"{p.stem}.json"
    if not args.force and ck.exists() and json.loads(ck.read_text()).get("setup") == SETUP:
        print(f"[skip] {p.stem}: already done", flush=True)
        continue
    run_day(p)
results = [r for r in (json.loads(f.read_text()) for f in sorted(DONE.glob("*.json"))) if r.get("setup") == SETUP]
out = ROOT / "reports" / "incident_eval.csv"
pd.DataFrame(results).to_csv(out, index=False)
print(f"\nSaved {out}")
print(pd.DataFrame(results)[["day", "queue", "real_in_queue", "noise_in_queue", "attacks_found", "secs"]].to_string(index=False))
