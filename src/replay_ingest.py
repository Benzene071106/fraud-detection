# Replay one day's TEST flows through /ingest with synthetic network metadata (incident pipeline demo)
# Usage: python -m src.replay_ingest --day Bruteforce-Wednesday-14-02-2018 --max-rows 40000
import argparse
import json
import urllib.request
import zlib
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--day", default="Bruteforce-Wednesday-14-02-2018")
ap.add_argument("--max-rows", type=int, default=40000)
ap.add_argument("--batch", type=int, default=2000)
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--attack-minutes", type=int, default=60,
                help="each attack type happens as one burst of this length (0 = spread over the whole day)")
args = ap.parse_args()

ROOT = Path(__file__).resolve().parents[1]
info = json.load(urllib.request.urlopen(f"{args.url}/model-info"))
REQ, CLASSES = info["required_features"], info["classes"]

df = pd.read_parquet(ROOT / "data" / "processed" / "all_days.parquet")
y = df["family"].map({c: i for i, c in enumerate(CLASSES)}).values
idx = np.arange(len(df))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=42)
te = df.iloc[test_idx]
te = te[te["day"] == args.day]
if te.empty:
    raise SystemExit(f"No test rows for day {args.day}. Days: {sorted(df['day'].unique())}")
te = te.sample(min(args.max_rows, len(te)), random_state=42)          # random order = interleaved traffic

labels = te["Label"].astype(str).values
start = datetime.strptime(args.day[-10:], "%d-%m-%Y").replace(hour=9)
DAY = timedelta(hours=8)
rng = np.random.default_rng(7)
hosts = [f"172.31.{rng.integers(1, 64)}.{rng.integers(2, 250)}" for _ in range(300)]
victim = {l: f"172.31.69.{zlib.crc32(l.encode()) % 200 + 10}" for l in set(labels) if l != "Benign"}
attackers = [f"203.0.113.{k}" for k in range(1, 60)]

# Timing: benign traffic is spread over the whole working day; each attack type happens as one
# contiguous burst (like the real CIC-IDS2018 schedule), unless --attack-minutes 0.
burst = timedelta(minutes=args.attack_minutes)
burst_start = {l: start + (DAY - burst) * rng.random() for l in victim} if args.attack_minutes else {}

flows = []
for lab, row in zip(labels, te[REQ].astype(float).to_dict("records")):
    if lab == "Benign":
        src, dst = hosts[rng.integers(300)], hosts[rng.integers(300)]
        ts = start + DAY * rng.random()
    else:
        dst = victim[lab]
        src = attackers[rng.integers(59)] if "DDoS" in lab or "DDOS" in lab else attackers[0]
        ts = burst_start[lab] + burst * rng.random() if lab in burst_start else start + DAY * rng.random()
    flows.append({"features": row, "src_ip": src, "dst_ip": dst, "timestamp": ts.isoformat()})
flows.sort(key=lambda f: f["timestamp"])
for lab, t0 in sorted(burst_start.items(), key=lambda x: x[1]):
    print(f"  attack burst: {lab:<26} {t0:%H:%M} - {t0 + burst:%H:%M}  -> target {victim[lab]}")

print(f"Replaying {len(flows):,} test flows from {args.day}: {dict(Counter(labels))}")
tot, touched = Counter(), set()
for b in range(0, len(flows), args.batch):
    req = urllib.request.Request(f"{args.url}/ingest", data=json.dumps({"flows": flows[b:b + args.batch]}).encode(),
                                 headers={"Content-Type": "application/json"})
    res = json.load(urllib.request.urlopen(req))
    tot.update({"scored": res["scored"], "alerts": res["alerts"], "reviews": res["reviews"]})
    touched.update(res["incidents_touched"])
    print(f"  batch {b // args.batch + 1}: scored {res['scored']}, alerts {res['alerts']}, reviews {res['reviews']}")

# ---- report on THIS run's incidents only (the database may hold earlier replays) ----
attack_targets = set(victim.values())
every = json.load(urllib.request.urlopen(f"{args.url}/incidents?limit=100000&include_noise=true"))["incidents"]
mine = [i for i in every if i["id"] in touched]
queue_ids = {i["id"] for i in json.load(urllib.request.urlopen(f"{args.url}/incidents?limit=100000"))["incidents"]}
shown = [i for i in mine if i["id"] in queue_ids]
hidden = [i for i in mine if i["id"] not in queue_ids]

def is_real(i):
    return i["target"] in attack_targets

print(f"\nTotal: {dict(tot)}")
print(f"Flagged flows: {tot['alerts'] + tot['reviews']:,}  ->  incidents this run: {len(mine):,}   "
      f"by severity: {dict(Counter(i['severity'] for i in mine))}")
print(f"  on attacked targets: {sum(map(is_real, mine))}   on benign hosts (noise): {sum(not is_real(i) for i in mine)}")

real_shown = [i for i in shown if is_real(i)]
noise_shown = [i for i in shown if not is_real(i)]
real_hidden = [i for i in hidden if is_real(i)]
print(f"\nANALYST QUEUE (noise filter on): {len(shown)} shown = {len(real_shown)} real + {len(noise_shown)} noise   "
      f"| hidden: {len(hidden)} ({len(real_hidden)} real attacks hidden by mistake)")
print(f"  queue precision: {len(real_shown) / max(len(shown), 1):.0%}   "
      f"queue reduction: {1 - len(shown) / max(len(mine), 1):.1%}")
for name, group in [("Real attacks in queue", real_shown), ("Noise still in queue", noise_shown),
                    ("Real attacks HIDDEN", real_hidden)]:
    if group:
        print(f"\n{name}:")
        for i in sorted(group, key=lambda x: (-x["alerts"], -x["events"]))[:10]:
            print(f"  #{i['id']:<5} {i['severity']:<6} {i['family']:<15} target={i['target']:<15} events={i['events']:<6} "
                  f"alerts={i['alerts']:<6} reviews={i['reviews']:<4} sources={i['sources']:<3} max_risk={i['max_risk']}")
            for t in i.get("mitre", [])[:2]:
                print(f"         MITRE {t['technique_id']:<10} {t['technique']:<42} [{t['tactic']}] confidence={t['confidence']}")

print("\nNoise threshold comparison (this run): hide LOW incidents with <= N flagged flows")
print(f"  {'N':>2}  {'shown':>6}  {'real':>5}  {'noise':>6}  {'real hidden':>11}  {'precision':>9}")
for n in [1, 2, 3, 5]:
    ids = {i["id"] for i in json.load(urllib.request.urlopen(
        f"{args.url}/incidents?limit=100000&noise_max_events={n}"))["incidents"]}
    sh = [i for i in mine if i["id"] in ids]
    r = sum(map(is_real, sh))
    rh = sum(is_real(i) for i in mine if i["id"] not in ids)
    print(f"  {n:>2}  {len(sh):>6}  {r:>5}  {len(sh) - r:>6}  {rh:>11}  {r / max(len(sh), 1):>9.0%}")

summ = json.load(urllib.request.urlopen(f"{args.url}/mitre/summary"))
print("\nMITRE techniques across queue incidents (all runs, noise excluded):")
for t in summ["techniques"]:
    print(f"  {t['technique_id']:<10} {t['technique']:<42} [{t['tactic']}]  incidents={t['incidents']}  flows={t['flows']}")
