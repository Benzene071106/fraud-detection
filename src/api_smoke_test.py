# Send example flows to the running API and print verdicts
# Usage: python -m src.api_smoke_test   (API must be running on port 8000)
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
examples = json.load(open(ROOT / "models" / "example_flows.json"))
flows = [{k: v for k, v in e.items() if k != "true_label"} for e in examples]

body = json.dumps({"flows": flows, "explain": True, "top_k": 3}).encode()
req = urllib.request.Request("http://127.0.0.1:8000/score", data=body,
                             headers={"Content-Type": "application/json"})
res = json.load(urllib.request.urlopen(req))

print(f"{'TRUE LABEL':<26} {'VERDICT':<7} {'FAMILY':<16} RISK  TOP REASONS")
for e, r in zip(examples, res["results"]):
    print(f"{e['true_label']:<26} {r['verdict']:<7} {r['family']:<16} {r['risk_score']:>4}  "
          f"{[x['feature'] for x in r['reasons']]}")