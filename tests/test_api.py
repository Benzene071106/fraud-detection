"""End-to-end checks of the detection pipeline (scoring -> incidents -> noise filter -> MITRE -> feedback)."""
from datetime import datetime, timedelta

import numpy as np


def _features(df, detector):
    return df[detector.required].astype(float).to_dict("records")


def test_health_and_model_info(client):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model-info").json()
    assert info["model"].startswith("xgb_")
    assert info["risk"]["calibrated"] is True


def test_missing_features_rejected(client):
    r = client.post("/score", json={"flows": [{"Flow Duration": 1.0}]})
    assert r.status_code == 422 and "missing_features" in r.json()["detail"]


def test_known_attacks_alert_and_benign_quiet(client, detector, sample_flows):
    res = client.post("/score", json={"flows": _features(sample_flows, detector), "explain": False}).json()["results"]
    verdict = np.array([r["verdict"] for r in res])
    label = sample_flows["Label"].values
    assert (verdict[label == "SSH-Bruteforce"] == "ALERT").mean() >= 0.95
    assert (verdict[label == "Bot"] == "ALERT").mean() >= 0.95
    assert (verdict[label == "Benign"] == "ALERT").mean() <= 0.02          # false alarms stay rare


def test_explanations_and_mitre_present(client, detector, sample_flows):
    ssh = sample_flows[sample_flows["Label"] == "SSH-Bruteforce"].head(3)
    res = client.post("/score", json={"flows": _features(ssh, detector), "explain": True, "top_k": 3}).json()["results"]
    for r in res:
        assert len(r["reasons"]) == 3 and {"feature", "value", "contribution"} <= r["reasons"][0].keys()
        assert any(t["technique_id"].startswith("T1110") for t in r["mitre"])


def test_risk_separates_attacks_from_benign(client, detector, sample_flows):
    res = client.post("/score", json={"flows": _features(sample_flows, detector), "explain": False}).json()["results"]
    risk = np.array([r["risk_score"] for r in res])
    label = sample_flows["Label"].values
    assert np.median(risk[label == "SSH-Bruteforce"]) > 90
    assert np.median(risk[label == "Benign"]) < 5


def test_incident_grouping_noise_and_feedback(client, detector, sample_flows):
    t0 = datetime(2018, 2, 14, 10, 0)
    ssh = sample_flows[sample_flows["Label"] == "SSH-Bruteforce"].head(40)
    benign = sample_flows[sample_flows["Label"] == "Benign"].head(200)
    flows = [{"features": f, "src_ip": "203.0.113.9", "dst_ip": "10.9.9.9",
              "timestamp": (t0 + timedelta(seconds=15 * i)).isoformat()} for i, f in enumerate(_features(ssh, detector))]
    flows += [{"features": f, "src_ip": f"10.1.{i % 50}.{i % 200}", "dst_ip": f"10.2.{i % 60}.{i % 90}",
               "timestamp": (t0 + timedelta(minutes=i)).isoformat()} for i, f in enumerate(_features(benign, detector))]
    r = client.post("/ingest", json={"flows": flows}).json()
    assert r["alerts"] >= 38
    queue = client.get("/incidents", params={"limit": 1000}).json()
    ssh_inc = [i for i in queue["incidents"] if i["target"] == "10.9.9.9"]
    assert len(ssh_inc) == 1, "40 SSH flows from one attacker within 10 min must be ONE incident"
    inc = ssh_inc[0]
    assert inc["severity"] == "HIGH" and inc["family"] == "BruteForce"
    assert any(t["technique_id"] == "T1110.001" for t in inc["mitre"])
    noise_targets = [i for i in queue["incidents"] if i["target"].startswith("10.2.")]
    assert len(noise_targets) <= 1, "scattered benign false alarms must be hidden as noise"
    fb = client.post(f"/incidents/{inc['id']}/feedback", json={"verdict": "TP", "note": "test"}).json()
    assert fb["status"] == "CLOSED" and fb["analyst_verdict"] == "TP"


def test_unknown_detector_budget(detector, sample_flows):
    if detector.nov is None:
        return
    benign = sample_flows[sample_flows["Label"] == "Benign"]
    pct = detector.novelty(benign[detector.required].astype(float))
    assert (pct > detector.nov["threshold"]).mean() <= 0.06               # ~2% budget, small-sample slack
    bot = sample_flows[sample_flows["Label"] == "Bot"]
    assert (detector.novelty(bot[detector.required].astype(float)) > detector.nov["threshold"]).mean() >= 0.9


def test_metrics_and_mitre_endpoints(client):
    m = client.get("/metrics")
    assert m.status_code == 200
    assert "soc_flows_scored_total" in m.text or "prometheus-client not installed" in m.text
    assert "techniques" in client.get("/mitre/summary").json()
    assert client.get("/mitre/coverage").status_code == 200
