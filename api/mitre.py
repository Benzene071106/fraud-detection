"""Evidence-grounded MITRE ATT&CK mapping for flow-level detections.

Each detection (attack family + verdict) maps to candidate ATT&CK techniques. A technique is
only reported with its supporting EVIDENCE: concrete flow-feature checks that hold for this flow
(e.g. "UDP protocol", "connection held open 42 s with < 1 packet/s"). Confidence combines the
model's probability, whether the evidence checks passed, and whether Rule R1 matched.

Flow data alone cannot prove intent, so every mapping is labelled as a *candidate* technique
with a confidence level, never as a certainty.
"""

ATTACK_VERSION = "Enterprise ATT&CK v15"

TECHNIQUES = {
    "T1110":     ("Brute Force",                          "Credential Access",   "TA0006"),
    "T1110.001": ("Brute Force: Password Guessing",       "Credential Access",   "TA0006"),
    "T1499":     ("Endpoint Denial of Service",           "Impact",              "TA0040"),
    "T1499.001": ("Endpoint DoS: OS Exhaustion Flood",    "Impact",              "TA0040"),
    "T1499.002": ("Endpoint DoS: Service Exhaustion Flood", "Impact",            "TA0040"),
    "T1498":     ("Network Denial of Service",            "Impact",              "TA0040"),
    "T1498.001": ("Network DoS: Direct Network Flood",    "Impact",              "TA0040"),
    "T1071.001": ("Application Layer Protocol: Web Protocols", "Command and Control", "TA0011"),
    "T1190":     ("Exploit Public-Facing Application",    "Initial Access",      "TA0001"),
    "T1046":     ("Network Service Discovery",            "Discovery",           "TA0007"),
    "T1071":     ("Application Layer Protocol",           "Command and Control", "TA0011"),
}

US = 1_000_000  # CICFlowMeter durations/IATs are in microseconds


def _g(row, name):
    v = row.get(name, 0.0)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _ev(text, ok):
    return {"check": text, "passed": bool(ok)}


def _candidates(family, row, rule_match):
    """Return [(technique_id, [evidence checks])] for one flow."""
    proto = _g(row, "Protocol")
    dur_s = _g(row, "Flow Duration") / US
    fwd_pkts, bwd_pkts = _g(row, "Total Fwd Packets"), _g(row, "Total Backward Packets")
    data_pkts = _g(row, "Fwd Act Data Packets")
    fwd_bytes = _g(row, "Fwd Packets Length Total")
    fwd_rate = _g(row, "Fwd Packets/s")
    fwd_max = _g(row, "Fwd Packet Length Max")
    syn, rst = _g(row, "SYN Flag Count"), _g(row, "RST Flag Count")

    if family == "BruteForce":
        return [("T1110.001", [
            _ev("TCP session", proto == 6),
            _ev(f"{data_pkts:.0f} client data packets (repeated credential exchange, 8-40)", 8 <= data_pkts <= 40),
            _ev(f"short-lived session ({dur_s:.2f} s < 5 s)", dur_s < 5),
            _ev("matches SSH login-session signature (Rule R1)", rule_match),
        ])]

    if family == "DoS":
        if syn > 0 and data_pkts == 0 and bwd_pkts <= 1:
            return [("T1499.001", [
                _ev("SYN set, no client payload", syn > 0 and data_pkts == 0),
                _ev(f"server sent {bwd_pkts:.0f} packets (half-open connection)", bwd_pkts <= 1),
            ])]
        slow = dur_s > 5 and fwd_rate < 5
        return [("T1499.002", [
            _ev("TCP session", proto == 6),
            _ev(f"slow-rate hold: {dur_s:.1f} s at {fwd_rate:.2f} pkt/s" if slow
                else f"high request rate: {fwd_rate:.1f} pkt/s over {dur_s:.2f} s", slow or fwd_rate >= 5),
            _ev(f"small client payload ({fwd_bytes:.0f} bytes)", fwd_bytes < 2000),
        ])]

    if family == "DDoS":
        if proto == 17:
            return [("T1498.001", [
                _ev("UDP protocol (volumetric flood)", True),
                _ev(f"{fwd_pkts:.0f} packets, {fwd_bytes:.0f} bytes sent", fwd_pkts >= 1),
            ])]
        return [("T1499.002", [
            _ev("TCP session to a service", proto == 6),
            _ev(f"tiny request ({fwd_max:.0f} bytes max) triggering a response", fwd_max <= 200 and bwd_pkts >= 1),
            _ev(f"short flow ({dur_s:.2f} s)", dur_s < 5),
        ])]

    if family == "Bot":
        return [("T1071.001", [
            _ev("TCP session", proto == 6),
            _ev(f"short, small exchange ({fwd_bytes:.0f} bytes, {dur_s:.2f} s): beacon-like", fwd_bytes < 2000 and dur_s < 2),
            _ev(f"server responded ({bwd_pkts:.0f} packets)", bwd_pkts >= 1),
        ])]

    if family == "Web":
        return [("T1190", [
            _ev("TCP session", proto == 6),
            _ev(f"client sent payload ({data_pkts:.0f} data packets, max {fwd_max:.0f} bytes)", data_pkts >= 1),
            _ev(f"large request packet (max {fwd_max:.0f} bytes > 300)", fwd_max > 300),
        ]), ("T1110.001", [
            _ev("repeated form submissions (needs incident-level count)", False),
            _ev(f"connection reset after request (RST={rst:.0f})", rst > 0),
        ])]

    if family == "Infiltration":
        scan_like = fwd_pkts <= 2 and data_pkts == 0
        return [("T1046", [
            _ev(f"probe-like flow ({fwd_pkts:.0f} packets, no payload)", scan_like),
            _ev(f"reset or no reply (RST={rst:.0f}, server packets={bwd_pkts:.0f})", rst > 0 or bwd_pkts <= 1),
        ]), ("T1071", [_ev("internal host talking out after compromise (weak flow evidence)", not scan_like)])]

    return []


def _confidence(prob, frac_passed, rule_match):
    score = prob * (0.4 + 0.6 * frac_passed) + (0.15 if rule_match else 0.0)
    return ("HIGH" if score >= 0.85 else "MEDIUM" if score >= 0.6 else "LOW"), round(min(score, 1.0), 3)


def map_flow(family, verdict, row, xgb_probability, rule_match):
    """Map one flagged flow to candidate ATT&CK techniques with evidence and confidence."""
    if verdict == "NORMAL" or family in ("Benign", "Unknown anomaly"):
        return []
    out = []
    for tid, evidence in _candidates(family, row, rule_match):
        passed = sum(e["passed"] for e in evidence)
        frac = passed / len(evidence) if evidence else 0.0
        if passed == 0:
            continue                                  # no supporting evidence -> don't claim it
        level, score = _confidence(xgb_probability, frac, rule_match)
        name, tactic, tactic_id = TECHNIQUES[tid]
        out.append({
            "technique_id": tid, "technique": name, "tactic": tactic, "tactic_id": tactic_id,
            "confidence": level, "score": score,
            "evidence": evidence,
            "url": f"https://attack.mitre.org/techniques/{tid.replace('.', '/')}/",
        })
    return sorted(out, key=lambda t: -t["score"])


def coverage():
    """Which detector families map to which techniques (for docs / dashboard)."""
    families = ["BruteForce", "DoS", "DDoS", "Bot", "Web", "Infiltration"]
    rows = []
    for fam in families:
        seen = set()
        for row in ({}, {"Protocol": 17}, {"SYN Flag Count": 1}):
            for tid, _ in _candidates(fam, row, False):
                if tid not in seen:
                    seen.add(tid)
                    name, tactic, tactic_id = TECHNIQUES[tid]
                    rows.append({"family": fam, "technique_id": tid, "technique": name, "tactic": tactic})
    return {"attack_version": ATTACK_VERSION, "mappings": rows}
