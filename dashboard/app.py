"""SOC dashboard for the threat-detection API (Streamlit).
Run the API first:   uvicorn api.main:app --port 8000
Then:                streamlit run dashboard/app.py
"""
import os
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

DEFAULT_API = os.getenv("SOC_API_URL", "http://127.0.0.1:8000")
SAMPLE_CSV = Path(__file__).with_name("sample_flows.csv")
SEV_COLORS = {"HIGH": "#d64545", "MEDIUM": "#e0a030", "LOW": "#8a94a6"}
CONF_ICON = {"HIGH": "🟥", "MEDIUM": "🟧", "LOW": "⬜"}
VERDICT_COLORS = {"ALERT": "#d64545", "REVIEW": "#e0a030", "NORMAL": "#5b8def"}

st.set_page_config(page_title="SOC Threat Dashboard", page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------- API helpers
def api_get(path, **params):
    r = requests.get(f"{st.session_state.api}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def api_post(path, payload):
    r = requests.post(f"{st.session_state.api}{path}", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=10, show_spinner=False)
def load_incidents(api, include_noise, limit):
    return requests.get(f"{api}/incidents", params={"limit": limit, "include_noise": include_noise},
                        timeout=30).json()


@st.cache_data(ttl=10, show_spinner=False)
def load_incident(api, incident_id, n_events):
    return requests.get(f"{api}/incidents/{incident_id}", params={"events": n_events}, timeout=30).json()


@st.cache_data(ttl=30, show_spinner=False)
def load_static(api, path):
    return requests.get(f"{api}{path}", timeout=30).json()


def fmt_time(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %H:%M:%S")
    except (TypeError, ValueError):
        return iso


def sev_badge(sev):
    return f"<span style='background:{SEV_COLORS.get(sev, '#888')};color:white;padding:2px 10px;" \
           f"border-radius:10px;font-weight:600;font-size:0.85em'>{sev}</span>"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🛡️ SOC Dashboard")
    st.session_state.api = st.text_input("API URL", value=st.session_state.get("api", DEFAULT_API))
    try:
        health = api_get("/health")
        st.success(f"API online · model `{health['model']}`")
    except Exception as exc:  # noqa: BLE001
        st.error(f"API not reachable at {st.session_state.api}\n\nStart it with:\n"
                 f"`uvicorn api.main:app --port 8000`\n\n({type(exc).__name__})")
        st.stop()

    st.subheader("Queue filters")
    status_f = st.radio("Status", ["OPEN", "CLOSED", "All"], horizontal=True)
    sev_f = st.multiselect("Severity", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
    fam_box = st.empty()
    include_noise = st.toggle("Show noise incidents", value=False,
                              help="Noise = LOW incidents with too little evidence (single flows, a few "
                                   "REVIEW-only flows, or scattered sources). Hidden by default.")
    limit = st.number_input("Max incidents loaded", 50, 100000, 2000, step=500)
    if st.button("🔄 Refresh", width="stretch"):
        st.cache_data.clear()
    auto = st.toggle("Auto-refresh", value=False, help="Reload incidents periodically (live monitoring)")
    every = st.select_slider("Every (seconds)", [5, 10, 15, 30, 60], value=15, disabled=not auto)

data = load_incidents(st.session_state.api, include_noise, int(limit))
inc = pd.DataFrame(data["incidents"])
if not inc.empty:
    inc["top_technique"] = inc["mitre"].apply(
        lambda m: f"{m[0]['technique_id']} {m[0]['technique']}" if m else "—")
    inc["first"] = inc["first_seen"].map(fmt_time)
    inc["last"] = inc["last_seen"].map(fmt_time)
families = sorted(inc["family"].unique()) if not inc.empty else []
fam_f = fam_box.multiselect("Family", families, default=families)

view = inc
if not inc.empty:
    if status_f != "All":
        view = view[view["status"] == status_f]
    view = view[view["severity"].isin(sev_f) & view["family"].isin(fam_f)]

# ---------------------------------------------------------------- KPI row
open_inc = inc[inc["status"] == "OPEN"] if not inc.empty else inc
k = st.columns(5)
k[0].metric("Open incidents", len(open_inc))
k[1].metric("Open HIGH", int((open_inc["severity"] == "HIGH").sum()) if not open_inc.empty else 0)
k[2].metric("Open MEDIUM", int((open_inc["severity"] == "MEDIUM").sum()) if not open_inc.empty else 0)
k[3].metric("Hidden as noise", data.get("hidden_noise_incidents", 0),
            help="Low-evidence incidents kept in the database but hidden from the queue")
k[4].metric("Closed (with feedback)", int((inc["status"] == "CLOSED").sum()) if not inc.empty else 0)

tab_o, tab_q, tab_m, tab_s = st.tabs(["📈 Overview", "📋 Incident queue", "🎯 MITRE ATT&CK", "🧪 Score flows"])

# ---------------------------------------------------------------- tab 0: overview
with tab_o:
    if inc.empty:
        st.info("No incidents yet.")
    else:
        ov = inc.copy()
        ov["start"] = pd.to_datetime(ov["first_seen"])
        ov["hour"] = ov["start"].dt.floor("h")
        sev_scale = alt.Scale(domain=list(SEV_COLORS), range=list(SEV_COLORS.values()))
        a, b = st.columns([1.4, 1])
        with a:
            st.markdown("#### Incidents over time")
            st.altair_chart(alt.Chart(ov).mark_bar().encode(
                x=alt.X("hour:T", title=None), y=alt.Y("count():Q", title="incidents started"),
                color=alt.Color("severity:N", scale=sev_scale, legend=alt.Legend(orient="bottom")),
                tooltip=["hour:T", "severity:N", "count():Q"]).properties(height=280), width="stretch")
        with b:
            st.markdown("#### Attack families")
            fam = ov.groupby(["family", "severity"], as_index=False).agg(incidents=("id", "count"),
                                                                       flows=("events", "sum"))
            st.altair_chart(alt.Chart(fam).mark_bar().encode(
                x=alt.X("incidents:Q"), y=alt.Y("family:N", sort="-x", title=None),
                color=alt.Color("severity:N", scale=sev_scale, legend=None),
                tooltip=["family", "severity", "incidents", "flows"]).properties(height=280), width="stretch")
        st.markdown("#### Most attacked targets")
        tgt = (ov.groupby("target", as_index=False)
               .agg(incidents=("id", "count"), flows=("events", "sum"), max_risk=("max_risk", "max"),
                    families=("family", lambda x: ", ".join(sorted(set(x)))))
               .sort_values(["flows"], ascending=False).head(10))
        st.dataframe(tgt, hide_index=True, width="stretch",
                     column_config={"max_risk": st.column_config.ProgressColumn("Max risk", min_value=0,
                                                                                max_value=100, format="%d")})

# ---------------------------------------------------------------- tab 1: queue + detail
with tab_q:
    if view.empty:
        st.info("No incidents match the filters. Ingest flows with `python -m src.replay_ingest`, "
                "or turn on 'Show noise incidents'.")
    else:
        left, right = st.columns([1.15, 1])
        with left:
            st.caption(f"{len(view)} incidents · sorted by severity, then most recent")
            table = view[["id", "severity", "family", "target", "events", "alerts", "reviews", "sources",
                          "max_risk", "top_technique", "first", "last", "status"]]
            sel = st.dataframe(
                table, hide_index=True, width="stretch", height=520,
                on_select="rerun", selection_mode="single-row",
                column_config={
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "max_risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%d"),
                    "top_technique": "Top ATT&CK technique",
                    "first": "First seen", "last": "Last seen",
                })
            rows = sel.selection.rows if sel and sel.selection else []
            chosen = int(table.iloc[rows[0]]["id"]) if rows else int(table.iloc[0]["id"])

        with right:
            d = load_incident(st.session_state.api, chosen, 50)
            st.markdown(f"### Incident #{d['id']} &nbsp; {sev_badge(d['severity'])}", unsafe_allow_html=True)
            st.markdown(f"**{d['family']}** against **`{d['target']}`** · {d['sources']} source(s) · "
                        f"{fmt_time(d['first_seen'])} → {fmt_time(d['last_seen'])}")
            if d.get("notified_at"):
                st.caption(f"🔔 Slack/email notification sent {fmt_time(d['notified_at'])}")
            c = st.columns(4)
            c[0].metric("Flows", d["events"]); c[1].metric("Alerts", d["alerts"])
            c[2].metric("Reviews", d["reviews"]); c[3].metric("Max risk", d["max_risk"])

            st.markdown("#### MITRE ATT&CK")
            if not d["mitre"]:
                st.caption("No technique claimed: no evidence check passed (e.g. unknown anomaly).")
            for t in d["mitre"]:
                with st.container(border=True):
                    st.markdown(f"{CONF_ICON.get(t['confidence'], '')} **[{t['technique_id']}]({t['url']}) "
                                f"{t['technique']}** · {t['tactic']} · confidence **{t['confidence']}**")
                    for e in t["evidence"]:
                        st.caption(f"✓ {e}")

            ev = pd.DataFrame(d.get("sample_events", []))
            if not ev.empty:
                st.markdown("#### Why the model flagged it (top event, SHAP)")
                top = ev.iloc[0]
                reasons = pd.DataFrame(top.get("reasons") or [])
                if not reasons.empty:
                    reasons["label"] = reasons.apply(
                        lambda r: f"{r['feature']} = {r['value']:,.4g} (normal ≈ {r['normal_median']:,.4g})", axis=1)
                    chart = alt.Chart(reasons).mark_bar(color="#d64545").encode(
                        x=alt.X("contribution:Q", title="push towards this attack class"),
                        y=alt.Y("label:N", sort="-x", title=None),
                        tooltip=["feature", "value", "normal_median", "contribution"]).properties(height=180)
                    st.altair_chart(chart, width="stretch")
                with st.expander(f"Sample events ({len(ev)})"):
                    st.dataframe(ev[["ts", "src_ip", "dst_port", "verdict", "risk", "xgb_prob", "rule_match",
                                     "iforest_percentile"]], hide_index=True, width="stretch")

            st.markdown("#### Analyst feedback")
            if d["status"] == "CLOSED":
                st.success(f"Closed as **{d['analyst_verdict']}**" +
                           (f" — {d['analyst_note']}" if d.get("analyst_note") else ""))
            else:
                note = st.text_input("Note (optional)", key=f"note_{chosen}")
                b1, b2 = st.columns(2)
                for col, verdict, label in [(b1, "TP", "✅ True positive"), (b2, "FP", "❌ False positive")]:
                    if col.button(label, key=f"{verdict}_{chosen}", width="stretch"):
                        api_post(f"/incidents/{chosen}/feedback", {"verdict": verdict, "note": note or None})
                        st.cache_data.clear()
                        st.rerun()

# ---------------------------------------------------------------- tab 2: MITRE
with tab_m:
    summ = pd.DataFrame(api_get("/mitre/summary", include_noise=include_noise)["techniques"])
    if summ.empty:
        st.info("No techniques yet: ingest some flows first.")
    else:
        summ["label"] = summ["technique_id"] + " " + summ["technique"]
        a, b = st.columns([1.3, 1])
        with a:
            st.markdown("#### Techniques observed (incidents)")
            st.altair_chart(alt.Chart(summ).mark_bar().encode(
                x=alt.X("incidents:Q", title="incidents"),
                y=alt.Y("label:N", sort="-x", title=None),
                color=alt.Color("tactic:N", legend=alt.Legend(orient="bottom", title="Tactic")),
                tooltip=["technique_id", "technique", "tactic", "incidents", "flows"]).properties(height=320),
                width="stretch")
        with b:
            st.markdown("#### By tactic")
            tac = summ.groupby("tactic", as_index=False)[["incidents", "flows"]].sum().sort_values(
                "incidents", ascending=False)
            st.dataframe(tac, hide_index=True, width="stretch")
        st.dataframe(summ[["technique_id", "technique", "tactic", "incidents", "flows"]],
                     hide_index=True, width="stretch")
    with st.expander("Detector coverage: which family can map to which technique"):
        cov = load_static(st.session_state.api, "/mitre/coverage")
        st.json(cov, expanded=False)

# ---------------------------------------------------------------- tab 3: score a CSV
with tab_s:
    st.markdown("Upload a CSV of CICFlowMeter flows (same column names as the training data). "
                "Each row is scored by the hybrid detector; nothing is stored.")
    up = st.file_uploader("Flows CSV", type=["csv"])
    use_sample = SAMPLE_CSV.exists() and st.checkbox(
        f"Use the bundled demo file ({SAMPLE_CSV.name}: 1,060 labelled test flows, 9 attack types)")
    src = up if up is not None else (SAMPLE_CSV if use_sample else None)
    if src is not None:
        flows = pd.read_csv(src)
        info = load_static(st.session_state.api, "/model-info")
        need = info.get("required_features") or info.get("features") or []
        missing = [f for f in need if f not in flows.columns]
        if missing:
            st.error(f"{len(missing)} required columns missing, e.g. {missing[:5]}")
        else:
            n = min(len(flows), 20000)
            if len(flows) > n:
                st.warning(f"Scoring the first {n:,} of {len(flows):,} rows.")
            feats = flows[need].iloc[:n].astype(float)
            res = []
            prog = st.progress(0.0, "Scoring...")
            for s in range(0, n, 5000):
                out = api_post("/score", {"flows": feats.iloc[s:s + 5000].to_dict("records"), "explain": False})
                res += out["results"]
                prog.progress(min(1.0, (s + 5000) / n))
            prog.empty()
            r = pd.DataFrame([{"verdict": x["verdict"], "family": x["family"], "risk": x["risk_score"],
                               "techniques": ", ".join(t["technique_id"] for t in x.get("mitre", []))} for x in res])
            if "Label" in flows.columns:
                r.insert(0, "true_label", flows["Label"].iloc[:n].values)
            c = st.columns(3)
            for col, v in zip(c, ["ALERT", "REVIEW", "NORMAL"]):
                col.metric(v, int((r["verdict"] == v).sum()))
            st.altair_chart(alt.Chart(r).mark_bar().encode(
                x=alt.X("count():Q", title="flows"), y=alt.Y("family:N", sort="-x", title=None),
                color=alt.Color("verdict:N", scale=alt.Scale(domain=list(VERDICT_COLORS),
                                                             range=list(VERDICT_COLORS.values())))),
                width="stretch")
            if "true_label" in r.columns:
                st.markdown("#### Verdict by true label")
                ct = pd.crosstab(r["true_label"], r["verdict"]).reindex(columns=list(VERDICT_COLORS), fill_value=0)
                ct["flagged %"] = (100 * (ct["ALERT"] + ct["REVIEW"]) / ct.sum(axis=1)).round(1)
                st.dataframe(ct, width="stretch")
            st.dataframe(r, hide_index=True, width="stretch")
            st.download_button("Download results CSV", r.to_csv(index=False), "scored_flows.csv", "text/csv")

# ---------------------------------------------------------------- live refresh
if auto:
    st.caption(f"Auto-refresh on: next reload in {every}s")
    time.sleep(every)
    st.cache_data.clear()
    st.rerun()
