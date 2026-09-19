import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.rail_channels import side_dataframe_columns
from src.rail_diagram import train_diagram
from src.rail_features import SAMPLE_RATE_HZ
from src.rail_inference import load_metadata, predict_rail_file
from src.rail_interpretation import assess, describe_feature, explain

st.set_page_config(page_title="RailGuard", page_icon="🚆", layout="wide", initial_sidebar_state="collapsed")

MODEL_PATH = Path("models/rail_model.joblib")
REQUIRED_COLUMNS = 129
SIDE1_COLOR = "#5B7CFF"
SIDE2_COLOR = "#35D6A0"
SEVERITY_RANK = {"severe": 4, "high": 3, "moderate": 2, "low": 1, "normal": 0}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

:root{
  --bg:#0D1117; --surface:#161B26; --surface-2:#1D2333; --border:#2A3142;
  --text:#E9EDF5; --text-muted:#98A2B8;
  --primary:#5B7CFF; --primary-soft:#1B2447;
  --success:#35D6A0; --success-soft:#0E2A22;
  --danger:#FF6B72; --danger-soft:#2E1418;
  --warn:#F0B34A; --warn-soft:#2C2110;
  --radius-sm:10px; --radius-md:14px; --radius-lg:20px;
  --shadow:0 2px 10px rgba(0,0,0,.35);
}

html, body, [class*="css"], .stMarkdown, p, span, div, label{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
}
code, pre, [data-testid="stMetricValue"]{ font-family:'JetBrains Mono','SFMono-Regular',monospace !important; }

.stApp{ background:var(--bg); }
.block-container{ max-width:1560px; padding:1.5rem 2.2rem 4rem; }
h1,h2,h3,h4{ letter-spacing:-0.02em; color:var(--text); }

/* Top bar */
.topbar{ display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px;
  padding:18px 24px; margin-bottom:16px; border:1px solid var(--border); border-radius:var(--radius-lg);
  background:linear-gradient(120deg,var(--surface) 0%,var(--surface-2) 100%); box-shadow:var(--shadow); }
.brand{ display:flex; align-items:center; gap:15px; }
.brand-badge{ width:48px; height:48px; border-radius:14px; background:var(--primary);
  display:flex; align-items:center; justify-content:center; font-size:1.55rem; flex-shrink:0; }
.brand-name{ font-size:1.45rem; font-weight:800; margin:0; line-height:1.1; color:var(--text); }
.brand-tag{ font-size:.87rem; color:var(--text-muted); margin:3px 0 0; }
.pill{ display:inline-flex; align-items:center; gap:8px; padding:8px 15px; border-radius:999px;
  font-size:.83rem; font-weight:600; white-space:nowrap; border:1px solid var(--border); }
.pill-dot{ width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.pill-ok{ color:var(--success); background:var(--success-soft); border-color:#1C4A3B; }
.pill-ok .pill-dot{ background:var(--success); }
.pill-warn{ color:var(--warn); background:var(--warn-soft); border-color:#4A3A16; }
.pill-warn .pill-dot{ background:var(--warn); }

/* Sections */
.sec{ display:flex; align-items:baseline; gap:12px; margin:26px 0 12px; }
.sec-num{ font-family:'JetBrains Mono',monospace; font-size:.78rem; font-weight:600;
  color:var(--primary); background:var(--primary-soft); padding:4px 9px; border-radius:6px; }
.sec-title{ font-size:1.15rem; font-weight:700; color:var(--text); margin:0; }
.sec-sub{ font-size:.87rem; color:var(--text-muted); margin:0 0 14px; }

/* ACTION CARD — the operator's primary output */
.action{ border-radius:var(--radius-lg); padding:22px 26px; border:1px solid var(--border);
  border-left-width:6px; margin-bottom:14px; }
.action.ok{ background:var(--success-soft); border-left-color:var(--success); }
.action.warn{ background:var(--warn-soft); border-left-color:var(--warn); }
.action.bad{ background:var(--danger-soft); border-left-color:var(--danger); }
.action-top{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:10px; }
.action-head{ font-size:1.32rem; font-weight:800; color:var(--text); margin:0; letter-spacing:-.02em; }
.sev-chip{ font-size:.72rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase;
  padding:5px 11px; border-radius:6px; }
.sev-ok{ background:var(--success); color:#07120E; }
.sev-warn{ background:var(--warn); color:#1A1305; }
.sev-bad{ background:var(--danger); color:#2A0A0C; }
.action-what{ font-size:1.02rem; font-weight:600; color:var(--text); margin:0 0 4px; }
.action-meta{ font-size:.88rem; color:var(--text-muted); margin:0; }
.action-meta b{ color:var(--text); font-weight:600; }

/* Plain-English explanation */
.why{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-md);
  padding:18px 20px; box-shadow:var(--shadow); }
.why p{ color:var(--text); font-size:.93rem; line-height:1.65; margin:0 0 9px; }
.why p:last-child{ margin-bottom:0; }
.eyebrow{ text-transform:uppercase; letter-spacing:.09em; font-size:.7rem; font-weight:700;
  color:var(--primary); margin:0 0 9px; }

/* Cards / facts */
.card{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-md);
  padding:18px 20px; box-shadow:var(--shadow); height:100%; }
.card p{ color:var(--text); font-size:.9rem; line-height:1.55; margin:0; }
.fact{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-md);
  padding:15px 18px; box-shadow:var(--shadow); height:100%; }
.fact-label{ color:var(--text-muted); font-size:.72rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.06em; margin:0 0 6px; }
.fact-value{ font-family:'JetBrains Mono',monospace; font-size:1.12rem; font-weight:600;
  color:var(--text); word-break:break-word; margin:0; line-height:1.3; }
.fact-value .sub{ font-size:.8rem; color:var(--text-muted); font-weight:500; }
.panel{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-md);
  padding:20px; box-shadow:var(--shadow); }
.legend-row{ display:flex; align-items:center; gap:10px; padding:9px 0; border-bottom:1px solid var(--border); }
.legend-row:last-child{ border-bottom:none; }
.legend-chip{ width:10px; height:10px; border-radius:3px; flex-shrink:0; }
.legend-name{ font-weight:600; font-size:.88rem; color:var(--text); min-width:62px; }
.legend-desc{ font-size:.83rem; color:var(--text-muted); }
.codeblock{ background:var(--surface-2); border:1px solid var(--border); border-radius:8px;
  padding:12px 14px; font-family:'JetBrains Mono',monospace; font-size:.82rem;
  color:var(--text); line-height:1.7; }
.diagram{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-md);
  padding:16px 18px 10px; box-shadow:var(--shadow); }

/* Streamlit widgets */
[data-testid="stMetric"]{ background:var(--surface); border:1px solid var(--border);
  border-radius:var(--radius-md); padding:14px 16px; box-shadow:var(--shadow); }
[data-testid="stMetricLabel"] p{ color:var(--text-muted) !important; font-size:.72rem !important;
  font-weight:700 !important; text-transform:uppercase; letter-spacing:.06em; }
[data-testid="stMetricValue"]{ color:var(--text) !important; }
div[data-testid="stFileUploader"] section{ border:1.5px dashed var(--border) !important;
  border-radius:var(--radius-md) !important; background:var(--surface-2) !important; }
div[data-testid="stFileUploader"] section small{ color:var(--text-muted) !important; }
button[kind="primary"]{ border-radius:var(--radius-sm) !important; font-weight:600 !important; }
div[data-testid="stDownloadButton"] button{ width:100%; border-radius:var(--radius-sm) !important;
  font-weight:600 !important; }
[data-testid="stExpander"]{ border:1px solid var(--border) !important;
  border-radius:var(--radius-md) !important; background:var(--surface) !important; }
hr{ border-color:var(--border) !important; }
.footer-note{ margin-top:34px; padding-top:16px; border-top:1px solid var(--border);
  font-size:.79rem; color:var(--text-muted); text-align:center; }
</style>
""", unsafe_allow_html=True)

st.session_state.setdefault("batch_key", 0)
st.session_state.setdefault("batch_out", None)


def clear_batch():
    st.session_state["batch_out"] = None
    st.session_state["batch_key"] += 1


def section(num, title, sub=None):
    st.markdown(
        f'<div class="sec"><span class="sec-num">{num}</span><p class="sec-title">{title}</p></div>'
        + (f'<p class="sec-sub">{sub}</p>' if sub else ""),
        unsafe_allow_html=True,
    )


def fact_card(label, value, sub=None):
    sub_html = f'<span class="sub"> {sub}</span>' if sub else ""
    st.markdown(f'<div class="fact"><p class="fact-label">{label}</p>'
                f'<p class="fact-value">{value}{sub_html}</p></div>', unsafe_allow_html=True)


def validate(df):
    return df.shape[1] >= REQUIRED_COLUMNS and df.shape[0] > 0


@st.cache_data(show_spinner=False)
def analyse(file_bytes, _name):
    df = pd.read_csv(io.BytesIO(file_bytes))
    if not validate(df):
        return df, None
    pred, conf, features, probs = predict_rail_file(df, MODEL_PATH)
    return df, {"pred": pred, "conf": conf, "features": features, "probs": probs}


def side_signals(df):
    s1 = df.iloc[:, side_dataframe_columns(1, True)].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0).to_numpy()
    s2 = df.iloc[:, side_dataframe_columns(2, True)].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0).to_numpy()
    return s1, s2


def spectrum(x):
    x = np.asarray(x, dtype=float) - np.mean(x)
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freq = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE_HZ)
    return pd.DataFrame({"Frequency (Hz)": freq, "Magnitude": mag}).set_index("Frequency (Hz)")


meta = load_metadata(MODEL_PATH) if MODEL_PATH.exists() else None
baselines = (meta or {}).get("healthy_baselines") or {}
best_name = meta.get("selected_model", "") if meta else ""
best_cv = meta.get("cv_scores_macro_f1", {}).get(best_name, {}) if meta else {}

# Status is phrased for an operator: "is this thing working and trustworthy",
# not a model metric. Macro F1 lives in section 03 for the engineers.
if meta:
    pill = (f'<span class="pill pill-ok"><span class="pill-dot"></span>System ready · '
            f'validated on {meta.get("n_training_files", "?")} recordings</span>')
else:
    pill = '<span class="pill pill-warn"><span class="pill-dot"></span>No trained model yet</span>'

st.markdown(f"""
<div class="topbar">
  <div class="brand">
    <div class="brand-badge">🚆</div>
    <div>
      <p class="brand-name">RailGuard</p>
      <p class="brand-tag">Rail corrugation detection from axle-box vibration</p>
    </div>
  </div>
  {pill}
</div>
""", unsafe_allow_html=True)

view = st.radio("View", ["Operator", "Engineer"], horizontal=True, label_visibility="collapsed",
                help="Operator: what to do. Engineer: the signals and features behind it.")
engineer = view == "Engineer"

if meta and not baselines:
    st.caption("Severity grading is unavailable — this model was trained before baseline statistics were "
               "added. Re-run training/train_rail.py to enable it.")

# ============================================== 01 · DIAGNOSE
section("01", "Diagnose a recording",
        "Upload a one-second axle-box CSV. The result tells you what to do, not just what was measured.")

left, right = st.columns([4, 8], gap="large")

with left:
    uploaded = st.file_uploader("Rail Corrugation CSV", type=["csv"], key="single", label_visibility="collapsed")
    df = outcome = error = None
    if uploaded and MODEL_PATH.exists():
        with st.spinner("Analysing Side I / Side II..."):
            try:
                df, outcome = analyse(uploaded.getvalue(), uploaded.name)
            except Exception as exc:
                error = str(exc)

    if uploaded and not MODEL_PATH.exists():
        st.warning("No trained model at models/rail_model.joblib. Run training/train_rail.py first.")
    elif error:
        st.error(f"Could not analyse this file: {error}")
    elif df is not None:
        m1, m2 = st.columns(2)
        m1.metric("Samples", f"{df.shape[0]:,}")
        m2.metric("Duration", f"{df.shape[0] / SAMPLE_RATE_HZ:.2f} s")
        if outcome is None:
            st.error(f"Needs speed + 128 sensor channels ({REQUIRED_COLUMNS} columns); this file has {df.shape[1]}.")
        else:
            verdict = assess(outcome["pred"], outcome["conf"], outcome["features"].iloc[0], baselines)
            st.metric("Reliability", verdict["confidence_label"])
            st.caption(verdict["confidence_note"])

            # Plain-language side comparison — the one measurement an operator
            # can sanity-check the verdict against.
            fr = outcome["features"].iloc[0]
            v1, v2 = float(fr["side1_rms"]), float(fr["side2_rms"])
            hi = "Side I" if v1 > v2 else "Side II"
            factor = max(v1, v2) / max(min(v1, v2), 1e-12)
            bar1 = int(round(100 * v1 / max(v1, v2)))
            bar2 = int(round(100 * v2 / max(v1, v2)))
            c1 = "#FF6B72" if (hi == "Side I" and verdict["side"]) else "#5B7CFF"
            c2 = "#FF6B72" if (hi == "Side II" and verdict["side"]) else "#5B7CFF"
            st.markdown(f"""<div class="panel" style="margin-top:14px">
              <p class="eyebrow">Vibration level by rail</p>
              <div style="margin-bottom:12px">
                <div style="display:flex;justify-content:space-between;font-size:.84rem;color:var(--text);
                     font-weight:600;margin-bottom:5px"><span>Side I</span><span>{v1:.3f} m/s²</span></div>
                <div style="background:var(--surface-2);border-radius:4px;height:9px">
                  <div style="width:{bar1}%;background:{c1};height:9px;border-radius:4px"></div></div>
              </div>
              <div>
                <div style="display:flex;justify-content:space-between;font-size:.84rem;color:var(--text);
                     font-weight:600;margin-bottom:5px"><span>Side II</span><span>{v2:.3f} m/s²</span></div>
                <div style="background:var(--surface-2);border-radius:4px;height:9px">
                  <div style="width:{bar2}%;background:{c2};height:9px;border-radius:4px"></div></div>
              </div>
              <p style="font-size:.82rem;color:var(--text-muted);margin:12px 0 0">
                {hi} is {factor:.1f}× the other rail.</p>
            </div>""", unsafe_allow_html=True)

with right:
    if outcome:
        verdict = assess(outcome["pred"], outcome["conf"], outcome["features"].iloc[0], baselines)
        tone = verdict["tone"]
        chip = {"ok": "sev-ok", "warn": "sev-warn", "bad": "sev-bad"}[tone]
        pct = f' · {verdict["percentile_text"]}' if verdict["percentile_text"] else ""
        st.markdown(f"""<div class="action {tone}">
          <div class="action-top">
            <p class="action-head">{verdict['headline']}</p>
            <span class="sev-chip {chip}">{verdict['label']}</span>
          </div>
          <p class="action-what">→ {verdict['action']}</p>
          <p class="action-meta"><b>Act by:</b> {verdict['timeframe']}{pct}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="diagram">{train_diagram(verdict["side"])}</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="why"><p class="eyebrow">Why the system says this</p>'
                    + "".join(f"<p>{line}</p>" for line in explain(outcome["pred"], outcome["features"].iloc[0]))
                    + "</div>", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="panel">
          <p class="eyebrow">What you'll get back</p>
          <div class="legend-row"><span class="legend-chip" style="background:#35D6A0"></span>
            <span class="legend-name">Normal</span><span class="legend-desc">Both rails healthy — no action needed</span></div>
          <div class="legend-row"><span class="legend-chip" style="background:#FF6B72"></span>
            <span class="legend-name">Side I</span><span class="legend-desc">Corrugation on the rail at axle positions 1, 3, 5, 7</span></div>
          <div class="legend-row"><span class="legend-chip" style="background:#F0B34A"></span>
            <span class="legend-name">Side II</span><span class="legend-desc">Corrugation on the rail at axle positions 2, 4, 6, 8</span></div>
        </div>""", unsafe_allow_html=True)
        st.write("")
        st.markdown(f'<div class="diagram">{train_diagram(None)}</div>', unsafe_allow_html=True)

# Engineer-only detail for the same recording
if outcome and engineer and df is not None:
    f = outcome["features"].iloc[0]
    st.write("")
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Side I RMS", f"{float(f['side1_rms']):.4f}")
    e2.metric("Side II RMS", f"{float(f['side2_rms']):.4f}")
    e3.metric("Side I/II ratio", f"{float(f['side_rms_ratio']):.3f}")
    e4.metric("Est. speed", f"{float(f['speed_signal_mean']):.2f} m/s")
    e5.metric("Model confidence", f"{outcome['conf']:.1%}")

    g1, g2, g3 = st.columns([1, 1, 1], gap="large")
    with g1:
        st.caption("Class probabilities")
        probs = outcome["probs"]
        st.bar_chart(pd.DataFrame({"Probability": [probs.get("Normal", 0), probs.get("Side I", 0), probs.get("Side II", 0)]},
                                  index=["Normal", "Side I", "Side II"]), height=220)
    sig1, sig2 = side_signals(df)
    with g2:
        st.caption("Mean vibration waveform — first 0.2 s")
        st.line_chart(pd.DataFrame({"Side I": sig1[:2000], "Side II": sig2[:2000]}),
                      height=220, color=[SIDE1_COLOR, SIDE2_COLOR])
    with g3:
        st.caption("Frequency spectrum — 1 to 2000 Hz")
        spec = pd.DataFrame({"Side I": spectrum(sig1)["Magnitude"], "Side II": spectrum(sig2)["Magnitude"]})
        spec = spec.loc[(spec.index >= 1) & (spec.index <= 2000)]
        st.line_chart(spec, height=220, color=[SIDE1_COLOR, SIDE2_COLOR])

    with st.expander("Inspect raw recording"):
        st.dataframe(df.head(15), use_container_width=True)

if outcome:
    st.download_button(
        "Download prediction CSV",
        pd.DataFrame([{"file_id": uploaded.name, "prediction": outcome["pred"]}]).to_csv(index=False).encode(),
        "rail_predictions.csv", "text/csv",
    )

# ============================================== 02 · BATCH TRIAGE
st.divider()
section("02", "Fleet triage",
        "Run many recordings at once. Results are ordered worst-first so the most urgent rail comes to the top.")

b_left, b_right = st.columns([4, 8], gap="large")
with b_left:
    files = st.file_uploader("Upload Rail CSV files", type=["csv"], accept_multiple_files=True,
                             key=f"batch_{st.session_state['batch_key']}")
    ba, bc = st.columns([2, 1])
    with ba:
        run_batch = st.button("Analyse all recordings", type="primary", use_container_width=True,
                              disabled=not (files and MODEL_PATH.exists()))
    with bc:
        st.button("Clear all", use_container_width=True, on_click=clear_batch,
                  disabled=not (files or st.session_state.get("batch_out") is not None))
    if files:
        st.caption(f"{len(files)} file(s) selected")
    if files and not MODEL_PATH.exists():
        st.warning("No trained model at models/rail_model.joblib.")

with b_right:
    if run_batch and files:
        rows = []
        progress = st.progress(0, text="Starting...")
        for idx, fobj in enumerate(files):
            try:
                d = pd.read_csv(fobj)
                if not validate(d):
                    raise ValueError(f"expected {REQUIRED_COLUMNS} columns, got {d.shape[1]}")
                pred, conf, feats, _ = predict_rail_file(d, MODEL_PATH)
                v = assess(pred, conf, feats.iloc[0], baselines)
                rows.append({"file_id": fobj.name, "prediction": pred, "severity": v["label"],
                             "action": v["action"], "act_by": v["timeframe"],
                             "reliability": v["confidence_label"], "_rank": SEVERITY_RANK[v["key"]],
                             "status": "OK"})
            except Exception as exc:
                rows.append({"file_id": fobj.name, "prediction": "ERROR", "severity": "—", "action": str(exc),
                             "act_by": "—", "reliability": "—", "_rank": -1, "status": str(exc)})
            progress.progress((idx + 1) / len(files), text=f"Analysed {idx + 1} of {len(files)}")
        progress.empty()
        st.session_state["batch_out"] = pd.DataFrame(rows).sort_values("_rank", ascending=False)

    out = st.session_state.get("batch_out")
    if out is not None:
        needs_action = int((out["_rank"] >= 2).sum())
        total_ok = int((out.status == "OK").sum())
        if needs_action:
            st.markdown(f'<div class="action bad"><div class="action-top">'
                        f'<p class="action-head">{needs_action} of {total_ok} recordings need attention</p></div>'
                        f'<p class="action-meta">Listed worst-first below. Start at the top.</p></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="action ok"><div class="action-top">'
                        f'<p class="action-head">All {total_ok} recordings clear</p></div>'
                        f'<p class="action-meta">No rail requires action beyond routine monitoring.</p></div>',
                        unsafe_allow_html=True)

        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Normal", int((out.prediction == "Normal").sum()))
        n2.metric("Side I", int((out.prediction == "Side I").sum()))
        n3.metric("Side II", int((out.prediction == "Side II").sum()))
        n4.metric("Failed", int((out.status != "OK").sum()))

        display_cols = ["file_id", "prediction", "severity", "act_by", "reliability"]
        if engineer:
            display_cols.insert(3, "action")
        st.dataframe(out[display_cols], use_container_width=True, hide_index=True, height=300,
                     column_config={"file_id": "File", "prediction": "Rail", "severity": "Severity",
                                    "act_by": "Act by", "reliability": "Reliability", "action": "Recommended action"})
        st.download_button(
            "Download submission-ready rail_predictions.csv",
            out[out.status == "OK"][["file_id", "prediction"]].to_csv(index=False).encode(),
            "rail_predictions.csv", "text/csv", use_container_width=True,
        )
    else:
        st.markdown('<div class="panel"><p class="eyebrow">Output format</p>'
                    '<p style="color:var(--text);font-size:.89rem;margin:0 0 10px">The downloaded file carries '
                    'exactly the two columns the competition expects, one row per file:</p>'
                    '<div class="codeblock">file_id,prediction<br>Test1.csv,Normal<br>'
                    'Test2.csv,Side II<br>Test3.csv,Side I</div></div>', unsafe_allow_html=True)

# ============================================== 03 · METHOD
st.divider()
section("03", "How it works",
        "Plain-language background for anyone new to the system, plus the validation detail behind it.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""<div class="card"><p class="eyebrow">What corrugation is</p>
    <p>A repeating wave-like wear pattern that forms along the rail head. It causes the squealing and
    rumbling you hear, accelerates damage to track and vehicle, and is fixed by grinding the rail.</p></div>""",
                unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="card"><p class="eyebrow">How it's detected</p>
    <p>Accelerometers on each axle box measure vibration as the train runs. Corrugation on one rail makes
    that side vibrate measurably harder than the other — an imbalance the model is trained to recognise.</p></div>""",
                unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="card"><p class="eyebrow">Side I vs Side II</p>
    <p>The two rails under the train. Axle positions 1, 3, 5, 7 run on Side I; positions 2, 4, 6, 8 run on
    Side II. Each rail is judged separately, so one can be faulty while the other is fine.</p></div>""",
                unsafe_allow_html=True)
with c4:
    st.markdown("""<div class="card"><p class="eyebrow">What it can't do</p>
    <p>It flags which rail is affected for a one-second recording — not the exact track location, and not
    the remaining life of the rail. Severity bands are an operating policy, not a certified standard.</p></div>""",
                unsafe_allow_html=True)

if meta and engineer:
    st.write("")
    st.markdown('<p class="eyebrow">Validation detail</p>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        fact_card("Training files", meta.get("n_training_files", "—"))
    with f2:
        fact_card("Model", best_name.replace("_", " ").title())
    with f3:
        fact_card("CV macro F1", f"{best_cv.get('mean', 0):.4f}", f"± {best_cv.get('std', 0):.4f}")
    with f4:
        dist = meta.get("class_distribution", {})
        fact_card("Class balance", ", ".join(f"{k}: {v}" for k, v in dist.items()) if dist else "—")

    st.write("")
    v1, v2 = st.columns(2, gap="large")
    with v1:
        st.caption("Model comparison — 5-fold stratified CV, macro F1")
        cmp_rows = meta.get("cv_scores_macro_f1", {})
        st.bar_chart(pd.DataFrame({"Macro F1": {k.replace("_", " ").title(): v["mean"] for k, v in cmp_rows.items()}}),
                     height=280, horizontal=True)
    with v2:
        top_feats = meta.get("top_feature_importances", [])
        if top_feats:
            st.caption("What the model weighs most — in plain terms")
            feat_df = pd.DataFrame(top_feats).head(10)
            feat_df["Measurement"] = feat_df["feature"].map(describe_feature)
            st.dataframe(feat_df[["Measurement", "importance"]], use_container_width=True, hide_index=True, height=280,
                         column_config={"importance": st.column_config.ProgressColumn(
                             "Weight", format="%.3f", min_value=0.0,
                             max_value=float(feat_df["importance"].max()))})
elif not meta:
    st.info("No trained model yet. Run `python -m training.train_rail --train-dir <Train folder> "
            "--labels <Train_Labels.csv>` to train and populate this section.")

st.caption("Decision-support tool. Severity bands and maintenance timeframes are a configurable operating "
           "policy for this prototype, not certified engineering thresholds — confirm against your own standards.")
st.markdown('<div class="footer-note">RailGuard — NebulaX Train Condition Monitoring hackathon</div>',
            unsafe_allow_html=True)
