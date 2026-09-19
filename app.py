import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.rail_channels import side_dataframe_columns
from src.rail_features import SAMPLE_RATE_HZ
from src.rail_inference import load_metadata, predict_rail_file

st.set_page_config(page_title="RailGuard", page_icon="🚆", layout="wide", initial_sidebar_state="collapsed")

MODEL_PATH = Path("models/rail_model.joblib")
REQUIRED_COLUMNS = 129
SIDE1_COLOR = "#5B7CFF"
SIDE2_COLOR = "#35D6A0"

# One dark palette only. No prefers-color-scheme media query: it previously
# fought with Streamlit's own theme and produced dark-on-dark text whenever
# the two disagreed. These values mirror .streamlit/config.toml exactly.
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

html, body, [class*="css"], .stMarkdown, p, span, div, label {
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
}
code, pre, [data-testid="stMetricValue"]{ font-family:'JetBrains Mono','SFMono-Regular',monospace !important; }

.stApp{ background:var(--bg); }
.block-container{ max-width:1560px; padding:1.5rem 2.2rem 4rem; }
h1,h2,h3,h4{ letter-spacing:-0.02em; color:var(--text); }

/* ---------- Top bar ---------- */
.topbar{ display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px;
  padding:18px 24px; margin-bottom:18px; border:1px solid var(--border); border-radius:var(--radius-lg);
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

/* ---------- Section headers ---------- */
.sec{ display:flex; align-items:baseline; gap:12px; margin:26px 0 14px; }
.sec-num{ font-family:'JetBrains Mono',monospace; font-size:.78rem; font-weight:600;
  color:var(--primary); background:var(--primary-soft); padding:4px 9px; border-radius:6px; }
.sec-title{ font-size:1.15rem; font-weight:700; color:var(--text); margin:0; }
.sec-sub{ font-size:.87rem; color:var(--text-muted); margin:0 0 14px; }

/* ---------- Cards ---------- */
.card{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-md);
  padding:18px 20px; box-shadow:var(--shadow); height:100%; }
.card p{ color:var(--text); font-size:.9rem; line-height:1.55; margin:0; }
.eyebrow{ text-transform:uppercase; letter-spacing:.09em; font-size:.7rem; font-weight:700;
  color:var(--primary); margin:0 0 8px; }
.fact{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-md);
  padding:15px 18px; box-shadow:var(--shadow); height:100%; }
.fact-label{ color:var(--text-muted); font-size:.72rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.06em; margin:0 0 6px; }
.fact-value{ font-family:'JetBrains Mono',monospace; font-size:1.12rem; font-weight:600;
  color:var(--text); word-break:break-word; margin:0; line-height:1.3; }
.fact-value .sub{ font-size:.8rem; color:var(--text-muted); font-weight:500; }

/* ---------- Dropzone panel ---------- */
.panel{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-md);
  padding:20px; box-shadow:var(--shadow); }
.legend-row{ display:flex; align-items:center; gap:10px; padding:9px 0; border-bottom:1px solid var(--border); }
.legend-row:last-child{ border-bottom:none; }
.legend-chip{ width:10px; height:10px; border-radius:3px; flex-shrink:0; }
.legend-name{ font-weight:600; font-size:.88rem; color:var(--text); min-width:62px; }
.legend-desc{ font-size:.83rem; color:var(--text-muted); }

/* ---------- Result banner ---------- */
.result{ display:flex; align-items:center; gap:16px; padding:20px 22px;
  border-radius:var(--radius-lg); border:1px solid var(--border); margin-bottom:14px; }
.result-icon{ width:48px; height:48px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-size:1.4rem; font-weight:700; flex-shrink:0; color:#0D1117; }
.result.ok{ background:var(--success-soft); border-color:#1C4A3B; }
.result.ok .result-icon{ background:var(--success); }
.result.bad{ background:var(--danger-soft); border-color:#5A2A2E; }
.result.bad .result-icon{ background:var(--danger); }
.result h2{ margin:0; font-size:1.2rem; color:var(--text); }
.result p{ margin:3px 0 0; font-size:.89rem; color:var(--text-muted); }

/* ---------- Streamlit widget overrides ---------- */
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
[data-testid="stExpander"]{ border:1px solid var(--border) !important; border-radius:var(--radius-md) !important;
  background:var(--surface) !important; }
hr{ border-color:var(--border) !important; }
.codeblock{ background:var(--surface-2); border:1px solid var(--border); border-radius:8px;
  padding:12px 14px; font-family:'JetBrains Mono',monospace; font-size:.82rem;
  color:var(--text); line-height:1.7; }
.footer-note{ margin-top:34px; padding-top:16px; border-top:1px solid var(--border);
  font-size:.79rem; color:var(--text-muted); text-align:center; }
</style>
""", unsafe_allow_html=True)


def section(num, title, sub=None):
    st.markdown(
        f'<div class="sec"><span class="sec-num">{num}</span><p class="sec-title">{title}</p></div>'
        + (f'<p class="sec-sub">{sub}</p>' if sub else ""),
        unsafe_allow_html=True,
    )


def fact_card(label, value, sub=None):
    sub_html = f'<span class="sub"> {sub}</span>' if sub else ""
    st.markdown(
        f'<div class="fact"><p class="fact-label">{label}</p>'
        f'<p class="fact-value">{value}{sub_html}</p></div>',
        unsafe_allow_html=True,
    )


def validate(df):
    return df.shape[1] >= REQUIRED_COLUMNS and df.shape[0] > 0


@st.cache_data(show_spinner=False)
def analyse(file_bytes, _name):
    """Cached so toggling an expander doesn't re-run feature extraction."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    if not validate(df):
        return df, None
    pred, conf, features, probs = predict_rail_file(df, MODEL_PATH)
    return df, {"pred": pred, "conf": conf, "features": features, "probs": probs}


def side_signals(df):
    s1 = df.iloc[:, side_dataframe_columns(1, vibration_only=True)].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0).to_numpy()
    s2 = df.iloc[:, side_dataframe_columns(2, vibration_only=True)].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0).to_numpy()
    return s1, s2


def spectrum(x):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freq = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE_HZ)
    return pd.DataFrame({"Frequency (Hz)": freq, "Magnitude": mag}).set_index("Frequency (Hz)")


# ------------------------------------------------------------------ top bar
meta = load_metadata(MODEL_PATH) if MODEL_PATH.exists() else None
best_name = meta.get("selected_model", "") if meta else ""
best_cv = meta.get("cv_scores_macro_f1", {}).get(best_name, {}) if meta else {}

if meta:
    pill = f'<span class="pill pill-ok"><span class="pill-dot"></span>Model live · macro F1 {best_cv.get("mean", 0):.3f}</span>'
else:
    pill = '<span class="pill pill-warn"><span class="pill-dot"></span>No trained model yet</span>'

st.markdown(f"""
<div class="topbar">
  <div class="brand">
    <div class="brand-badge">🚆</div>
    <div>
      <p class="brand-name">RailGuard</p>
      <p class="brand-tag">Axle-box vibration analysis for rail corrugation monitoring</p>
    </div>
  </div>
  {pill}
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------- stats strip
if meta:
    dist = meta.get("class_distribution", {})
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        fact_card("Training files", meta.get("n_training_files", "—"))
    with s2:
        fact_card("Model", best_name.replace("_", " ").title())
    with s3:
        fact_card("CV macro F1", f"{best_cv.get('mean', 0):.4f}", f"± {best_cv.get('std', 0):.4f}")
    with s4:
        fact_card("Features per file", f"{len(meta.get('feature_names', [])):,}")

# ================================================= 01 · SINGLE DIAGNOSIS
section("01", "Diagnose a recording", "Upload a one-second axle-box CSV (10,000 samples × 129 columns). Analysis runs automatically.")

left, right = st.columns([5, 7], gap="large")

with left:
    uploaded = st.file_uploader("Rail Corrugation CSV", type=["csv"], key="single", label_visibility="collapsed")
    if uploaded and MODEL_PATH.exists():
        with st.spinner("Extracting features and analysing Side I / Side II..."):
            try:
                df, outcome = analyse(uploaded.getvalue(), uploaded.name)
                error = None
            except Exception as exc:
                df, outcome, error = None, None, str(exc)
    else:
        df, outcome, error = None, None, None

    if uploaded and not MODEL_PATH.exists():
        st.warning("No trained model at models/rail_model.joblib. Run training/train_rail.py first.")
    elif error:
        st.error(f"Could not analyse this file: {error}")
    elif df is not None:
        m1, m2 = st.columns(2)
        m1.metric("Samples", f"{df.shape[0]:,}")
        m2.metric("Columns", df.shape[1])
        m3, m4 = st.columns(2)
        m3.metric("Duration", f"{df.shape[0] / SAMPLE_RATE_HZ:.2f} s")
        if outcome:
            m4.metric("Confidence", f"{outcome['conf']:.1%}")
        else:
            m4.metric("Status", "Invalid")
        if outcome is None:
            st.error(f"Needs speed + 128 sensor channels ({REQUIRED_COLUMNS} columns); this file has {df.shape[1]}.")

with right:
    if outcome:
        ok = outcome["pred"] == "Normal"
        st.markdown(f"""<div class="result {'ok' if ok else 'bad'}">
          <div class="result-icon">{'✓' if ok else '!'}</div>
          <div>
            <h2>{'Normal condition' if ok else 'Corrugation alert — ' + outcome['pred']}</h2>
            <p>{'No corrugation detected on either rail side.' if ok else 'Prioritise inspection of the ' + outcome['pred'] + ' rail.'}</p>
          </div>
        </div>""", unsafe_allow_html=True)

        p1, p2 = st.columns(2)
        with p1:
            st.caption("Class probabilities")
            probs = outcome["probs"]
            st.bar_chart(
                pd.DataFrame({"Probability": [probs.get("Normal", 0), probs.get("Side I", 0), probs.get("Side II", 0)]},
                             index=["Normal", "Side I", "Side II"]),
                height=210,
            )
        with p2:
            st.caption("Side RMS (m/s²)")
            f = outcome["features"].iloc[0]
            st.bar_chart(
                pd.DataFrame({"RMS": [float(f["side1_rms"]), float(f["side2_rms"])]}, index=["Side I", "Side II"]),
                height=210,
            )
    else:
        st.markdown("""<div class="panel">
          <p class="eyebrow">What this tells you</p>
          <div class="legend-row"><span class="legend-chip" style="background:#35D6A0"></span>
            <span class="legend-name">Normal</span><span class="legend-desc">Both rails healthy — routine monitoring</span></div>
          <div class="legend-row"><span class="legend-chip" style="background:#FF6B72"></span>
            <span class="legend-name">Side I</span><span class="legend-desc">Corrugation on axle positions 1, 3, 5, 7</span></div>
          <div class="legend-row"><span class="legend-chip" style="background:#F0B34A"></span>
            <span class="legend-name">Side II</span><span class="legend-desc">Corrugation on axle positions 2, 4, 6, 8</span></div>
        </div>""", unsafe_allow_html=True)

if outcome is not None and df is not None:
    f = outcome["features"].iloc[0]
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Side I RMS", f"{float(f['side1_rms']):.4f}")
    e2.metric("Side II RMS", f"{float(f['side2_rms']):.4f}")
    e3.metric("Side I/II ratio", f"{float(f['side_rms_ratio']):.3f}")
    e4.metric("Est. speed", f"{float(f['speed_signal_mean']):.2f} m/s")
    e5.metric("Prediction", outcome["pred"])

    sig1, sig2 = side_signals(df)
    g1, g2 = st.columns(2)
    with g1:
        st.caption("Mean vibration waveform — first 0.2 s")
        st.line_chart(pd.DataFrame({"Side I": sig1[:2000], "Side II": sig2[:2000]}),
                      height=250, color=[SIDE1_COLOR, SIDE2_COLOR])
    with g2:
        st.caption("Frequency spectrum — 1 to 2000 Hz")
        spec = pd.DataFrame({"Side I": spectrum(sig1)["Magnitude"], "Side II": spectrum(sig2)["Magnitude"]})
        spec = spec.loc[(spec.index >= 1) & (spec.index <= 2000)]
        st.line_chart(spec, height=250, color=[SIDE1_COLOR, SIDE2_COLOR])

    d1, d2 = st.columns([3, 1])
    with d1:
        with st.expander("Inspect raw recording"):
            st.dataframe(df.head(15), use_container_width=True)
    with d2:
        st.download_button(
            "Download prediction CSV",
            pd.DataFrame([{"file_id": uploaded.name, "prediction": outcome["pred"]}]).to_csv(index=False).encode(),
            "rail_predictions.csv", "text/csv",
        )

# ==================================================== 02 · BATCH ANALYSIS
st.divider()
section("02", "Batch analysis", "Run every held-out file at once and export the submission-ready rail_predictions.csv.")

b_left, b_right = st.columns([5, 7], gap="large")
with b_left:
    files = st.file_uploader("Upload Rail CSV files", type=["csv"], accept_multiple_files=True, key="batch")
    run_batch = st.button("Analyse all recordings", type="primary", use_container_width=True,
                          disabled=not (files and MODEL_PATH.exists()))
    if files:
        st.caption(f"{len(files)} file(s) selected")
    if files and not MODEL_PATH.exists():
        st.warning("No trained model at models/rail_model.joblib.")

with b_right:
    if run_batch and files:
        results = []
        progress = st.progress(0, text="Starting batch analysis...")
        for idx, f in enumerate(files):
            try:
                d = pd.read_csv(f)
                if not validate(d):
                    raise ValueError(f"expected {REQUIRED_COLUMNS} columns, got {d.shape[1]}")
                pred, conf, _, _ = predict_rail_file(d, MODEL_PATH)
                results.append({"file_id": f.name, "prediction": pred, "probability": round(conf, 4), "status": "OK"})
            except Exception as exc:
                results.append({"file_id": f.name, "prediction": "ERROR", "probability": 0.0, "status": str(exc)})
            progress.progress((idx + 1) / len(files), text=f"Analysed {idx + 1} of {len(files)}")
        progress.empty()
        out = pd.DataFrame(results)
        st.session_state["batch_out"] = out

    out = st.session_state.get("batch_out")
    if out is not None:
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Normal", int((out.prediction == "Normal").sum()))
        n2.metric("Side I", int((out.prediction == "Side I").sum()))
        n3.metric("Side II", int((out.prediction == "Side II").sum()))
        n4.metric("Failed", int((out.status != "OK").sum()))
        st.dataframe(out, use_container_width=True, hide_index=True, height=260)
        st.download_button(
            "Download submission-ready rail_predictions.csv",
            out[out.status == "OK"][["file_id", "prediction"]].to_csv(index=False).encode(),
            "rail_predictions.csv", "text/csv", use_container_width=True,
        )
    else:
        st.markdown(
            '<div class="panel"><p class="eyebrow">Output format</p>'
            '<p style="color:var(--text);font-size:.89rem;margin:0 0 10px">Exports exactly the two columns '
            'the competition expects, one row per file:</p>'
            '<div class="codeblock">file_id,prediction<br>Test1.csv,Normal<br>'
            'Test2.csv,Side II<br>Test3.csv,Side I</div></div>',
            unsafe_allow_html=True,
        )

# ==================================================== 03 · MODEL & METHOD
st.divider()
section("03", "Model & method", "How the classifier works, how it was validated, and what drives its predictions.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""<div class="card"><p class="eyebrow">Input</p>
    <p>One-second recordings at 10 kHz: rotational speed plus 128 axle-box vibration/shock channels
    (8 cars × 8 positions × vibration+shock).</p></div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="card"><p class="eyebrow">Physical layout</p>
    <p>Axle positions 1, 3, 5, 7 → Side I. Positions 2, 4, 6, 8 → Side II. Each side is judged
    independently from the same recording.</p></div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="card"><p class="eyebrow">Pipeline</p>
    <p>Per-channel time/frequency features + per-side aggregates (RMS, crest factor, spectral bands,
    speed-normalised wavelength) → best-of-3 benchmarked classifier.</p></div>""", unsafe_allow_html=True)
with c4:
    st.markdown("""<div class="card"><p class="eyebrow">Scoring metric</p>
    <p>Macro F1 across the three classes — credits detecting the rare Side I / Side II cases, not just
    the common Normal case, unlike plain accuracy.</p></div>""", unsafe_allow_html=True)

if meta:
    st.write("")
    v1, v2 = st.columns(2, gap="large")
    with v1:
        st.caption("Model comparison — 5-fold stratified CV, macro F1")
        cmp_rows = meta.get("cv_scores_macro_f1", {})
        st.bar_chart(pd.DataFrame({"Macro F1": {k.replace("_", " ").title(): v["mean"] for k, v in cmp_rows.items()}}),
                     height=280, horizontal=True)
        dist = meta.get("class_distribution", {})
        st.caption(f"Class balance — {', '.join(f'{k}: {v}' for k, v in dist.items())}")
    with v2:
        top_feats = meta.get("top_feature_importances", [])
        if top_feats:
            st.caption("Top features driving predictions — ranked by model weight")
            feat_df = pd.DataFrame(top_feats).head(10)
            st.dataframe(
                feat_df, use_container_width=True, hide_index=True, height=280,
                column_config={
                    "feature": st.column_config.TextColumn("Feature"),
                    "importance": st.column_config.ProgressColumn(
                        "Weight", format="%.3f", min_value=0.0,
                        max_value=float(feat_df["importance"].max()),
                    ),
                },
            )
else:
    st.info("No trained model yet. Run `python -m training.train_rail --train-dir <Train folder> --labels <Train_Labels.csv>` to train and populate this section.")

st.caption("Model probabilities are not calibrated diagnostic certainty. This is a condition-monitoring decision-support tool.")
st.markdown('<div class="footer-note">RailGuard — NebulaX Train Condition Monitoring hackathon</div>', unsafe_allow_html=True)
