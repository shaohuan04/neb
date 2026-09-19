from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.rail_channels import side_dataframe_columns
from src.rail_features import SAMPLE_RATE_HZ, extract_rail_features
from src.rail_inference import load_metadata, predict_rail_file

st.set_page_config(page_title="RailGuard", page_icon="🚆", layout="wide", initial_sidebar_state="collapsed")

MODEL_PATH = Path("models/rail_model.joblib")
REQUIRED_COLUMNS = 129

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

:root{
  --bg:#F5F6FA; --surface:#FFFFFF; --border:#E4E6F0;
  --text:#12141D; --text-muted:#5B5F73;
  --primary:#3552F0; --primary-soft:#EEF1FE;
  --success:#12875D; --success-soft:#E9F9F1;
  --danger:#D1373F; --danger-soft:#FCEBEC;
  --radius-sm:10px; --radius-md:16px; --radius-lg:22px;
  --shadow-sm:0 1px 2px rgba(18,20,29,.06);
  --shadow-md:0 12px 28px -10px rgba(18,20,29,.20);
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#12141D; --surface:#1B1E2B; --border:#2B2F42;
    --text:#F2F3F8; --text-muted:#9FA3B8;
    --primary:#7C8EFF; --primary-soft:#222A55;
    --success:#3ED9A0; --success-soft:#0F2E24;
    --danger:#FF7A7E; --danger-soft:#3A1A1D;
  }
}

html, body, [class*="css"] { font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important; }
code, pre, [data-testid="stMetricValue"] { font-family:'JetBrains Mono','SFMono-Regular',monospace !important; }

.block-container{max-width:1180px; padding-top:1.6rem; padding-bottom:4rem;}
h1,h2,h3{ letter-spacing:-0.02em; }

/* ---- Top bar ---- */
.topbar{ display:flex; align-items:center; justify-content:space-between; gap:16px;
  padding-bottom:22px; margin-bottom:26px; border-bottom:1px solid var(--border); }
.brand{ display:flex; align-items:center; gap:14px; }
.brand-badge{ width:46px; height:46px; border-radius:13px; background:var(--primary);
  display:flex; align-items:center; justify-content:center; font-size:1.5rem; box-shadow:var(--shadow-sm); flex-shrink:0; }
.brand-name{ font-size:1.4rem; font-weight:800; margin:0; line-height:1.1; }
.brand-tag{ font-size:.86rem; color:var(--text-muted); margin:2px 0 0; }
.pill{ display:inline-flex; align-items:center; gap:7px; padding:7px 14px; border-radius:999px;
  font-size:.82rem; font-weight:600; border:1px solid var(--border); background:var(--surface); white-space:nowrap; }
.pill-dot{ width:8px; height:8px; border-radius:50%; }
.pill-ok .pill-dot{ background:var(--success); } .pill-ok{ color:var(--success); border-color:var(--success-soft); background:var(--success-soft); }
.pill-warn .pill-dot{ background:#B7791F; } .pill-warn{ color:#B7791F; border-color:#FBEEDB; background:#FBF3E5; }

/* ---- Eyebrow + section headers ---- */
.eyebrow{ text-transform:uppercase; letter-spacing:.09em; font-size:.72rem; font-weight:700; color:var(--primary); margin:0 0 6px; }
.section-title{ font-size:1.05rem; font-weight:700; margin:0 0 4px; }
.section-sub{ font-size:.88rem; color:var(--text-muted); margin:0 0 16px; }

/* ---- Stepper ---- */
.stepper{ display:flex; align-items:center; gap:10px; margin-bottom:16px; }
.stepper .dot{ width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  font-size:.74rem; font-weight:700; flex-shrink:0; }
.stepper .dot.active{ background:var(--primary); color:#fff; }
.stepper .dot.pending{ background:var(--border); color:var(--text-muted); }
.stepper .line{ width:34px; height:2px; background:var(--border); }
.stepper .label{ font-size:.84rem; font-weight:600; color:var(--text-muted); }
.stepper .label.active{ color:var(--text); }

/* ---- Cards ---- */
.card{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-md);
  padding:20px 22px; box-shadow:var(--shadow-sm); margin-bottom:16px; }
.hint{ background:var(--primary-soft); border:1px solid transparent; border-radius:var(--radius-sm);
  padding:14px 16px; font-size:.88rem; color:var(--text); }

/* ---- Result banner ---- */
.result{ display:flex; align-items:center; gap:16px; padding:22px 24px; border-radius:var(--radius-lg); margin:4px 0 18px; }
.result-icon{ width:52px; height:52px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:1.5rem; flex-shrink:0; }
.result.ok{ background:var(--success-soft); } .result.ok .result-icon{ background:var(--success); color:#fff; }
.result.bad{ background:var(--danger-soft); } .result.bad .result-icon{ background:var(--danger); color:#fff; }
.result h2{ margin:0; font-size:1.25rem; } .result p{ margin:2px 0 0; color:var(--text-muted); font-size:.92rem; }

/* ---- Streamlit widget refinements ---- */
[data-testid="stMetric"]{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-sm);
  padding:14px 16px; box-shadow:var(--shadow-sm); }
[data-testid="stMetricLabel"]{ color:var(--text-muted) !important; font-size:.78rem !important; font-weight:600 !important; text-transform:uppercase; letter-spacing:.03em; }
div[data-testid="stFileUploader"] section{ border-radius:var(--radius-md) !important; border:1.5px dashed var(--border) !important; background:var(--bg) !important; }
button[kind="primary"]{ border-radius:var(--radius-sm) !important; font-weight:600 !important; box-shadow:var(--shadow-sm); }
div[data-testid="stDownloadButton"] button{ width:100%; border-radius:var(--radius-sm) !important; font-weight:600 !important; }
.stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--border); }
.stTabs [data-baseweb="tab"]{ font-weight:600; color:var(--text-muted); }
.stTabs [aria-selected="true"]{ color:var(--primary) !important; }
.footer-note{ margin-top:36px; padding-top:16px; border-top:1px solid var(--border); font-size:.8rem; color:var(--text-muted); text-align:center; }
.fact{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-sm); padding:14px 16px; box-shadow:var(--shadow-sm); height:100%; }
.fact-label{ color:var(--text-muted); font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.03em; margin:0 0 4px; }
.fact-value{ font-family:'JetBrains Mono','SFMono-Regular',monospace; font-size:1.05rem; font-weight:600; word-break:break-word; margin:0; }
</style>
""", unsafe_allow_html=True)


def fact_card(label, value):
    st.markdown(f'<div class="fact"><p class="fact-label">{label}</p><p class="fact-value">{value}</p></div>', unsafe_allow_html=True)


def validate(df):
    return df.shape[1] >= REQUIRED_COLUMNS and df.shape[0] > 0


def read_uploaded_csv(uploaded):
    try:
        return pd.read_csv(uploaded), None
    except Exception as exc:
        return None, f"Unable to read this file as CSV: {exc}"


def stepper(step_num, labels):
    parts = []
    for i, label in enumerate(labels, start=1):
        state = "active" if i == step_num else "pending"
        parts.append(f'<div class="dot {state}">{i}</div><div class="label {state}">{label}</div>')
        if i < len(labels):
            parts.append('<div class="line"></div>')
    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


meta = load_metadata(MODEL_PATH) if MODEL_PATH.exists() else None
if meta:
    best = meta.get("selected_model", "")
    cv = meta.get("cv_scores_macro_f1", {}).get(best, {})
    status_pill = f'<span class="pill pill-ok"><span class="pill-dot"></span>Model live · macro F1 {cv.get("mean", 0):.3f}</span>'
else:
    status_pill = '<span class="pill pill-warn"><span class="pill-dot"></span>No trained model yet</span>'

st.markdown(f"""
<div class="topbar">
  <div class="brand">
    <div class="brand-badge">🚆</div>
    <div>
      <p class="brand-name">RailGuard</p>
      <p class="brand-tag">Axle-box vibration analysis for rail corrugation monitoring</p>
    </div>
  </div>
  {status_pill}
</div>
""", unsafe_allow_html=True)

single_tab, batch_tab, about_tab = st.tabs(["Single diagnosis", "Batch analysis", "Model & method"])

with single_tab:
    stepper(1, ["Upload recording", "Run diagnosis"])
    st.markdown('<p class="section-title">Load a sensor recording</p><p class="section-sub">One-second axle-box vibration/shock CSV, sampled at 10 kHz.</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Rail Corrugation CSV", type=["csv"], key="single", label_visibility="collapsed")

    if not uploaded:
        st.markdown(f'<div class="hint">Drag a Rail Corrugation CSV here to start. Expected format: 10,000 samples × {REQUIRED_COLUMNS} columns (speed + 128 axle-box channels).</div>', unsafe_allow_html=True)
    else:
        df, err = read_uploaded_csv(uploaded)
        if err:
            st.error(err)
        else:
            rows, cols = df.shape
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("File", uploaded.name)
            q2.metric("Samples", f"{rows:,}")
            q3.metric("Columns", cols)
            q4.metric("Duration", f"{rows / SAMPLE_RATE_HZ:.2f} s")

            if not validate(df):
                st.error(f"File validation failed. Rail recordings require speed + 128 sensor channels ({REQUIRED_COLUMNS} columns); this file has {cols}.")
            elif not MODEL_PATH.exists():
                st.warning("Trained model is missing from models/rail_model.joblib. Run training/train_rail.py first.")
            else:
                with st.expander("Inspect raw recording"):
                    st.dataframe(df.head(15), use_container_width=True)

                st.write("")
                stepper(2, ["Upload recording", "Run diagnosis"])
                if st.button("Run rail diagnosis", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Extracting features and analysing Side I / Side II..."):
                            pred, conf, features, probs = predict_rail_file(df, MODEL_PATH)
                    except Exception as exc:
                        st.error(f"Diagnosis failed: {exc}")
                        pred = None

                    if pred is not None:
                        ok = pred == "Normal"
                        icon = "✓" if ok else "⚠"
                        title = "Normal condition" if ok else "Corrugation alert"
                        subtitle = "No corrugation class detected" if ok else f"Model localised the anomaly to {pred}"
                        st.markdown(f"""<div class="result {'ok' if ok else 'bad'}">
                            <div class="result-icon">{icon}</div>
                            <div><h2>{title}</h2><p>{subtitle}</p></div>
                        </div>""", unsafe_allow_html=True)

                        a, b, c = st.columns(3)
                        a.metric("Classification", pred)
                        a.metric("Highest model probability", f"{conf:.1%}")
                        recommendation = "Continue routine monitoring" if ok else f"Prioritise inspection of {pred} rail"
                        a.info("Action: " + recommendation)

                        with b:
                            st.markdown('<p class="section-title" style="font-size:.92rem">Class probabilities</p>', unsafe_allow_html=True)
                            p = pd.DataFrame({"Probability": [probs.get("Normal", 0), probs.get("Side I", 0), probs.get("Side II", 0)]}, index=["Normal", "Side I", "Side II"])
                            st.bar_chart(p, height=230)
                        with c:
                            st.markdown('<p class="section-title" style="font-size:.92rem">Side RMS</p>', unsafe_allow_html=True)
                            s1 = float(features.iloc[0]["side1_rms"])
                            s2 = float(features.iloc[0]["side2_rms"])
                            r = pd.DataFrame({"RMS (m/s²)": [s1, s2]}, index=["Side I", "Side II"])
                            st.bar_chart(r, height=230)

                        st.divider()
                        st.markdown('<p class="section-title">Signal evidence</p>', unsafe_allow_html=True)
                        sig1 = df.iloc[:, side_dataframe_columns(1, vibration_only=True)].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0).to_numpy()
                        sig2 = df.iloc[:, side_dataframe_columns(2, vibration_only=True)].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0).to_numpy()
                        view = pd.DataFrame({"Side I": sig1[:2000], "Side II": sig2[:2000]})
                        st.caption("Mean vibration waveform — first 0.2 s")
                        st.line_chart(view, height=230)

                        def spectrum(x):
                            x = np.asarray(x, dtype=float)
                            x = x - np.mean(x)
                            win = np.hanning(len(x))
                            mag = np.abs(np.fft.rfft(x * win))
                            freq = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE_HZ)
                            return pd.DataFrame({"Frequency (Hz)": freq, "Magnitude": mag}).set_index("Frequency (Hz)")

                        sp1, sp2 = spectrum(sig1), spectrum(sig2)
                        spec = pd.DataFrame({"Side I": sp1["Magnitude"], "Side II": sp2["Magnitude"]})
                        spec = spec.loc[(spec.index >= 1) & (spec.index <= 2000)]
                        st.caption("Frequency spectrum — 1 to 2000 Hz")
                        st.line_chart(spec, height=260)
                        st.caption("Plots provide engineering context; the classifier uses its full extracted feature set rather than these plots directly.")

                        with st.expander("Detailed engineering indicators"):
                            e1, e2, e3, e4 = st.columns(4)
                            e1.metric("Side I mean RMS", f"{s1:.4f} m/s²")
                            e2.metric("Side II mean RMS", f"{s2:.4f} m/s²")
                            e3.metric("Side I / II RMS ratio", f"{float(features.iloc[0]['side_rms_ratio']):.3f}")
                            e4.metric("Estimated speed", f"{float(features.iloc[0]['speed_signal_mean']):.2f} m/s")

                        result = pd.DataFrame([{"file_id": uploaded.name, "prediction": pred}])
                        st.download_button("Download prediction CSV", result.to_csv(index=False).encode(), "rail_predictions.csv", "text/csv")
                        st.caption("Classifier probability is not calibrated diagnostic certainty. Competition performance is evaluated using Macro F1.")

with batch_tab:
    st.markdown('<p class="section-title">Batch inference</p><p class="section-sub">Process multiple recordings in one run and export the required two-column submission file.</p>', unsafe_allow_html=True)
    files = st.file_uploader("Upload Rail CSV files", type=["csv"], accept_multiple_files=True, key="batch")
    if files:
        st.caption(f"{len(files)} file(s) selected")
    if files and MODEL_PATH.exists() and st.button("Analyse all recordings", type="primary", use_container_width=True):
        results = []
        progress = st.progress(0, text="Starting batch analysis...")
        for idx, f in enumerate(files):
            try:
                d = pd.read_csv(f)
                if not validate(d):
                    raise ValueError(f"expected {REQUIRED_COLUMNS} columns, got {d.shape[1]}")
                pred, conf, _, probs = predict_rail_file(d, MODEL_PATH)
                results.append({"file_id": f.name, "prediction": pred, "probability": conf, "status": "OK"})
            except Exception as exc:
                results.append({"file_id": f.name, "prediction": "ERROR", "probability": 0.0, "status": str(exc)})
            progress.progress((idx + 1) / len(files), text=f"Analysed {idx + 1} of {len(files)}")
        out = pd.DataFrame(results)
        st.success(f"Completed {len(files)} recordings")
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Normal", int((out.prediction == "Normal").sum()))
        n2.metric("Side I", int((out.prediction == "Side I").sum()))
        n3.metric("Side II", int((out.prediction == "Side II").sum()))
        n4.metric("Failed", int((out.status != "OK").sum()))
        st.dataframe(out, use_container_width=True, hide_index=True)
        submission = out[out.status == "OK"][["file_id", "prediction"]]
        st.download_button("Download submission-ready rail_predictions.csv", submission.to_csv(index=False).encode(), "rail_predictions.csv", "text/csv", use_container_width=True)
    elif files and not MODEL_PATH.exists():
        st.warning("Trained model is missing from models/rail_model.joblib. Run training/train_rail.py first.")

with about_tab:
    st.markdown('<p class="section-title">Model & method</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="card">
        <p class="eyebrow">Input</p>
        One-second recordings sampled at 10 kHz: rotational speed plus 128 axle-box vibration/shock channels
        (8 cars × 8 axle positions × vibration+shock).
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="card">
        <p class="eyebrow">Physical layout</p>
        Axle positions 1, 3, 5, 7 → Side I. Positions 2, 4, 6, 8 → Side II. Each side's condition is judged
        independently from the same recording.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card">
        <p class="eyebrow">Pipeline</p>
        Signal validation → per-channel time/frequency features (128 channels) + per-side aggregates
        (RMS, crest factor, spectral bands, speed-normalised wavelength) → best-of-3 benchmarked classifier
        (Random Forest / Gradient Boosting / Logistic Regression, selected by cross-validated macro F1) →
        Normal / Side I / Side II.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="card">
        <p class="eyebrow">Evaluation metric</p>
        Macro F1 across the three classes — the subsystem's official scoring metric. This credits correctly
        detecting the rare Side I / Side II cases, not just the common Normal case, unlike plain accuracy.
        </div>""", unsafe_allow_html=True)

    st.caption("Model probabilities should not be interpreted as guarantees or calibrated maintenance risk; this is a condition-monitoring decision-support tool.")

    if meta:
        st.divider()
        st.markdown('<p class="section-title">Training provenance</p>', unsafe_allow_html=True)
        cv = meta.get("cv_scores_macro_f1", {}).get(meta.get("selected_model", ""), {})
        dist = meta.get("class_distribution", {})
        m1, m2 = st.columns(2)
        with m1:
            fact_card("Training files", meta.get("n_training_files", "—"))
        with m2:
            fact_card("Selected model", meta.get("selected_model", "—"))
        st.write("")
        m3, m4 = st.columns(2)
        with m3:
            fact_card("CV macro F1 (selected model)", f"{cv.get('mean', 0):.4f} ± {cv.get('std', 0):.4f}" if cv else "—")
        with m4:
            fact_card("Class balance", ", ".join(f"{k}: {v}" for k, v in dist.items()) if dist else "—")

        cmp_rows = meta.get("cv_scores_macro_f1", {})
        if cmp_rows:
            st.markdown('<p class="section-title" style="font-size:.92rem;margin-top:20px">Model comparison (5-fold CV, macro F1)</p>', unsafe_allow_html=True)
            cmp_df = pd.DataFrame({"Macro F1": {k: v["mean"] for k, v in cmp_rows.items()}})
            st.bar_chart(cmp_df, height=220)

        top_feats = meta.get("top_feature_importances", [])
        if top_feats:
            st.markdown('<p class="section-title" style="font-size:.92rem;margin-top:20px">Top features driving predictions</p>', unsafe_allow_html=True)
            feat_df = pd.DataFrame(top_feats).set_index("feature")
            st.bar_chart(feat_df, height=320)
    else:
        st.info("No trained model metadata yet. Run `python -m training.train_rail --train-dir <Train folder> --labels <Train_Labels.csv>` to train, benchmark, and populate this section with real performance numbers.")

st.markdown('<div class="footer-note">RailGuard — NebulaX Train Condition Monitoring hackathon</div>', unsafe_allow_html=True)
