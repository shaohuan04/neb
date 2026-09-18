from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.rail_channels import side_dataframe_columns
from src.rail_features import SAMPLE_RATE_HZ, extract_rail_features
from src.rail_inference import load_metadata, predict_rail_file

st.set_page_config(page_title="Train Condition Monitoring", page_icon="🚆", layout="wide", initial_sidebar_state="expanded")
MODEL_PATH = Path("models/rail_model.joblib")
REQUIRED_COLUMNS = 129

st.markdown("""
<style>
.block-container{max-width:1380px;padding-top:1.2rem;padding-bottom:3rem}
.hero{padding:1.35rem 1.55rem;border:1px solid rgba(120,120,120,.18);border-radius:18px;margin-bottom:.8rem;background:linear-gradient(120deg,rgba(30,120,220,.08),rgba(80,80,80,.03))}
.hero p{opacity:.72}
[data-testid="stMetric"]{background:rgba(120,120,120,.06);border:1px solid rgba(120,120,120,.16);padding:14px;border-radius:13px}
.status{padding:22px;border-radius:15px;margin:.3rem 0 1rem}
.normal{background:rgba(30,160,90,.14);border:1px solid rgba(30,160,90,.35)}
.fault{background:rgba(230,70,70,.13);border:1px solid rgba(230,70,70,.35)}
.step{opacity:.72;font-size:.9rem;margin-bottom:.25rem}
.soon{padding:1.1rem 1.3rem;border-radius:14px;border:1px dashed rgba(120,120,120,.35);background:rgba(120,120,120,.05)}
div[data-testid="stDownloadButton"] button{width:100%}
</style>
""", unsafe_allow_html=True)

SUBSYSTEMS = {
    "🚆 Rail Corrugation": "rail",
    "🚪 Door": "door",
    "❄️ ACV": "acv",
    "📈 SHM": "shm",
}

with st.sidebar:
    st.markdown("### Train Condition Monitoring")
    st.caption("NebulaX hackathon — pick a subsystem")
    choice = st.radio("Subsystem", list(SUBSYSTEMS.keys()), label_visibility="collapsed")
    st.divider()
    st.caption("Each subsystem is an independent model. Rail Corrugation is fully wired up; the others are staged for their datasets.")

subsystem = SUBSYSTEMS[choice]


def validate(df):
    return df.shape[1] >= REQUIRED_COLUMNS and df.shape[0] > 0


def read_uploaded_csv(uploaded):
    try:
        return pd.read_csv(uploaded), None
    except Exception as exc:
        return None, f"Unable to read this file as CSV: {exc}"


def render_rail():
    st.markdown("""<div class="hero"><h1 style="margin:0">🚆 RailGuard</h1>
    <p style="margin:.35rem 0 0">AI-assisted axle-box vibration analysis for rail corrugation monitoring</p></div>""", unsafe_allow_html=True)

    single_tab, batch_tab, about_tab = st.tabs(["🔎 Single diagnosis", "📚 Batch analysis", "ℹ️ Model & method"])

    with single_tab:
        st.markdown('<div class="step">STEP 1 OF 2</div>', unsafe_allow_html=True)
        st.subheader("Load a sensor recording")
        uploaded = st.file_uploader("Rail Corrugation CSV", type=["csv"], key="single", label_visibility="collapsed")

        if not uploaded:
            st.info(f"Drag a Rail Corrugation CSV here to start. Expected format: 10,000 samples x {REQUIRED_COLUMNS} columns (speed + 128 axle-box channels).")
            return

        df, err = read_uploaded_csv(uploaded)
        if err:
            st.error(err)
            return

        rows, cols = df.shape
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("File", uploaded.name)
        q2.metric("Samples", f"{rows:,}")
        q3.metric("Columns", cols)
        q4.metric("Duration", f"{rows / SAMPLE_RATE_HZ:.2f} s")

        if not validate(df):
            st.error(f"File validation failed. Rail recordings require speed + 128 sensor channels ({REQUIRED_COLUMNS} columns); this file has {cols}.")
            return
        if not MODEL_PATH.exists():
            st.warning("Trained model is missing from models/rail_model.joblib. Run training/train_rail.py first.")
            return

        st.success("Recording validated and ready for diagnosis.")
        with st.expander("Inspect raw recording"):
            st.dataframe(df.head(15), use_container_width=True)

        st.markdown('<div class="step">STEP 2 OF 2</div>', unsafe_allow_html=True)
        if st.button("Run rail diagnosis", type="primary", use_container_width=True):
            try:
                with st.spinner("Extracting features and analysing Side I / Side II..."):
                    pred, conf, features, probs = predict_rail_file(df, MODEL_PATH)
            except Exception as exc:
                st.error(f"Diagnosis failed: {exc}")
                return

            cls = "normal" if pred == "Normal" else "fault"
            title = "NORMAL CONDITION" if pred == "Normal" else "CORRUGATION ALERT"
            subtitle = "No corrugation class detected" if pred == "Normal" else f"Model localised the anomaly to {pred}"
            st.markdown(f'<div class="status {cls}"><h2 style="margin:0">{title}</h2><p style="margin:.35rem 0 0">{subtitle}</p></div>', unsafe_allow_html=True)

            a, b, c = st.columns(3)
            a.metric("Classification", pred)
            a.metric("Highest model probability", f"{conf:.1%}")
            recommendation = "Continue routine monitoring" if pred == "Normal" else f"Prioritise inspection of {pred} rail"
            a.info("Action: " + recommendation)

            with b:
                st.markdown("#### Class probabilities")
                p = pd.DataFrame({"Probability": [probs.get("Normal", 0), probs.get("Side I", 0), probs.get("Side II", 0)]}, index=["Normal", "Side I", "Side II"])
                st.bar_chart(p, height=245)
            with c:
                st.markdown("#### Side RMS")
                s1 = float(features.iloc[0]["side1_rms"])
                s2 = float(features.iloc[0]["side2_rms"])
                r = pd.DataFrame({"RMS (m/s²)": [s1, s2]}, index=["Side I", "Side II"])
                st.bar_chart(r, height=245)

            st.divider()
            st.subheader("Signal evidence")
            sig1 = df.iloc[:, side_dataframe_columns(1, vibration_only=True)].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0).to_numpy()
            sig2 = df.iloc[:, side_dataframe_columns(2, vibration_only=True)].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0).to_numpy()
            view = pd.DataFrame({"Side I": sig1[:2000], "Side II": sig2[:2000]})
            st.markdown("**Mean vibration waveform — first 0.2 s**")
            st.line_chart(view, height=250)

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
            st.markdown("**Frequency spectrum — 1 to 2000 Hz**")
            st.line_chart(spec, height=280)
            st.caption("The plots provide engineering context; the classifier uses its full extracted feature set (time + frequency domain, all 128 channels) rather than these plots directly.")

            with st.expander("Detailed engineering indicators"):
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Side I mean RMS", f"{s1:.4f} m/s²")
                e2.metric("Side II mean RMS", f"{s2:.4f} m/s²")
                e3.metric("Side I / II RMS ratio", f"{float(features.iloc[0]['side_rms_ratio']):.3f}")
                e4.metric("Estimated speed", f"{float(features.iloc[0]['speed_signal_mean']):.2f} m/s")

            result = pd.DataFrame([{"file_id": uploaded.name, "prediction": pred}])
            st.download_button("⬇ Download prediction CSV", result.to_csv(index=False).encode(), "rail_predictions.csv", "text/csv")
            st.caption("Classifier probability is not calibrated diagnostic certainty. Competition performance is evaluated using Macro F1.")

    with batch_tab:
        st.subheader("Batch inference")
        st.write("Process multiple recordings in one run and export the required two-column submission file.")
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
            st.download_button("⬇ Download submission-ready rail_predictions.csv", submission.to_csv(index=False).encode(), "rail_predictions.csv", "text/csv", use_container_width=True)
        elif files and not MODEL_PATH.exists():
            st.warning("Trained model is missing from models/rail_model.joblib. Run training/train_rail.py first.")

    with about_tab:
        st.subheader("Model & method")
        st.markdown("""
**Input:** one-second recordings sampled at 10 kHz, containing rotational speed plus 128 axle-box vibration/shock channels (8 cars x 8 axle positions x vibration+shock).

**Physical layout:** axle positions 1, 3, 5 and 7 correspond to Side I; positions 2, 4, 6 and 8 correspond to Side II.

**Pipeline:** signal validation -> per-channel time/frequency feature extraction (128 channels) + per-side aggregate features (RMS, crest factor, spectral bands, speed-normalised wavelength) -> best-of-3 benchmarked classifier (Random Forest / Gradient Boosting / Logistic Regression, selected by cross-validated macro F1) -> Normal / Side I / Side II.

**Evaluation metric:** macro F1 across the three classes, matching the subsystem's official scoring — this credits correctly detecting the rare Side I / Side II cases, not just the common Normal case, unlike plain accuracy.

Model probabilities should not be interpreted as guarantees or calibrated maintenance risk; this is a condition-monitoring decision-support tool.
""")
        st.markdown("#### Pipeline")
        st.code("CSV -> sensor grouping -> per-channel + per-side features -> classifier benchmark -> best model -> side localisation -> maintenance output", language=None)

        meta = load_metadata(MODEL_PATH) if MODEL_PATH.exists() else None
        if meta:
            st.divider()
            st.markdown("#### Training provenance")
            m1, m2, m3 = st.columns(3)
            m1.metric("Training files", meta.get("n_training_files", "—"))
            m1.metric("Selected model", meta.get("selected_model", "—"))
            cv = meta.get("cv_scores_macro_f1", {}).get(meta.get("selected_model", ""), {})
            m2.metric("CV macro F1 (selected model)", f"{cv.get('mean', 0):.4f} ± {cv.get('std', 0):.4f}" if cv else "—")
            dist = meta.get("class_distribution", {})
            m3.metric("Class balance", ", ".join(f"{k}: {v}" for k, v in dist.items()) if dist else "—")

            st.markdown("**Model comparison (5-fold CV, macro F1)**")
            cmp_rows = meta.get("cv_scores_macro_f1", {})
            if cmp_rows:
                cmp_df = pd.DataFrame(
                    {"Macro F1": {k: v["mean"] for k, v in cmp_rows.items()}}
                )
                st.bar_chart(cmp_df, height=220)

            top_feats = meta.get("top_feature_importances", [])
            if top_feats:
                st.markdown("**Top features driving predictions (explainability)**")
                feat_df = pd.DataFrame(top_feats).set_index("feature")
                st.bar_chart(feat_df, height=320)
        else:
            st.info("No trained model metadata yet. Run `python -m training.train_rail --train-dir <Train folder> --labels <Train_Labels.csv>` to train, benchmark, and populate this section with real performance numbers.")


def render_coming_soon(name, blurb, needs):
    st.markdown(f"""<div class="hero"><h1 style="margin:0">{name}</h1>
    <p style="margin:.35rem 0 0">{blurb}</p></div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="soon"><strong>Not wired up yet.</strong> This tab is staged in the app's navigation so it can be dropped in without restructuring the app.
    <br><br>To activate it: {needs}</div>""", unsafe_allow_html=True)


if subsystem == "rail":
    render_rail()
elif subsystem == "door":
    render_coming_soon(
        "🚪 Door — abnormal resistance detection",
        "Temporal segment detection over a continuous motor current/voltage/back-EMF + door-position stream, classifying each open/close cycle as Normal or Abnormal resistance.",
        "add Train.csv, Train_Segments_Answer.csv, and Test.csv under data/Door/, then build src/door_features.py (cycle segmentation + classification) and training/train_door.py.",
    )
elif subsystem == "acv":
    render_coming_soon(
        "❄️ ACV — refrigerant leak localisation",
        "Ranks the 8 cars of a train from most- to least-likely to have a refrigerant leak, using per-car temperature and control-mode telemetry.",
        "add the Train/ case files + Train_Labels.csv under data/ACV/, then build src/acv_features.py (per-car anomaly scoring) and training/train_acv.py.",
    )
elif subsystem == "shm":
    render_coming_soon(
        "📈 SHM — cumulative fatigue damage",
        "Regresses a single cumulative fatigue-damage value from a dynamic stress time series at a structural measurement point.",
        "add the Train/ files + Train_Labels.csv under data/SHM/, then build src/shm_features.py (rainflow/stress-cycle features) and training/train_shm.py.",
    )
