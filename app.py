from pathlib import Path
import pandas as pd
import streamlit as st
from src.rail_inference import predict_rail_file

st.set_page_config(
    page_title="RailGuard | Condition Monitoring",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px;}
[data-testid="stMetric"] {
    background: rgba(120,120,120,0.08);
    border: 1px solid rgba(120,120,120,0.20);
    padding: 16px;
    border-radius: 12px;
}
.hero {
    padding: 1.4rem 1.6rem;
    border: 1px solid rgba(120,120,120,0.20);
    border-radius: 16px;
    margin-bottom: 1.5rem;
}
.small-note {opacity: 0.72; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1 style="margin:0">🚆 RailGuard</h1>
<p style="font-size:1.1rem;margin:.35rem 0 0 0">
AI-assisted train condition monitoring for rail corrugation
</p>
</div>
""", unsafe_allow_html=True)

MODEL_PATH = Path("models/rail_model.joblib")

with st.sidebar:
    st.header("Monitoring Console")
    subsystem = st.selectbox("Subsystem", ["Rail Corrugation"])
    st.markdown("**Model output**")
    st.caption("Normal · Side I · Side II")
    st.divider()
    st.markdown("**Expected input**")
    st.caption("1-second Rail Corrugation CSV recording with rotational-speed and axle-box sensor channels.")
    st.divider()
    st.caption("Prototype decision-support tool. Predictions should be interpreted with engineering context.")

tab_analyse, tab_about = st.tabs(["Analyse recording", "How it works"])

with tab_analyse:
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        st.subheader("1. Upload sensor recording")
        uploaded = st.file_uploader(
            "Drag and drop a Rail Corrugation CSV",
            type=["csv"],
            help="Use a CSV from the Rail_Corrugation dataset.",
        )

        if uploaded is None:
            st.info("Upload a recording to begin analysis.")
        else:
            try:
                df = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Could not read this CSV: {exc}")
                st.stop()

            rows, cols = df.shape
            c1, c2, c3 = st.columns(3)
            c1.metric("Samples", f"{rows:,}")
            c2.metric("Columns", cols)
            c3.metric("Duration", f"{rows / 10000:.2f} s")

            if cols < 129:
                st.error(
                    "This does not look like a Rail Corrugation recording. "
                    "Expected at least 129 columns (speed + 128 sensor channels)."
                )
            else:
                st.success("File structure passed the basic Rail Corrugation check.")

            with st.expander("Preview raw data"):
                st.dataframe(df.head(20), use_container_width=True)

            st.subheader("2. Run diagnosis")
            analyse = st.button(
                "Analyse recording",
                type="primary",
                use_container_width=True,
                disabled=(cols < 129 or not MODEL_PATH.exists()),
            )

            if not MODEL_PATH.exists():
                st.warning(
                    "Trained model not found at models/rail_model.joblib. "
                    "Train the model before running diagnosis."
                )

    with right:
        st.subheader("Diagnostic result")
        if uploaded is None:
            st.markdown("### Waiting for recording")
            st.caption("The diagnosis and signal indicators will appear here.")
        elif cols >= 129 and MODEL_PATH.exists() and analyse:
            with st.spinner("Extracting vibration features and running model..."):
                pred, conf, features = predict_rail_file(df, MODEL_PATH)

            if pred == "Normal":
                st.success(f"### ✓ {pred}")
                st.caption("No corrugation class was identified by the model.")
            else:
                st.error(f"### ⚠ {pred} corrugation")
                st.caption(f"The model identified a corrugation pattern on {pred}.")

            m1, m2 = st.columns(2)
            m1.metric("Model confidence", f"{100 * conf:.1f}%")
            m2.metric("Classification", pred)

            side1 = float(features.iloc[0]["side1_mean_rms"])
            side2 = float(features.iloc[0]["side2_mean_rms"])

            st.markdown("#### Side comparison")
            side_df = pd.DataFrame(
                {"Mean RMS acceleration": [side1, side2]},
                index=["Side I", "Side II"],
            )
            st.bar_chart(side_df)

            st.markdown("#### Engineering indicators")
            i1, i2, i3 = st.columns(3)
            i1.metric("Side I mean RMS", f"{side1:.3f}")
            i2.metric("Side II mean RMS", f"{side2:.3f}")
            i3.metric(
                "I / II RMS ratio",
                f"{float(features.iloc[0]['side_rms_ratio']):.3f}",
            )

            result = pd.DataFrame(
                [{"file_id": uploaded.name, "prediction": pred}]
            )
            st.download_button(
                "Download rail_predictions.csv",
                result.to_csv(index=False).encode("utf-8"),
                file_name="rail_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        elif uploaded is not None:
            st.caption("Run the diagnosis after the file passes validation.")

with tab_about:
    st.subheader("What the model does")
    st.write(
        "Each recording contains axle-box vibration and shock measurements. "
        "The pipeline extracts time- and frequency-domain signal features, "
        "compares measurements associated with Side I and Side II, and uses "
        "a trained classifier to predict Normal, Side I, or Side II."
    )
    st.markdown("#### Processing pipeline")
    st.code(
        "CSV recording → validate → extract signal features → classifier "
        "→ diagnosis → downloadable prediction",
        language=None,
    )
    st.markdown("#### Important limitation")
    st.write(
        "The confidence value is the classifier's estimated probability, not "
        "a guarantee that the diagnosis is correct. The competition evaluates "
        "the Rail Corrugation model using Macro F1 across all three classes."
    )
