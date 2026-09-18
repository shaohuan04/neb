from pathlib import Path
import pandas as pd
import streamlit as st
from src.rail_inference import predict_rail_file

st.set_page_config(page_title="RailGuard", page_icon="🚆", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container {max-width: 1320px; padding-top: 1.4rem;}
.hero {padding:1.3rem 1.5rem;border:1px solid rgba(120,120,120,.2);border-radius:16px;margin-bottom:1rem}
[data-testid="stMetric"] {background:rgba(120,120,120,.07);border:1px solid rgba(120,120,120,.18);padding:14px;border-radius:12px}
.status {padding:20px;border-radius:14px;margin:.4rem 0 1rem 0}
.normal {background:rgba(30,160,90,.16);border:1px solid rgba(30,160,90,.35)}
.fault {background:rgba(230,70,70,.14);border:1px solid rgba(230,70,70,.35)}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1 style="margin:0">🚆 RailGuard</h1>
<p style="margin:.35rem 0 0">Rail corrugation condition monitoring dashboard</p></div>""", unsafe_allow_html=True)

MODEL_PATH = Path("models/rail_model.joblib")
single_tab, batch_tab, about_tab = st.tabs(["Single diagnosis", "Batch analysis", "How it works"])

with single_tab:
    uploaded = st.file_uploader("Upload one Rail Corrugation CSV", type=["csv"], key="single")
    if uploaded:
        df = pd.read_csv(uploaded)
        rows, cols = df.shape
        a,b,c = st.columns(3)
        a.metric("Samples", f"{rows:,}"); b.metric("Channels + speed", cols); c.metric("Duration", f"{rows/10000:.2f} s")

        if cols < 129:
            st.error("Invalid Rail file: expected at least 129 columns.")
        elif not MODEL_PATH.exists():
            st.warning("Model file is missing. Train the model first.")
        elif st.button("Run diagnosis", type="primary", use_container_width=True):
            with st.spinner("Analysing axle-box signals..."):
                pred, conf, features, probs = predict_rail_file(df, MODEL_PATH)

            cls = "normal" if pred == "Normal" else "fault"
            title = "NORMAL" if pred == "Normal" else "CORRUGATION DETECTED"
            detail = "No fault side detected" if pred == "Normal" else f"Detected side: {pred}"
            st.markdown(f'<div class="status {cls}"><h2 style="margin:0">{title}</h2><p style="margin:.3rem 0 0">{detail}</p></div>', unsafe_allow_html=True)

            left,right = st.columns([1,1], gap="large")
            with left:
                st.subheader("Class probabilities")
                prob_df = pd.DataFrame({
                    "Probability": [probs.get("Normal",0), probs.get("Side I",0), probs.get("Side II",0)]
                }, index=["Normal","Side I","Side II"])
                st.bar_chart(prob_df)
                st.caption("Model probabilities are not guarantees of correctness.")
            with right:
                st.subheader("Engineering indicators")
                s1=float(features.iloc[0]["side1_mean_rms"]); s2=float(features.iloc[0]["side2_mean_rms"])
                x,y = st.columns(2)
                x.metric("Side I mean RMS", f"{s1:.3f} m/s²")
                y.metric("Side II mean RMS", f"{s2:.3f} m/s²")
                st.metric("Side I / Side II RMS ratio", f"{float(features.iloc[0]['side_rms_ratio']):.3f}")
                recommendation = "No corrugation alert; continue routine monitoring." if pred=="Normal" else f"Flag {pred} rail for engineering inspection."
                st.info("Recommended action: " + recommendation)

            result=pd.DataFrame([{"file_id":uploaded.name,"prediction":pred}])
            st.download_button("Download prediction CSV", result.to_csv(index=False).encode(), "rail_predictions.csv", "text/csv", use_container_width=True)

with batch_tab:
    st.subheader("Analyse multiple recordings")
    st.write("Upload multiple Rail CSV files and generate one submission-ready prediction file.")
    files = st.file_uploader("Upload Rail CSVs", type=["csv"], accept_multiple_files=True, key="batch")
    if files and MODEL_PATH.exists() and st.button("Analyse all files", type="primary"):
        results=[]; progress=st.progress(0)
        for idx,f in enumerate(files):
            try:
                df=pd.read_csv(f)
                if df.shape[1] < 129: raise ValueError("expected at least 129 columns")
                pred,conf,_,_=predict_rail_file(df,MODEL_PATH)
                results.append({"file_id":f.name,"prediction":pred,"model_probability":conf})
            except Exception as exc:
                results.append({"file_id":f.name,"prediction":"ERROR","model_probability":0.0})
            progress.progress((idx+1)/len(files))
        batch_df=pd.DataFrame(results)
        st.dataframe(batch_df,use_container_width=True)
        submission=batch_df[batch_df["prediction"]!="ERROR"][["file_id","prediction"]]
        st.download_button("Download rail_predictions.csv", submission.to_csv(index=False).encode(), "rail_predictions.csv", "text/csv", use_container_width=True)

with about_tab:
    st.subheader("Processing pipeline")
    st.code("CSV → validation → time/frequency features → classifier → Normal / Side I / Side II", language=None)
    st.write("The current model is a baseline Random Forest evaluated using Macro F1. The probability shown by the classifier should not be interpreted as validated diagnostic certainty.")
