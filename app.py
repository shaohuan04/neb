from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from src.rail_inference import predict_rail_file

st.set_page_config(page_title="RailGuard", page_icon="🚆", layout="wide", initial_sidebar_state="collapsed")
MODEL_PATH = Path("models/rail_model.joblib")
FS = 10000

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
div[data-testid="stDownloadButton"] button{width:100%}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1 style="margin:0">🚆 RailGuard</h1>
<p style="margin:.35rem 0 0">AI-assisted axle-box vibration analysis for rail corrugation monitoring</p></div>""", unsafe_allow_html=True)

def validate(df):
    return df.shape[1] >= 129 and df.shape[0] > 0

def side_signal(df, side):
    # Vibration channels only. Each axle has vibration then shock.
    indices=[]
    for ch in range(128):
        axle=((ch)//2)%8+1
        is_vibration=(ch%2==0)
        if is_vibration and ((side==1 and axle%2==1) or (side==2 and axle%2==0)):
            indices.append(ch+1)  # +1 skips speed column
    return df.iloc[:,indices].apply(pd.to_numeric,errors="coerce").mean(axis=1).fillna(0).to_numpy()

def spectrum(x):
    x=np.asarray(x,dtype=float)
    x=x-np.mean(x)
    win=np.hanning(len(x))
    mag=np.abs(np.fft.rfft(x*win))
    freq=np.fft.rfftfreq(len(x),1/FS)
    return pd.DataFrame({"Frequency (Hz)":freq,"Magnitude":mag}).set_index("Frequency (Hz)")

single_tab,batch_tab,about_tab=st.tabs(["🔎 Single diagnosis","📚 Batch analysis","ℹ️ Model & method"])

with single_tab:
    st.markdown('<div class="step">STEP 1 OF 2</div>',unsafe_allow_html=True)
    st.subheader("Load a sensor recording")
    uploaded=st.file_uploader("Rail Corrugation CSV",type=["csv"],key="single",label_visibility="collapsed")

    if not uploaded:
        st.info("Drag a Rail Corrugation CSV here to start. Expected format: 10,000 samples × 129 columns.")
    else:
        try:
            df=pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Unable to read file: {exc}"); st.stop()

        rows,cols=df.shape
        q1,q2,q3,q4=st.columns(4)
        q1.metric("File",uploaded.name)
        q2.metric("Samples",f"{rows:,}")
        q3.metric("Columns",cols)
        q4.metric("Duration",f"{rows/FS:.2f} s")

        if not validate(df):
            st.error("File validation failed. Rail recordings require speed + 128 sensor channels.")
        elif not MODEL_PATH.exists():
            st.warning("Trained model is missing from models/rail_model.joblib.")
        else:
            st.success("Recording validated and ready for diagnosis.")
            with st.expander("Inspect raw recording"):
                st.dataframe(df.head(15),use_container_width=True)

            st.markdown('<div class="step">STEP 2 OF 2</div>',unsafe_allow_html=True)
            if st.button("Run rail diagnosis",type="primary",use_container_width=True):
                with st.spinner("Extracting features and analysing Side I / Side II..."):
                    pred,conf,features,probs=predict_rail_file(df,MODEL_PATH)

                cls="normal" if pred=="Normal" else "fault"
                title="NORMAL CONDITION" if pred=="Normal" else "CORRUGATION ALERT"
                subtitle="No corrugation class detected" if pred=="Normal" else f"Model localised the anomaly to {pred}"
                st.markdown(f'<div class="status {cls}"><h2 style="margin:0">{title}</h2><p style="margin:.35rem 0 0">{subtitle}</p></div>',unsafe_allow_html=True)

                a,b,c=st.columns(3)
                a.metric("Classification",pred)
                a.metric("Highest model probability",f"{conf:.1%}")
                recommendation="Continue routine monitoring" if pred=="Normal" else f"Prioritise inspection of {pred} rail"
                a.info("Action: "+recommendation)

                with b:
                    st.markdown("#### Class probabilities")
                    p=pd.DataFrame({"Probability":[probs.get("Normal",0),probs.get("Side I",0),probs.get("Side II",0)]},index=["Normal","Side I","Side II"])
                    st.bar_chart(p,height=245)
                with c:
                    st.markdown("#### Side RMS")
                    s1=float(features.iloc[0]["side1_mean_rms"]); s2=float(features.iloc[0]["side2_mean_rms"])
                    r=pd.DataFrame({"RMS (m/s²)":[s1,s2]},index=["Side I","Side II"])
                    st.bar_chart(r,height=245)

                st.divider()
                st.subheader("Signal evidence")
                sig1=side_signal(df,1); sig2=side_signal(df,2)
                view=pd.DataFrame({"Side I":sig1[:2000],"Side II":sig2[:2000]})
                st.markdown("**Mean vibration waveform — first 0.2 s**")
                st.line_chart(view,height=250)

                sp1=spectrum(sig1); sp2=spectrum(sig2)
                spec=pd.DataFrame({"Side I":sp1["Magnitude"],"Side II":sp2["Magnitude"]})
                spec=spec.loc[(spec.index>=1)&(spec.index<=2000)]
                st.markdown("**Frequency spectrum — 1 to 2000 Hz**")
                st.line_chart(spec,height=280)
                st.caption("The plots provide engineering context; the current classifier uses its extracted feature set rather than these plots directly.")

                with st.expander("Detailed engineering indicators"):
                    e1,e2,e3=st.columns(3)
                    e1.metric("Side I mean RMS",f"{s1:.4f} m/s²")
                    e2.metric("Side II mean RMS",f"{s2:.4f} m/s²")
                    e3.metric("Side I / II RMS ratio",f"{float(features.iloc[0]['side_rms_ratio']):.3f}")

                result=pd.DataFrame([{"file_id":uploaded.name,"prediction":pred}])
                st.download_button("⬇ Download prediction CSV",result.to_csv(index=False).encode(),"rail_predictions.csv","text/csv")
                st.caption("Classifier probability is not calibrated diagnostic certainty. Competition performance is evaluated using Macro F1.")

with batch_tab:
    st.subheader("Batch inference")
    st.write("Process multiple recordings in one run and export the required two-column submission file.")
    files=st.file_uploader("Upload Rail CSV files",type=["csv"],accept_multiple_files=True,key="batch")
    if files:
        st.caption(f"{len(files)} file(s) selected")
    if files and MODEL_PATH.exists() and st.button("Analyse all recordings",type="primary",use_container_width=True):
        results=[]; progress=st.progress(0,text="Starting batch analysis...")
        for idx,f in enumerate(files):
            try:
                d=pd.read_csv(f)
                if not validate(d): raise ValueError("invalid Rail file structure")
                pred,conf,_,probs=predict_rail_file(d,MODEL_PATH)
                results.append({"file_id":f.name,"prediction":pred,"probability":conf,"status":"OK"})
            except Exception as exc:
                results.append({"file_id":f.name,"prediction":"ERROR","probability":0.0,"status":str(exc)})
            progress.progress((idx+1)/len(files),text=f"Analysed {idx+1} of {len(files)}")
        out=pd.DataFrame(results)
        st.success(f"Completed {len(files)} recordings")
        n1,n2,n3=st.columns(3)
        n1.metric("Normal",int((out.prediction=="Normal").sum()))
        n2.metric("Side I",int((out.prediction=="Side I").sum()))
        n3.metric("Side II",int((out.prediction=="Side II").sum()))
        st.dataframe(out,use_container_width=True,hide_index=True)
        submission=out[out.status=="OK"][["file_id","prediction"]]
        st.download_button("⬇ Download submission-ready rail_predictions.csv",submission.to_csv(index=False).encode(),"rail_predictions.csv","text/csv",use_container_width=True)

with about_tab:
    st.subheader("Model & method")
    st.markdown("""
**Input:** one-second recordings sampled at 10 kHz, containing rotational speed plus 128 axle-box vibration/shock channels.

**Physical layout:** axle positions 1, 3, 5 and 7 correspond to Side I; positions 2, 4, 6 and 8 correspond to Side II.

**Current pipeline:** signal validation → time/frequency feature extraction → balanced Random Forest → Normal / Side I / Side II.

**Current validation baseline:** 5-fold cross-validation Macro F1 = **0.6026 ± 0.0906**.

The dashboard is a condition-monitoring prototype. Model probabilities should not be interpreted as guarantees or calibrated maintenance risk.
""")
    st.markdown("#### Pipeline")
    st.code("CSV → sensor grouping → signal features → classifier → side localisation → maintenance output",language=None)
