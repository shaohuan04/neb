from pathlib import Path
import pandas as pd
import streamlit as st
from src.rail_inference import predict_rail_file

st.set_page_config(page_title='Train Condition Monitoring',page_icon='🚆',layout='wide')
st.title('Train Condition Monitoring')
st.caption('Rail Corrugation diagnostic prototype')
st.sidebar.selectbox('Subsystem',['Rail Corrugation'])
uploaded=st.file_uploader('Upload a Rail Corrugation CSV',type=['csv'])
model_path=Path('models/rail_model.joblib')
if uploaded:
    df=pd.read_csv(uploaded)
    st.write('Input shape:',df.shape)
    st.dataframe(df.head(),use_container_width=True)
    if not model_path.exists(): st.warning('Model not trained yet. Run training/train_rail.py first.')
    elif st.button('Analyse',type='primary'):
        pred,conf,x=predict_rail_file(df,model_path)
        st.metric('Prediction',pred); st.metric('Confidence','%.1f%%'%(100*conf))
        st.write('Side I mean RMS:',float(x.iloc[0]['side1_mean_rms']))
        st.write('Side II mean RMS:',float(x.iloc[0]['side2_mean_rms']))
        result=pd.DataFrame([{'file_id':uploaded.name,'prediction':pred}])
        st.download_button('Download rail_predictions.csv',result.to_csv(index=False),file_name='rail_predictions.csv',mime='text/csv')
