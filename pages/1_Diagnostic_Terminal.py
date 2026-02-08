import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Diagnostic Terminal", layout="wide")

st.header("🔬 AI Health Diagnostic Terminal")
st.write("---")

# Logic to load data and train (copied from our working script)
@st.cache_data
def load_data():
    df = pd.read_csv("Patient_Dataset.csv")
    df[['Systolic', 'Diastolic']] = df['Blood Pressure (mmHg)'].str.split('/', expand=True)
    df['Systolic'] = pd.to_numeric(df['Systolic']); df['Diastolic'] = pd.to_numeric(df['Diastolic'])
    return df.drop(columns=['Blood Pressure (mmHg)', 'IP Address', 'Device ID'])

@st.cache_resource
def train_model(df):
    X = df[['Heart Rate (bpm)', 'Temperature (°C)', 'Systolic', 'Diastolic']]
    y = df['Target']
    ratio = (y == 0).sum() / (y == 1).sum()
    model = XGBClassifier(scale_pos_weight=ratio, eval_metric='logloss')
    model.fit(X, y)
    return model

data = load_data()
model = train_model(data)

# Inputs
col1, col2 = st.columns(2)
with col1:
    hr = st.number_input("Heart Rate", value=75)
    temp = st.number_input("Temperature (°C)", value=36.5)
with col2:
    sys = st.number_input("Systolic", value=120)
    dia = st.number_input("Diastolic", value=80)

if st.button("Analyze Vitals"):
    is_abnormal = (temp >= 38.5 or hr >= 110 or sys >= 150)
    input_df = pd.DataFrame([[hr, temp, sys, dia]], columns=['Heart Rate (bpm)', 'Temperature (°C)', 'Systolic', 'Diastolic'])
    
    if model.predict(input_df)[0] == 1 or is_abnormal:
        st.error("⚠️ High Risk Detected")
        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)
    else:
        st.success("✅ Vitals are Normal")
