import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="HealthMate AI - Smart City Dashboard", layout="wide")

st.title("🏥 AI for Predictive Healthcare in Smart Cities")

# 2. Data Loading
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("Patient_Dataset.csv")
    df[['Systolic', 'Diastolic']] = df['Blood Pressure (mmHg)'].str.split('/', expand=True)
    df['Systolic'] = pd.to_numeric(df['Systolic'])
    df['Diastolic'] = pd.to_numeric(df['Diastolic'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df.drop(columns=['Blood Pressure (mmHg)', 'IP Address', 'Device ID'])

# 3. AI TRAINING (With Class Balancing)
@st.cache_resource
def train_automatic_model(df):
    features = ['Heart Rate (bpm)', 'Temperature (°C)', 'Systolic', 'Diastolic']
    X = df[features]
    y = df['Target']
    
    # Calculate ratio to handle imbalance
    ratio = (y == 0).sum() / (y == 1).sum()
    
    model = XGBClassifier(scale_pos_weight=ratio, eval_metric='logloss')
    model.fit(X, y)
    return model

data = load_and_clean_data()
health_model = train_automatic_model(data)

# 4. PREDICTION INTERFACE
st.header("🔍 Personal Health Risk Predictor")

col1, col2 = st.columns(2)
with col1:
    hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)
    temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=36.5)
with col2:
    sys = st.number_input("Systolic BP (top number)", min_value=80, max_value=200, value=120)
    dia = st.number_input("Diastolic BP (bottom number)", min_value=40, max_value=130, value=80)

if st.button("Analyze My Vitals", key="analyze_vitals_main"):
    # --- MEDICAL SAFETY LAYER (Rule-Based) ---
    # We define medical "Red Flags" manually to assist the AI
    is_medically_abnormal = (temp >= 38.5 or hr >= 110 or sys >= 150)
    
    # --- AI PREDICTION ---
    features_list = ['Heart Rate (bpm)', 'Temperature (°C)', 'Systolic', 'Diastolic']
    input_df = pd.DataFrame([[hr, temp, sys, dia]], columns=features_list)
    ai_prediction = health_model.predict(input_df)[0]
    
    # If EITHER the AI or the Medical Rules say it's bad, we trigger the Alert
    if ai_prediction == 1 or is_medically_abnormal:
        st.error("⚠️ ALERT: High Risk Detected.")
        
        # Explain the AI's part
        st.write("### AI Reasoning Analysis")
        explainer = shap.Explainer(health_model)
        shap_values = explainer(input_df)
        
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)
        
        if is_medically_abnormal:
            st.warning("Note: This alert was also triggered by clinical safety thresholds (Red Flags).")
    else:
        st.success("✅ NORMAL: Vitals are within the healthy range.")
