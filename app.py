import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# 1. FORCE DESKTOP FULL-WIDTH LAYOUT
st.set_page_config(page_title="HealthMate AI Dashboard", layout="wide")

# 2. DATA & AI LOGIC (Pre-loaded in background)
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("Patient_Dataset.csv")
    df[['Systolic', 'Diastolic']] = df['Blood Pressure (mmHg)'].str.split('/', expand=True)
    df['Systolic'] = pd.to_numeric(df['Systolic'])
    df['Diastolic'] = pd.to_numeric(df['Diastolic'])
    return df.drop(columns=['Blood Pressure (mmHg)', 'IP Address', 'Device ID'])

@st.cache_resource
def train_automatic_model(df):
    features = ['Heart Rate (bpm)', 'Temperature (°C)', 'Systolic', 'Diastolic']
    X = df[features]; y = df['Target']
    ratio = (y == 0).sum() / (y == 1).sum()
    model = XGBClassifier(scale_pos_weight=ratio, eval_metric='logloss')
    model.fit(X, y)
    return model

data = load_and_clean_data()
health_model = train_automatic_model(data)

# 3. CREATE DESKTOP TABS
# This creates a navigation bar at the top
tab1, tab2 = st.tabs(["🏠 Home Dashboard", "🔬 AI Diagnostic Terminal"])

# --- TAB 1: YOUR DESIGN ---
with tab1:
    try:
        with open("app.html", "r", encoding='utf-8') as f:
            html_design = f.read()
        # High height ensures it looks like a full desktop page
        components.html(html_design, height=1000, scrolling=True)
    except FileNotFoundError:
        st.error("app.html not found.")

# --- TAB 2: THE AI TERMINAL ---
with tab2:
    st.header("🔬 AI Health Diagnostic Terminal")
    st.write("Enter patient vitals to trigger the Smart City Predictive Engine.")
    
    # Using columns for a desktop layout
    col1, col2 = st.columns(2)
    with col1:
        hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)
        temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=36.5)
    with col2:
        sys = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
        dia = st.number_input("Diastolic BP", min_value=40, max_value=130, value=80)

    if st.button("Analyze My Vitals", key="tab_analyze"):
        is_medically_abnormal = (temp >= 38.5 or hr >= 110 or sys >= 150)
        input_df = pd.DataFrame([[hr, temp, sys, dia]], 
                               columns=['Heart Rate (bpm)', 'Temperature (°C)', 'Systolic', 'Diastolic'])
        
        prediction = health_model.predict(input_df)[0]
        
        if prediction == 1 or is_medically_abnormal:
            st.error("⚠️ ALERT: High Health Risk Detected.")
            
            # SHAP Analysis
            explainer = shap.Explainer(health_model)
            shap_values = explainer(input_df)
            fig, ax = plt.subplots()
            shap.plots.bar(shap_values[0], show=False)
            st.pyplot(fig)
        else:
            st.success("✅ NORMAL: Vitals are within healthy range.")
