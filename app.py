import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="HealthMate AI - Smart City Dashboard", layout="wide")

# 2. DESIGN LAYER: LOAD YOUR CUSTOM HTML
# This reads the app.html file you uploaded to GitHub
try:
    with open("app.html", "r", encoding='utf-8') as f:
        html_design = f.read()
    # height=800 ensures enough space for your design; adjust if needed
    components.html(html_design, height=800, scrolling=True)
except FileNotFoundError:
    st.error("⚠️ app.html not found! Please ensure it is uploaded to the same folder as app.py on GitHub.")
except Exception as e:
    st.error(f"⚠️ Error loading design: {e}")

st.markdown("---") # Visual separator between your design and the AI logic

# 3. DATA LOGIC
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("Patient_Dataset.csv")
    # Clean and split Blood Pressure
    df[['Systolic', 'Diastolic']] = df['Blood Pressure (mmHg)'].str.split('/', expand=True)
    df['Systolic'] = pd.to_numeric(df['Systolic'])
    df['Diastolic'] = pd.to_numeric(df['Diastolic'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # Remove noise
    return df.drop(columns=['Blood Pressure (mmHg)', 'IP Address', 'Device ID'])

# 4. AUTOMATIC AI TRAINING (Background)
@st.cache_resource
def train_automatic_model(df):
    features = ['Heart Rate (bpm)', 'Temperature (°C)', 'Systolic', 'Diastolic']
    X = df[features]
    y = df['Target']
    
    # Fix imbalance so abnormal values are detected
    ratio = (y == 0).sum() / (y == 1).sum()
    
    # Train the model
    model = XGBClassifier(scale_pos_weight=ratio, eval_metric='logloss')
    model.fit(X, y)
    
    # Test accuracy for display
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    acc = accuracy_score(y_test, model.predict(X_test))
    
    return model, acc

# Initialize data and AI
data = load_and_clean_data()
health_model, model_accuracy = train_automatic_model(data)

# 5. SIDEBAR STATUS
st.sidebar.success("✅ Smart City AI Online")
st.sidebar.metric("AI Prediction Accuracy", f"{model_accuracy:.2%}")

# 6. DIAGNOSTIC TERMINAL (THE "BRAIN")
st.header("🔬 AI Health Diagnostic Terminal")
st.info("This terminal connects your live vitals to the urban health predictive engine.")

col1, col2 = st.columns(2)
with col1:
    hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)
    temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=36.5)
with col2:
    sys = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
    dia = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=130, value=80)

# 7. ANALYSIS LOGIC (Hybrid: AI + Medical Rules)
if st.button("Analyze My Vitals", key="analyze_vitals_main"):
    # Medical Safety Thresholds (The override for abnormal values)
    is_medically_abnormal = (temp >= 38.5 or hr >= 110 or sys >= 150)
    
    # AI Prediction
    features_list = ['Heart Rate (bpm)', 'Temperature (°C)', 'Systolic', 'Diastolic']
    input_df = pd.DataFrame([[hr, temp, sys, dia]], columns=features_list)
    ai_prediction = health_model.predict(input_df)[0]
    
    if ai_prediction == 1 or is_medically_abnormal:
        st.error("⚠️ ALERT: High Health Risk Detected.")
        
        # Explainable AI Report
        st.write("### AI Decision Report (Explainability)")
        explainer = shap.Explainer(health_model)
        shap_values = explainer(input_df)
        
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)
        st.caption("Red bars show which vitals contributed most to the 'High Risk' status.")
    else:
        st.success("✅ NORMAL: Vitals are within healthy range.")

# 8. DATA TRANSPARENCY
with st.expander("View Training Dataset Logs"):
    st.dataframe(data.head())
