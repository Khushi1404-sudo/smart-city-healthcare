import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Diagnostic Terminal", layout="wide")


@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("Patient_Dataset.csv")
    df[['Systolic', 'Diastolic']] = df['Blood Pressure (mmHg)'].str.split('/', expand=True)
    df['Systolic'] = pd.to_numeric(df['Systolic'])
    df['Diastolic'] = pd.to_numeric(df['Diastolic'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df.drop(columns=['Blood Pressure (mmHg)', 'IP Address', 'Device ID'])

@st.cache_resource
def train_automatic_model(df):
    features = ['Heart Rate (bpm)', 'Temperature (°C)', 'Systolic', 'Diastolic']
    X = df[features]
    y = df['Target']
    ratio = (y == 0).sum() / (y == 1).sum()
    model = XGBClassifier(scale_pos_weight=ratio, eval_metric='logloss')
    model.fit(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

data = load_and_clean_data()
health_model, model_accuracy = train_automatic_model(data)

st.sidebar.metric("AI Accuracy", f"{model_accuracy:.2%}")

st.header("🔬 AI Health Diagnostic Terminal")

col1, col2 = st.columns(2)
with col1:
    hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)
    temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=36.5)
with col2:
    sys = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
    dia = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=130, value=80)

if st.button("Analyze My Vitals"):
    is_medically_abnormal = (temp >= 38.5 or hr >= 110 or sys >= 150)
    features_list = ['Heart Rate (bpm)', 'Temperature (°C)', 'Systolic', 'Diastolic']
    input_df = pd.DataFrame([[hr, temp, sys, dia]], columns=features_list)
    ai_prediction = health_model.predict(input_df)[0]
    
    if ai_prediction == 1 or is_medically_abnormal:
        st.error("⚠️ ALERT: High Health Risk Detected.")
        explainer = shap.Explainer(health_model)
        shap_values = explainer(input_df)
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)
    else:
        st.success("✅ NORMAL: Vitals are within healthy range.")
