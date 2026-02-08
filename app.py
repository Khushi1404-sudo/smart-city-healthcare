import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="HealthMate AI - Smart City Dashboard", layout="wide")

st.title("🏥 AI for Predictive Healthcare in Smart Cities")

# 2. Logic to Load and Clean Data
@st.cache_data
def load_and_clean_data():
    # Load the dataset
    df = pd.read_csv("Patient_Dataset.csv")
    
    # SPLITTING LOGIC: Handling Blood Pressure (e.g., "116/84")
    df[['Systolic', 'Diastolic']] = df['Blood Pressure (mmHg)'].str.split('/', expand=True)
    
    # Convert to numbers
    df['Systolic'] = pd.to_numeric(df['Systolic'])
    df['Diastolic'] = pd.to_numeric(df['Diastolic'])
    
    # TIME LOGIC: Convert Timestamps
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # CLEANING LOGIC: Remove non-medical "Noise"
    columns_to_drop = ['Blood Pressure (mmHg)', 'IP Address', 'Device ID']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    return df_cleaned

# 3. AUTOMATIC AI TRAINING LOGIC
# This runs in the background so the user doesn't have to click "Train"
@st.cache_resource
def train_automatic_model(df):
    features = ['Heart Rate (bpm)', 'Temperature (°C)', 'Systolic', 'Diastolic']
    X = df[features]
    y = df['Target']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and Train XGBoost
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    # Calculate accuracy for the sidebar
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    return model, acc

# --- Execute Data & AI Pipeline ---
data = load_and_clean_data()
health_model, model_accuracy = train_automatic_model(data)

# 4. Sidebar Information
st.sidebar.success("✅ System Online")
st.sidebar.metric("AI Accuracy", f"{model_accuracy:.2%}")
st.sidebar.write("The AI has been pre-trained on the urban patient dataset and is ready for real-time diagnosis.")

# 5. Display Data Preview (Optional, for your report)
with st.expander("View Processed Smart City Data"):
    st.dataframe(data.head())

# 6. PREDICTION & SHAP EXPLAINABILITY INTERFACE
st.header("🔍 Personal Health Risk Predictor")
st.write("Enter patient vitals below. The AI will analyze these against smart city health patterns.")

col1, col2 = st.columns(2)
with col1:
    hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)
    temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=36.5)
with col2:
    sys = st.number_input("Systolic BP (top number)", min_value=80, max_value=200, value=120)
    dia = st.number_input("Diastolic BP (bottom number)", min_value=40, max_value=130, value=80)

# Single Action Button
if st.button("Analyze My Vitals", key="analyze_vitals_main"):
    # Prepare input
    features = ['Heart Rate (bpm)', 'Temperature (°C)', 'Systolic', 'Diastolic']
    input_df = pd.DataFrame([[hr, temp, sys, dia]], columns=features)
    
    # Make prediction
    prediction = health_model.predict(input_df)
    
    if prediction[0] == 1:
        st.error("⚠️ ALERT: High Risk Detected. Patterns suggest potential health complications.")
        
        # --- SHAP Explainability ---
        st.write("### AI Reasoning (Why this prediction?)")
        explainer = shap.Explainer(health_model)
        shap_values = explainer(input_df)
        
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)
        st.caption("Red bars (positive) indicate factors that increased the risk. Blue bars (negative) indicate healthy factors.")
    else:
        st.success("✅ NORMAL: Your vitals appear to be within the healthy range based on current urban trends.")
