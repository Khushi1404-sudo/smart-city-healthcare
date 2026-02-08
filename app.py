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

# 2. Logic to Load the CSV
@st.cache_data
def load_and_clean_data():
    # Load the dataset you uploaded
    df = pd.read_csv("Patient_Dataset.csv")
    
    # 3. SPLITTING LOGIC: Handling Blood Pressure (e.g., "116/84")
    df[['Systolic', 'Diastolic']] = df['Blood Pressure (mmHg)'].str.split('/', expand=True)
    
    # Convert to numbers
    df['Systolic'] = pd.to_numeric(df['Systolic'])
    df['Diastolic'] = pd.to_numeric(df['Diastolic'])
    
    # 4. TIME LOGIC: Convert Timestamps
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # 5. CLEANING LOGIC: Remove non-medical "Noise"
    columns_to_drop = ['Blood Pressure (mmHg)', 'IP Address', 'Device ID']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    return df_cleaned

# Run the data loading
data = load_and_clean_data()

# 6. Displaying the results
st.subheader("Processed Patient Data")
st.write("Below is the cleaned data where Blood Pressure is now split for AI analysis:")
st.dataframe(data.head())

# Sidebar metrics
st.sidebar.success("Data Loaded Successfully!")
st.sidebar.metric("Total Patients", len(data))

# --- STEP 2: AI MODEL LOGIC ---
st.header("🤖 AI Health Risk Analysis")

features = ['Heart Rate (bpm)', 'Temperature (°C)', 'Systolic', 'Diastolic']
X = data[features]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Button with a unique key
if st.button("Train AI Model", key="train_model_btn"):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    st.success(f"Model Trained! Accuracy: {acc:.2%}")
    st.session_state['health_model'] = model

# --- STEP 3 & 4: PREDICTION & SHAP EXPLAINABILITY ---
st.header("🔍 Personal Health Risk Predictor")
st.write("Enter patient vitals below to check for potential health risks.")

col1, col2 = st.columns(2)
with col1:
    hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)
    temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=36.5)
with col2:
    sys = st.number_input("Systolic BP (top number)", min_value=80, max_value=200, value=120)
    dia = st.number_input("Diastolic BP (bottom number)", min_value=40, max_value=130, value=80)

# MERGED BUTTON LOGIC (Fixed the Duplicate ID Error here)
if st.button("Analyze My Vitals", key="analyze_vitals_main"):
    if 'health_model' in st.session_state:
        model = st.session_state['health_model']
        
        # Prepare input as a DataFrame to keep feature names for SHAP
        input_df = pd.DataFrame([[hr, temp, sys, dia]], columns=features)
        
        # Make prediction
        prediction = model.predict(input_df)
        
        if prediction[0] == 1:
            st.error("⚠️ ALERT: High Risk Detected. Please consult a healthcare professional.")
            
            # --- SHAP Explainability ---
            st.write("### AI Reasoning (Why this prediction?)")
            explainer = shap.Explainer(model)
            shap_values = explainer(input_df)
            
            fig, ax = plt.subplots()
            shap.plots.bar(shap_values[0], show=False)
            st.pyplot(fig)
            st.caption("Red bars increase risk; blue bars decrease it.")
        else:
            st.success("✅ NORMAL: Your vitals appear to be within the healthy range.")
    else:
        st.warning("Please click 'Train AI Model' above first to prepare the AI.")
