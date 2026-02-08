import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt
# 1. Page Configuration (This matches your design intent)
st.set_page_config(page_title="HealthMate AI - Smart City Dashboard", layout="wide")

st.title("🏥 AI for Predictive Healthcare in Smart Cities")

# 2. Logic to Load the CSV
# We use st.cache_data so the app doesn't reload the file every time you click a button
@st.cache_data
def load_and_clean_data():
    # Load the dataset you uploaded
    df = pd.read_csv("Patient_Dataset.csv")
    
    # 3. SPLITTING LOGIC: Handling Blood Pressure (e.g., "116/84")
    # We split the string by '/' and create two new numeric columns
    df[['Systolic', 'Diastolic']] = df['Blood Pressure (mmHg)'].str.split('/', expand=True)
    
    # Convert them to numbers (floats) so the AI can use them
    df['Systolic'] = pd.to_numeric(df['Systolic'])
    df['Diastolic'] = pd.to_numeric(df['Diastolic'])
    
    # 4. TIME LOGIC: Convert Timestamps
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # 5. CLEANING LOGIC: Remove non-medical "Noise"
    # Dropping IP Address and Device ID as they don't help predict health [cite: 17]
    columns_to_drop = ['Blood Pressure (mmHg)', 'IP Address', 'Device ID']
    df_cleaned = df.drop(columns=columns_to_drop)
    
    return df_cleaned

# Run the logic
data = load_and_clean_data()

# 6. Displaying the results in your Streamlit App
st.subheader("Processed Patient Data")
st.write("Below is the cleaned data where Blood Pressure is now split for AI analysis:")
st.dataframe(data.head())

# Show a quick statistic to verify the logic worked
st.sidebar.success("Data Loaded Successfully!")
st.sidebar.metric("Total Patients", len(data))

# --- STEP 2: AI MODEL LOGIC ---

st.header("🤖 AI Health Risk Analysis")

# 1. Selecting the Features (Inputs) and Target (Output)
# We use the numeric columns we processed in Step 1
features = ['Heart Rate (bpm)', 'Temperature (°C)', 'Systolic', 'Diastolic']
X = data[features]
y = data['Target']

# 2. Splitting the data
# This simulates "historical data" to train and "new data" to test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Training the XGBoost Model
# We'll create a simple button to "Train AI" so it doesn't run every single time
if st.button("Train AI Model"):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    # 4. Checking Accuracy
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    st.success(f"Model Trained! Accuracy: {acc:.2%}")
    st.info("The AI is now ready to predict health risks based on urban sensor data.")
    
    # Save the model in the session so we can use it for the buttons later
    st.session_state['health_model'] = model
    # --- STEP 3: PREDICTION INTERFACE ---

st.header("🔍 Personal Health Risk Predictor")
st.write("Enter patient vitals below to check for potential health risks.")

# Create columns for the input fields
col1, col2 = st.columns(2)

with col1:
    hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)
    temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=36.5)

with col2:
    sys = st.number_input("Systolic BP (top number)", min_value=80, max_value=200, value=120)
    dia = st.number_input("Diastolic BP (bottom number)", min_value=40, max_value=130, value=80)

# The Prediction Button logic
if st.button("Analyze My Vitals"):
    if 'health_model' in st.session_state:
        # Prepare the input for the model
        input_data = np.array([[hr, temp, sys, dia]])
        
        # Make the prediction
        prediction = st.session_state['health_model'].predict(input_data)
        
        if prediction[0] == 1:
            st.error("⚠️ ALERT: High Risk Detected. Please consult a healthcare professional.")
        else:
            st.success("✅ NORMAL: Your vitals appear to be within the healthy range.")
    else:
        st.warning("Please click 'Train AI Model' above first to prepare the AI.")

# --- STEP 3: EXPLAINABLE AI (SHAP) ---

if st.button("Analyze My Vitals"):
    if 'health_model' in st.session_state:
        model = st.session_state['health_model']
        input_data = pd.DataFrame([[hr, temp, sys, dia]], columns=features)
        
        # Prediction
        prediction = model.predict(input_data)
        
        if prediction[0] == 1:
            st.error("⚠️ ALERT: High Risk Detected.")
            
            # SHAP Logic: Why is it high risk?
            explainer = shap.Explainer(model)
            shap_values = explainer(input_data)
            
            st.write("### AI Reasoning (Why this prediction?)")
            fig, ax = plt.subplots()
            shap.plots.bar(shap_values[0], show=False)
            st.pyplot(fig)
            st.caption("Positive values (red) increase risk; negative values (blue) decrease it.")
            
        else:
            st.success("✅ NORMAL: Vitals are within healthy range.")
