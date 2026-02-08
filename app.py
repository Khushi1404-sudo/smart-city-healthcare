import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
