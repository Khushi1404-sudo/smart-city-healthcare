import streamlit as st
import pandas as pd
import numpy as np

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
