import streamlit as st
import streamlit.components.v1 as components

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="HealthMate AI - Smart City Home", layout="wide")

# 2. DESIGN LAYER: LOAD YOUR CUSTOM HTML
try:
    with open("app.html", "r", encoding='utf-8') as f:
        html_design = f.read()
    components.html(html_design, height=800, scrolling=True)
except FileNotFoundError:
    st.error("⚠️ app.html not found! Please ensure it is in the main folder.")

st.sidebar.title("Navigation")
st.sidebar.info("Use the menu above to access the AI Diagnostic Terminal.")
