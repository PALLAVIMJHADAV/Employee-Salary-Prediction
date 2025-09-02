# app.py

import streamlit as st
import joblib
import numpy as np
import os

# Page configuration
st.set_page_config(page_title="Employee Salary Prediction", layout="centered")

# Background options
backgrounds = {
    "Soft Blue": "linear-gradient(to right, #e0f7fa, #ffffff);",
    "Pastel Peach": "#FFF5E1;",
    "Misty Green": "linear-gradient(to right, #f0fff4, #d9fdd3);",
    "Elegant Grey": "linear-gradient(to right, #fdfbfb, #ebedee);"
}

# Sidebar background selector
bg_choice = st.sidebar.selectbox("Choose Background Style", list(backgrounds.keys()))

# Apply selected background
st.markdown(f"""
    <style>
    .stApp {{
        background: {backgrounds[bg_choice]};
        font-family: 'Segoe UI', sans-serif;
        color: #2C3E50;
    }}
    .title {{
        text-align: center;
        color: #2C3E50;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }}
    .subtitle {{
        text-align: center;
        color: #555;
        font-size: 18px;
        margin-bottom: 30px;
    }}
  
    h4 {{
        font-weight: bold;
        color: #1A1A1A;
        margin-top: 15px;
        margin-bottom: 5px;
    }}
    .stButton>button {{
        background-color: #3498db;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 24px;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        background-color: #2980b9;
    }}
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("<div class='title'>Employee Salary Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enter employee details to get an estimated salary</div>", unsafe_allow_html=True)

# File paths
MODEL_FILE = "salary_model.pkl"
DEPT_ENCODER_FILE = "Department_encoder.pkl"
POS_ENCODER_FILE = "Position_encoder.pkl"

if not (os.path.exists(MODEL_FILE) and os.path.exists(DEPT_ENCODER_FILE) and os.path.exists(POS_ENCODER_FILE)):
    st.error("Model files not found! Please train the model first and ensure salary_model.pkl, Department_encoder.pkl, and Position_encoder.pkl are in this folder.")
else:
    # Load model and encoders
    model = joblib.load(MODEL_FILE)
    dept_encoder = joblib.load(DEPT_ENCODER_FILE)
    pos_encoder = joblib.load(POS_ENCODER_FILE)

   # Input form inside a styled card
    st.markdown("<div class='form-card'>", unsafe_allow_html=True)

    st.markdown("<h4>👤 Age</h4>", unsafe_allow_html=True)
    age = st.number_input("Age", min_value=18, max_value=65, value=30, label_visibility="collapsed")

    st.markdown("<h4>📊 Projects Completed</h4>", unsafe_allow_html=True)
    projects = st.number_input("Projects", min_value=0, value=5, label_visibility="collapsed")

    st.markdown("<h4>⚡ Productivity (%)</h4>", unsafe_allow_html=True)
    productivity = st.slider("Productivity", min_value=0, max_value=100, value=75, label_visibility="collapsed")

    st.markdown("<h4>😊 Satisfaction Rate (%)</h4>", unsafe_allow_html=True)
    satisfaction = st.slider("Satisfaction", min_value=0, max_value=100, value=80, label_visibility="collapsed")

    st.markdown("<h4>🏢 Department</h4>", unsafe_allow_html=True)
    department = st.selectbox("Department", options=dept_encoder.classes_, label_visibility="collapsed")

    st.markdown("<h4>💼 Position</h4>", unsafe_allow_html=True)
    position = st.selectbox("Position", options=pos_encoder.classes_, label_visibility="collapsed")

    st.markdown("<h4>⭐ Feedback Score</h4>", unsafe_allow_html=True)
    feedback = st.number_input("Feedback", min_value=0.0, max_value=10.0, value=8.0, label_visibility="collapsed")


    if st.button("Predict Salary"):
        dept_encoded = dept_encoder.transform([department])[0]
        pos_encoded = pos_encoder.transform([position])[0]
        features = np.array([[age, projects, productivity, satisfaction, dept_encoded, pos_encoded, feedback]])
        prediction = model.predict(features)[0]

        # Dark, bold prediction text
        st.markdown(
            f"""
            <h3 style='text-align: center; color: #1A1A1A; font-weight: bold;'>
                Predicted Salary: ₹{prediction:,.2f}
            </h3>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
