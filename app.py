import streamlit as st
import joblib
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Campus Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load trained model
model = joblib.load("model_campus_placement")

# Header
st.title("🎓 Campus Placement Prediction System")
st.write("Predict whether a student is likely to be placed using a trained Logistic Regression model.")

st.divider()

# Personal details
st.subheader("👤 Personal Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    workex = st.selectbox("Work Experience", ["Yes", "No"])

# Academic details
st.subheader("📚 Academic Information")

col1, col2 = st.columns(2)

with col1:
    ssc_p = st.number_input("SSC Percentage (10th)", min_value=0.0, max_value=100.0, value=75.0)
    ssc_b = st.selectbox("SSC Board", ["Central", "Others"])
    hsc_p = st.number_input("HSC Percentage (12th)", min_value=0.0, max_value=100.0, value=72.0)
    hsc_b = st.selectbox("HSC Board", ["Central", "Others"])

with col2:
    hsc_s = st.selectbox("HSC Stream", ["Science", "Commerce", "Arts"])
    degree_p = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, value=70.0)
    degree_t = st.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt", "Others"])
    etest_p = st.number_input("E-Test Percentage", min_value=0.0, max_value=100.0, value=80.0)

# MBA details
st.subheader("🎓 MBA Information")

col1, col2 = st.columns(2)

with col1:
    specialisation = st.selectbox("MBA Specialisation", ["Mkt&HR", "Mkt&Fin"])

with col2:
    mba_p = st.number_input("MBA Percentage", min_value=0.0, max_value=100.0, value=75.0)

st.divider()

# Prediction button
if st.button("🔮 Predict Placement", use_container_width=True):

    # Convert categorical values exactly like your Tkinter code
    p1 = 1 if gender == "Male" else 0
    p3 = 1 if ssc_b == "Central" else 0
    p5 = 1 if hsc_b == "Central" else 0

    if hsc_s == "Science":
        p6 = 2
    elif hsc_s == "Commerce":
        p6 = 1
    else:
        p6 = 0

    if degree_t == "Sci&Tech":
        p8 = 2
    elif degree_t == "Comm&Mgmt":
        p8 = 1
    else:
        p8 = 0

    p9 = 1 if workex == "Yes" else 0
    p11 = 1 if specialisation == "Mkt&HR" else 0

    # Create DataFrame with exact model features
    new_data = pd.DataFrame({
        "gender": [p1],
        "ssc_p": [ssc_p],
        "ssc_b": [p3],
        "hsc_p": [hsc_p],
        "hsc_b": [p5],
        "hsc_s": [p6],
        "degree_p": [degree_p],
        "degree_t": [p8],
        "workex": [p9],
        "etest_p": [etest_p],
        "specialisation": [p11],
        "mba_p": [mba_p]
    })

    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[0][1] * 100

    st.divider()

    if prediction == 1:
        st.success("🎉 Student is likely to be PLACED")
        st.metric("Placement Probability", f"{probability:.2f}%")
    else:
        st.error("❌ Student is unlikely to be placed")
        st.metric("Placement Probability", f"{probability:.2f}%")

st.divider()
st.caption("Machine Learning • Logistic Regression • Campus Placement Prediction")