import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Smart Electricity Theft Detection")

st.title("⚡ Smart Electricity Theft Detection")
st.subheader("Real Electricity Data - Anomaly Detection")

# Load trained model
model = joblib.load("model/theft_model.pkl")

# Inputs
r_usage = st.number_input(
    "R Phase Usage",
    min_value=0.0,
    value=100.0
)

y_usage = st.number_input(
    "Y Phase Usage",
    min_value=0.0,
    value=100.0
)

b_usage = st.number_input(
    "B Phase Usage",
    min_value=0.0,
    value=100.0
)

ac_status = st.number_input(
    "AC Status",
    min_value=0.0,
    max_value=1.0,
    value=1.0
)

if st.button("Predict"):

    # Input data for ML model
    input_data = pd.DataFrame(
        [[r_usage, y_usage, b_usage, ac_status]],
        columns=["R", "Y", "B", "AC Status"]
    )

    # ML prediction
    model_prediction = model.predict(input_data)

    # -------- DEMO DECISION LOGIC --------
    total_usage = r_usage + y_usage + b_usage

    max_phase = max(r_usage, y_usage, b_usage)
    min_phase = min(r_usage, y_usage, b_usage)

    if min_phase == 0:
        phase_imbalance = max_phase
    else:
        phase_imbalance = max_phase / min_phase

    # Suspicious conditions
    suspicious = (
        total_usage > 500
        or phase_imbalance > 3
        or ac_status == 0
    )

    # Clear single result
    if suspicious:
        st.error("🚨 Suspicious Electricity Consumption")
        st.warning("⚠️ Possible Electricity Theft / Abnormal Usage Detected")
    else:
        st.success("✅ Normal Electricity Consumption")