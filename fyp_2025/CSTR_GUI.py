# CSTR_GUI.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load('best_CSTR_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("CSTR Conversion Prediction & Optimization")

st.header("🔢 Enter CSTR Operating Conditions")

# User input fields
flow_rate_ea = st.number_input("Flow Rate EA (L/s)", min_value=0.001, max_value=10.0, value=5.0, step=0.01)
flow_rate_naoh = st.number_input("Flow Rate NaOH (L/s)", min_value=0.001, max_value=10.0, value=5.0, step=0.01)
conc_ea = st.number_input("Concentration EA (Mol/L)", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
conc_naoh = st.number_input("Concentration NaOH (Mol/L)", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
reactor_vol = st.number_input("Reactor Volume (L)", min_value=0.5, max_value=500.0, value=100.0, step=0.1)
temp = st.number_input("Temperature (Kelvin)", min_value=298.15, max_value=333.15, value=320.0, step=0.1)

# Predict conversion for user input
if st.button("Predict Conversion"):
    input_df = pd.DataFrame({
        'Flow_Rate_EA (L/s)': [flow_rate_ea],
        'Flow_Rate_NaOH (L/s)': [flow_rate_naoh],
        'Conc_EA (Mol/L)': [conc_ea],
        'Conc_NaOH (Mol/L)': [conc_naoh],
        'Reactor_Vol (L)': [reactor_vol],
        'Temp (Kelvin)': [temp]
    })
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    st.success(f"Predicted Conversion (X_A): {pred:.4f}")

st.header("🌟 Optimum Conversion & Conditions")

# Show optimum value and features from Bayesian Optimization
if st.button("Show Optimum Conditions"):
    # If you saved the optimizer results, load them here.
    # Otherwise, you can paste the optimum values from your notebook run.
    # Example (replace with your actual optimum values):
    optimal_conditions = {
        'Flow Rate EA (L/s)': 1.2345,
        'Flow Rate NaOH (L/s)': 2.3456,
        'Concentration EA (Mol/L)': 0.5678,
        'Concentration NaOH (Mol/L)': 0.6789,
        'Reactor Volume (L)': 123.45,
        'Temperature (Kelvin)': 310.12
    }
    optimal_conversion = 0.9876

    st.write("**Optimal Conditions for Maximum Conversion:**")
    st.json(optimal_conditions)
    st.success(f"Maximum Predicted Conversion (X_A): {optimal_conversion:.4f}")

st.info("Tip: You can update the optimum values above with your actual results from Bayesian Optimization.")