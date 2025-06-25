import streamlit as st
import pickle
import pandas as pd
from bayes_opt import BayesianOptimization

# Load the trained model
with open('pfr_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the app
st.title("Plug Flow Reactor (PFR) Conversion Predictor and Optimizer")

# Description of feature ranges
st.markdown("""
### Feature Ranges:
- **Flow Rate of Ethyl Acetate (L/s)**: 0.001 to 10.0  
- **Flow Rate of NaOH (L/s)**: 0.001 to 10.0  
- **Concentration of Ethyl Acetate (Mol/L)**: 0.01 to 1.0  
- **Concentration of NaOH (Mol/L)**: 0.01 to 1.0  
- **Reactor Volume (L)**: 0.5 to 500.0  
- **Temperature (Kelvin)**: 298.15 to 333.15  
""")

# Input fields for the features
flow_rate_ea = st.number_input("Flow Rate of Ethyl Acetate (L/s)", min_value=0.001, max_value=10.0, value=1.0, step=0.01)
flow_rate_naoh = st.number_input("Flow Rate of NaOH (L/s)", min_value=0.001, max_value=10.0, value=1.0, step=0.01)
conc_ea = st.number_input("Concentration of Ethyl Acetate (Mol/L)", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
conc_naoh = st.number_input("Concentration of NaOH (Mol/L)", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
reactor_vol = st.number_input("Reactor Volume (L)", min_value=0.5, max_value=500.0, value=100.0, step=1.0)
temp = st.number_input("Temperature (Kelvin)", min_value=298.15, max_value=333.15, value=320.0, step=1.0)

# Button to predict conversion
if st.button("Predict Conversion"):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Flow_Rate_EA (L/s)': [flow_rate_ea],
        'Flow_Rate_NaOH (L/s)': [flow_rate_naoh],
        'Conc_EA (Mol/L)': [conc_ea],
        'Conc_NaOH (Mol/L)': [conc_naoh],
        'Reactor_Vol (L)': [reactor_vol],
        'Temp (Kelvin)': [temp]
    })

    # Predict conversion
    prediction = model.predict(input_data)

    # Display the result
    st.success(f"Predicted Conversion: {prediction[0]:.4f}")

# Optimization section
st.markdown("### Find Optimum Conditions for Maximum Conversion")

# Define the objective function for Bayesian Optimization
def objective_function(flow_rate_ea, flow_rate_naoh, conc_ea, conc_naoh, reactor_vol, temp):
    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'Flow_Rate_EA (L/s)': [flow_rate_ea],
        'Flow_Rate_NaOH (L/s)': [flow_rate_naoh],
        'Conc_EA (Mol/L)': [conc_ea],
        'Conc_NaOH (Mol/L)': [conc_naoh],
        'Reactor_Vol (L)': [reactor_vol],
        'Temp (Kelvin)': [temp]
    })
    
    # Predict the conversion using the trained model
    predicted_conversion = model.predict(input_data)
    
    # Return the predicted conversion as the objective to maximize
    return float(predicted_conversion[0])

# Define the bounds for the decision variables
pbounds = {
    'flow_rate_ea': (0.001, 10.0),
    'flow_rate_naoh': (0.001, 10.0),
    'conc_ea': (0.01, 1.0),
    'conc_naoh': (0.01, 1.0),
    'reactor_vol': (0.5, 500.0),
    'temp': (298.15, 333.15)
}

# Perform optimization when the button is clicked
if st.button("Find Optimum Conditions"):
    # Initialize the Bayesian Optimizer
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42
    )

    # Perform the optimization
    optimizer.maximize(
        init_points=10,  # Number of random initial points
        n_iter=50        # Number of iterations for optimization
    )

    # Extract the optimal conditions
    optimal_conditions = optimizer.max['params']
    optimal_conversion = optimizer.max['target']

    # Display the results
    st.success("Optimal Conditions Found!")
    st.write(f"**Flow Rate EA (L/s):** {optimal_conditions['flow_rate_ea']:.4f}")
    st.write(f"**Flow Rate NaOH (L/s):** {optimal_conditions['flow_rate_naoh']:.4f}")
    st.write(f"**Concentration EA (Mol/L):** {optimal_conditions['conc_ea']:.4f}")
    st.write(f"**Concentration NaOH (Mol/L):** {optimal_conditions['conc_naoh']:.4f}")
    st.write(f"**Reactor Volume (L):** {optimal_conditions['reactor_vol']:.4f}")
    st.write(f"**Temperature (Kelvin):** {optimal_conditions['temp']:.4f}")
    st.write(f"**Maximum Predicted Conversion:** {optimal_conversion:.4f}")