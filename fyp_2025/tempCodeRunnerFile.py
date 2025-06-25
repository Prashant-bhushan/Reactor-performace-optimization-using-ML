import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model_path = 'catboost_model.pkl'
scaler_path = 'scaler.pkl'

try:
    catboost_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()

# Streamlit app
st.title("Batch Reactor Conversion Prediction")
st.write("This app predicts the conversion (\(X_A\)) for a Batch Reactor and optimizes reactor performance.")

# Display feature ranges for user guidance
st.markdown("""
### Feature Ranges:
- **Ethyl Acetate Concentration (Ca0)**: 0.01 to 0.1 (Mol/L)  
- **NaOH Concentration (Cb0)**: 0.01 to 0.1 (Mol/L)  
- **Temperature (T)**: 22°C to 57°C  
- **Time of Reaction**: 5 to 30 minutes  
""")

# Input fields for the user
ethyl_acetate = st.number_input("Ethyl Acetate Concentration (Ca0) (Mol/L)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
naoh = st.number_input("NaOH Concentration (Cb0) (Mol/L)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
temperature = st.number_input("Temperature (T) (°C)", min_value=22.0, max_value=57.0, value=40.0, step=1.0)
time_of_reaction = st.number_input("Time of Reaction (minutes)", min_value=5.0, max_value=30.0, value=15.0, step=1.0)

# Predict button
if st.button("Predict Conversion"):
    try:
        # Create a DataFrame for the input data
        input_data = pd.DataFrame({
            'Ethyl_acetate(Ca0)': [ethyl_acetate],
            'NaOH(Cb0)': [naoh],
            'Temperature(T(°C))': [temperature],
            'Time_of_reaction(min)': [time_of_reaction]
        })

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Predict conversion
        prediction = catboost_model.predict(input_data_scaled)

        # Display the result
        st.success(f"Predicted Conversion (X_A): {prediction[0]:.4f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Optimization section
st.markdown("### Optimize Reactor Performance")
if st.button("Optimize"):
    from bayes_opt import BayesianOptimization

    # Define the objective function for optimization
    def objective_function(Ca0, Cb0, T, time):
        input_data = pd.DataFrame({
            'Ethyl_acetate(Ca0)': [Ca0],
            'NaOH(Cb0)': [Cb0],
            'Temperature(T(°C))': [T],
            'Time_of_reaction(min)': [time]
        })
        input_data_scaled = scaler.transform(input_data)
        predicted_conversion = catboost_model.predict(input_data_scaled)
        return float(predicted_conversion[0])

    # Define the bounds for the decision variables
    pbounds = {
        'Ca0': (0.01, 0.1),
        'Cb0': (0.01, 0.1),
        'T': (22.0, 57.0),
        'time': (5.0, 30.0)
    }

    # Perform Bayesian Optimization
    optimizer = BayesianOptimization(f=objective_function, pbounds=pbounds, random_state=42)
    optimizer.maximize(init_points=10, n_iter=50)

    # Extract the optimal conditions
    optimal_conditions = optimizer.max['params']
    optimal_conversion = optimizer.max['target']

    # Display the results
    st.write("#### Optimal Conditions:")
    st.write(f"- **Ethyl Acetate Concentration (Ca0)**: {optimal_conditions['Ca0']:.4f} Mol/L")
    st.write(f"- **NaOH Concentration (Cb0)**: {optimal_conditions['Cb0']:.4f} Mol/L")
    st.write(f"- **Temperature (T)**: {optimal_conditions['T']:.2f} °C")
    st.write(f"- **Time of Reaction**: {optimal_conditions['time']:.2f} minutes")
    st.success(f"Maximum Predicted Conversion (X_A): {optimal_conversion:.4f}")
    
    