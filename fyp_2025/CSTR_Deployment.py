from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Create a Flask app
app = Flask(__name__)

# Load the model
model_path = 'best_CSTR_model.pkl'
scaler_path = 'scaler.pkl'
loaded_model = joblib.load(model_path)

# Load the scaler
try:
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    raise Exception("Scaler file 'scaler.pkl' not found. Ensure the scaler used during training is saved.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame(data)
        input_data_scaled = scaler.transform(input_data)
        prediction = loaded_model.predict(input_data_scaled)
        return jsonify({'predicted_conversion': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    from app import app

if __name__ == '__main__':
    app.run(debug=True)