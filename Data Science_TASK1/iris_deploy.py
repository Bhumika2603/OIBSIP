from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
import pickle

# Load the trained model and scaler
model_filename = 'iris_flower_classification_model.pkl'
with open(model_filename, 'rb') as f:
    model = pickle.load(f)

scaler_filename = 'iris_flower_classification_scaler.pkl'
with open(scaler_filename, 'rb') as g:
    scaler = pickle.load(g)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Iris Flower Classification API"

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    
    # Preprocess the data
    df_scaled = scaler.transform(df)
    
    # Make prediction
    prediction = model.predict(df_scaled)
    
    # Send prediction back as JSON
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
