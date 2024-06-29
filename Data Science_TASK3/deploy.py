from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the saved model
model_filename = 'car_price_prediction_model.pkl'  
with open(model_filename, 'rb') as f:
    model = pickle.load(f)

## Load the saved encoder
encoder_filename = 'car_price_prediction_encoder.pkl'
with open(encoder_filename, 'rb') as e:
        encoder = pickle.load(e)
   

def preprocess_test_case(test_case, encoder):
    # Convert test case to DataFrame for prediction
    test_df = pd.DataFrame([test_case])
    preprocessed_test_df = encode_testcase(test_df, encoder)
    return preprocessed_test_df

def encode_testcase(test_df, encoder):
    categorical_features = test_df.select_dtypes(include=['object']).columns.tolist()
    numerical_features = test_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    test_df[categorical_features] = encoder.fit_transform(test_df[categorical_features])
    return test_df

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    test_df = preprocess_test_case(data,encoder)
    prediction = model.predict(test_df)
    predicted_price = prediction[0]
    response = {'predicted_price': predicted_price}
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
