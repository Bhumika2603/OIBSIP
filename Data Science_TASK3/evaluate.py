import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder

test_cases = [
    {
        'Car_Name': 'Toyota Camry',
        'Year': 2018,  
        'Present_Price': 10.15,
        'Driven_kms': 40000,
        'Fuel_Type': 'Petrol',
        'Selling_type': 'Dealer',
        'Transmission': 'Manual',
        'Owner': 1
    },
    {
        'Car_Name': 'Honda Accord',
        'Year': 2017,   
        'Present_Price': 11.5,
        'Driven_kms': 35000,
        'Fuel_Type': 'Petrol',
        'Selling_type': 'Dealer',
        'Transmission': 'Automatic',
        'Owner': 1
    }
]

def load_model(model_filename):
    # Load the saved model from pickle file
    with open(model_filename, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

def load_encoder(encoder_filename):
    # Load the saved encoder from pickle file
    with open(encoder_filename, 'rb') as e:
        loaded_encoder = pickle.load(e)
    return loaded_encoder

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

def predict_price(model, test_df):
    # Make predictions using the loaded model
    predicted_price = model.predict(test_df)
    return predicted_price

if __name__ == "__main__":
    model_filename = 'car_price_prediction_model.pkl'  
    loaded_model = load_model(model_filename)

    encoder_filename = 'car_price_prediction_encoder.pkl'
    loaded_encoder = load_encoder(encoder_filename)
    
    for idx, test_case in enumerate(test_cases):
        print(f"Test Case {idx + 1}:")
        print(f"Input: {test_case}")
        
        test_df = preprocess_test_case(test_case, loaded_encoder)
        predicted_price = predict_price(loaded_model, test_df)
        
        print(f"Predicted Price: ${predicted_price[0]:,.2f}")
        print()

