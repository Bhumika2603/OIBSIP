import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, ConfusionMatrixDisplay, classification_report,confusion_matrix
import pickle
import os

class CarPricePredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
        self.model_filename = 'car_price_prediction_model.pkl'
        self.encoder_filename = 'car_price_prediction_encoder.pkl'
        self.base_dir = 'results/'
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        self.eda_dir = 'EDA_results/'
        if not os.path.exists(self.eda_dir):
            os.mkdir(self.eda_dir)
    
    def load_data(self):
        self.data = pd.read_csv(self.data_path)
    
    def explore_data(self):
        print(f'data shape:{self.data.shape}')
        # Perform exploratory data analysis (EDA)

        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['Selling_Price'], bins=20, kde=True)
        plt.title('Distribution of Car Prices')
        plt.xlabel('Selling Price')
        plt.ylabel('Frequency')
        plt.show()

        plt.savefig(os.path.join(self.eda_dir, 'Distribution of Car Prices.png'))


    
    def preprocess_data(self):
        # Example: Handling missing values and encoding categorical variables
        self.data.dropna(inplace=True)
        self.encode_data()

    def encode_data(self):
        categorical_features = self.data.select_dtypes(include=['object']).columns.tolist()
        numerical_features = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        encoder = OrdinalEncoder()
        self.data[categorical_features] = encoder.fit_transform(self.data[categorical_features])

        # encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore') 
        # encoded_features = encoder.fit_transform(self.data[categorical_features])

        self.save_encoder(encoder)

        # encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
        # predictor_transformed = pd.concat([encoded_df, self.data[numerical_features]], axis=1)
        # self.data = predictor_transformed 

    def save_encoder(self,encoder):
        # Save the one hot encoder to a file using pickle
        with open(self.encoder_filename, 'wb') as g:
            pickle.dump(encoder, g)
        print(f'Encoder saved as {self.encoder_filename}')
    
    def split_data(self):
        # Split data into training and testing sets
        X = self.data.drop(columns=['Selling_Price'])
        y = self.data['Selling_Price']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_model(self):
        # Train a machine learning model (Random Forest Regressor as an example)
        self.model = RandomForestRegressor(random_state=42)
        self.model.fit(self.X_train, self.y_train)
    
    def evaluate_model(self):
        # Evaluate the model
        self.predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, self.predictions)
        r2 = r2_score(self.y_test, self.predictions)
        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}')
        self.save_results(mse,r2)

    def save_results(self,mse,r2):

        result_dict = {
            "Mean Squared Error": mse,
            "R-squared": r2
        }

        results_path = os.path.join(self.base_dir, 'result.txt')
        with open(results_path,"w") as file1:
            for k,v in result_dict.items():
                file1.write("{} : {} \n".format(k,v))
        
    
    def save_model(self):
        # Save the trained model to a file using pickle
        with open(self.model_filename, 'wb') as f:
            pickle.dump(self.model, f)
        print(f'Model saved as {self.model_filename}')

    
    def plot_results(self):

        # Plotting actual vs predicted prices
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.predictions, alpha=0.5)
        plt.xlabel('Actual Selling Price')
        plt.ylabel('Predicted Selling Price')
        plt.title('Actual vs Predicted Car Selling Prices')
        plt.show()

        plt.savefig(os.path.join(self.base_dir, 'actual_vs_predicted.png'))
    
    def run(self):
        # Execute all steps in sequence
        self.load_data()
        self.explore_data()
        self.preprocess_data()
        self.split_data()
        self.train_model()
        self.evaluate_model()
        self.save_model()
        self.plot_results()

if __name__ == "__main__":
    data_path = r"car data.csv"
    predictor = CarPricePredictor(data_path)
    predictor.run()
