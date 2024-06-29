# Car Price Prediction with Machine Learning

This project is a comprehensive solution for predicting the selling price of used cars using machine learning. It involves data preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment using a Flask API. The project utilizes a dataset containing various attributes of cars, such as the car's name, year, present price, kilometers driven, fuel type, seller type, transmission type, and the number of previous owners.

- In this model we have used Random Forest Regression algorithm to train a machine learning model

- We have achieved:
   Mean Squared Error : 0.782
   R-squared : 0.966 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Model Deployment](#model-deployment)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/car-price-prediction.git
   cd car-price-prediction
2. **Install the required packages:**
   pip install -r requirements.txt
3. **Prepare the dataset:**
   Ensure that you have the dataset file (car data.csv) in the project directory.

## Usage

### Data Exploration and Preprocessing
Run the main.py script to perform data exploration, preprocessing, and model training.
python main.py

#### Main Steps:

1. **Load Data:**
Loads the dataset from the specified path.

2. **Exploratory Data Analysis (EDA):**
Generates various plots to understand the distribution of car prices and other features.

3. **Preprocess Data:**
Handles missing values and encodes categorical variables using OrdinalEncoder.

4. **Split Data:**
Splits the data into training and testing sets.

5. **Train Model:**
Trains a Random Forest Regressor on the training data.

6. **Evaluate Model:**
Evaluates the model using mean squared error (MSE) and R-squared metrics.

7. **Save Model:**
Saves the trained model and encoder to disk.

8. **Plot Results:**
Plots the actual vs. predicted car prices.

## Model Training and Evaluation
The training and evaluation steps are encapsulated in the CarPricePredictor class in main.py. The class performs data loading, preprocessing, model training, and evaluation in sequence when the run method is called.

## Model Deployment
Deploy the trained model using a Flask API.

1. **Start the Flask server:**
    python deploy.py
2. **Send a POST request to the server to predict car prices:**
   You can use curl to send a test request.
   curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"Car_Name": "Toyota Camry", "Year": 2018, "Present_Price": 10.15, "Driven_kms": 40000, "Fuel_Type": "Petrol", "Selling_type": "Dealer", "Transmission": "Manual", "Owner": 1}'

## Evaluating Model
Use evaluate.py to load the trained model and encoder, and predict prices for new test cases.
python evaluate.py

## Exploratory Data Analysis (EDA)
Use eda.py to perform detailed exploratory data analysis and generate visualizations.
python eda.py

## Results
The results of the model evaluation are saved in the results/ directory. The result.txt file contains the Mean Squared Error (MSE) and R-squared metrics. Additionally, visualizations such as the distribution of car prices and actual vs. predicted prices are saved as PNG files in the results/ and EDA_results/ directories.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.




