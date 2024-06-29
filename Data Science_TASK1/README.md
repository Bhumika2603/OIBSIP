# Iris Flower Classification Model

## Overview
This project involves building and deploying a machine learning model to classify iris flowers into three species: setosa, versicolor, and virginica. The model is built using the `scikit-learn` library and deployed as a Flask API.

## Project Structure

Here's the complete content in code form for the README file:

markdown
Copy code
# Iris Flower Classification Model

## Overview
This project involves building and deploying a machine learning model to classify iris flowers into three species: setosa, versicolor, and virginica. The model is built using the `scikit-learn` library and deployed as a Flask API.

## Project Structure
.

├── EDA_results/

   │ ├── pairplot.png

   │ ├── Correlation Heatmap.png

   │ ├── feature_distributions.png

├── results/

   │ ├── result.txt

├── iris_flower_classification_model.pkl

├── iris_flower_classification_scaler.pkl

├── iris_main.py

├── iris_deploy.py

├── requirements.txt

├── README.md


## Requirements
- Python 3.7 or later
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `flask`, `pickle`, `os`

- Install the required packages using:
   pip install -r requirements.txt

## Dataset
The iris dataset is used in this project. It is a classic dataset in machine learning and is available in the scikit-learn library.

## Project Components
1. **iris_main.py**
This script performs the following tasks:

-Loads the iris dataset.
-Conducts exploratory data analysis (EDA) including pairplots, heatmaps, and feature distributions.
-Splits the dataset into training and testing sets.
-Trains a logistic regression model.
-Evaluates the model and saves the results.
-Saves the trained model and scaler for later use.

2. **iris_deploy.py**
This script creates a Flask web application that serves the trained model for making predictions. It includes:

- An endpoint for home which returns a welcome message.
- An endpoint /predict that accepts POST requests with JSON data, preprocesses the data, makes predictions, and returns the predicted class.

## Running the Project

### Step 1: Train the Model
Run the iris_flower_classifier.py script to train the model and save the results:
python iris_flower_classifier.py

### Step 2: Run the Flask App
Run the app.py script to start the Flask API:
python app.py

### Step 3: Make Predictions
Send a POST request to the /predict endpoint with the feature values. Here is an example using curl:

  curl -X POST -H "Content-Type: application/json" -d '{
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2
}' http://127.0.0.1:5000/predict

## Exploratory Data Analysis (EDA)
The EDA results are saved in the EDA_results directory:

- **pairplot.png:** Pairplot of the features colored by target species.
- **Correlation Heatmap.png:** Heatmap showing the correlation between features.
- **feature_distributions.png:** Histograms of feature distributions.

## Model Evaluation
The evaluation metrics are saved in the results directory in the result.txt file, which includes accuracy, classification report, and confusion matrix.

## Saving the Model
The trained model and scaler are saved as **iris_flower_classification_model.pkl** and **iris_flower_classification_scaler.pkl** respectively.

## Deployment on GitHub
1. Create a new repository on GitHub.
2. Clone the repository to your local machine.
3. Copy all project files to the cloned repository.
4. Add and commit the files:
   
    git add .
   
    git commit -m "Initial commit"
6. Push the changes to the GitHub repository:
   
    git push origin main

## License
This project is licensed under the MIT License.

## Acknowledgements
- The iris dataset is provided by scikit-learn.
- This project is inspired by various tutorials and documentation on machine learning and Flask


