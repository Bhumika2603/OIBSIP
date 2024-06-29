# Email Spam Detection Using Machine Learning

## Description
This project aims to build a machine learning model that can detect whether an email is spam or not. It utilizes natural language processing (NLP) techniques and a Multinomial Naive Bayes classifier to achieve this. The model is trained on a dataset of labeled emails and deployed as an API using Flask.

## Features
- Preprocesses email text to remove punctuation and stopwords.
- Uses CountVectorizer to convert text to a matrix of token counts.
- Trains a Multinomial Naive Bayes classifier to predict spam.
- Provides a Flask API for predicting whether an email is spam or not.
  
## Installation
- ### Prerequisites
  Python 3.6 or higher
  Flask
  scikit-learn
  NLTK
  pandas
  matplotlib
  
- ### Steps
1. #### Clone the repository:
   git clone https://github.com/yourusername/email-spam-detection.git
   cd email-spam-detection

2. #### Install the required packages:
   pip install -r requirements.txt

3. #### Download NLTK stopwords:
  import nltk
  nltk.download('stopwords')

## Dataset
The dataset used for training the model is available at: spam.csv. Make sure to place the dataset in the correct directory as specified in the code.

## Project Structure

email-spam-detection/

├── Confusion_matrix                      # Plot of the confusion matrix of the trained model

├── Deployment via API.py                 # Flask app to serve the model

├── main.py                               # Training script

├── utils.py                              # Some extra required functions

├── spam_classifier.pkl                    # Saved model

├── vectorizer.pkl                         # Saved vectorizer

├── results/
│   └── result.txt                        # Model evaluation results

└── README.md                             # Project documentation

## Results: 
This folder contains a result.text file which has the saved results of the various evaluation after training the model.
It contains accuracy, Classification Report and Confusion matrix.

The plot of confusion matrix is also present as Confusion_matrix.png

## Usage

### Training the Model
Ensure your dataset (spam.csv) is in the correct directory.
In this project, Naive Byes Classifier has been used for the email spam detection.

### Run the training script:
python main.py

This script will preprocess the data, train the model, evaluate it, and save the model and vectorizer as spam_classifier.pkl and vectorizer.pkl respectively. The evaluation results will be saved in the results/ directory.

### Run the evaluate script:
python evaluate.py

This script will predict an exmaple test-case using the trained model and saved vectorizer and will give results - spam(email is spam) or ham(email is not spam). You can run this script to predict a single email without using API.

## Running the Flask API

1. #### Start the Flask app:
python Deployment via API.py

The Flask app will run on http://127.0.0.1:5000.

## Making Predictions
You can test the API using tools like Postman or curl.

1. Using Postman
  Method: POST
  
  URL: http://127.0.0.1:5000/predict
  
  Headers: Content-Type: application/json
  
  Body (raw, JSON):

   {
    "text": "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize."
   }

2. Using curl
    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"text\":\"Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize.\"}"

## Response
The API will return a JSON response indicating whether the email is spam (1) or not spam (0).

## Model Evaluation
The model's performance is evaluated using accuracy, confusion matrix, and classification report. The results are saved in the results/result.txt file.
After training, we achieved an accuracy of 86% using naive byes classifier

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request. We welcome improvements and suggestions.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
The dataset is provided by UCI Machine Learning Repository.
Special thanks to the contributors of the Python libraries used in this project.






