# Desription : This is email spam detection model which detect whether email is spam(1) or not spam(0) 

#Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import nltk
from nltk.corpus import stopwords
import string
import utils
import matplotlib.pyplot as plt

# Download the stopwords from NLTK
nltk.download('stopwords')

#Function to preprocess the text

def preprocess_text(text):

        #1 remove punctuation
        #2 remove stopwords: useless words in nlp
        #3 Return a list of clean text words

        # Convert text to lowercase
        text = text.lower()

        #1 Remove punctuation
        nopunc = [char for char in text if char not in string.punctuation]
        nopunc = ''.join(nopunc)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        text = ''.join(word for word in nopunc.split() if word not in stop_words)
        return text

if __name__ == "__main__":

    #LOADING THE DATASET
    dataset_path = "spam.csv"
    df = pd.read_csv(dataset_path, encoding='latin-1')

    #PREPROCESSING

    # Display the first few rows of the dataset
    print(df.head())

    #Printing rows and columns in the dataset
    print(df.shape)

    # Keep only the necessary columns
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']

    #Removing duplicate rows
    df.drop_duplicates(inplace = True)
    print(df.shape)

    #Finding Missing values in the dataset
    print(df.isnull().sum())

    #Tokenization (A list of tokens also called lemma)

    df['text'] = df['text'].apply(preprocess_text)

    # Split the data into features and labels
    X = df['text']
    y = df['label']


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the CountVectorizer : To convert a collection of text to a matrix of tokens
    vectorizer = CountVectorizer()

    # Fit and transform the training data
    X_train_vec = vectorizer.fit_transform(X_train)

    # Transform the testing data
    X_test_vec = vectorizer.transform(X_test)

    # Initialize the Multinomial Naive Bayes model
    model = MultinomialNB()

    # Train the model
    model.fit(X_train_vec, y_train)



    #Evaluating the model

    # Make predictions on the test set
    y_pred = model.predict(X_test_vec)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Print the classification report
    report = classification_report(y_test, y_pred)
    print(report)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion_matrix: \n', conf_matrix)

    cm_display = ConfusionMatrixDisplay(conf_matrix)
    cm_display.plot()
    plt.show()

    import os
    base_dir = 'results/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    result_dict = {
        "Accuracy": accuracy,
        "ConfusionMatrix": conf_matrix.tolist(),  # convert numpy array to list for better readability in the file
        "ClassificationReport": report
    }

    results_path = os.path.join(base_dir, 'result.txt')
    with open(results_path,"w") as file1:
        for k,v in result_dict.items():
            file1.write("{} : {} \n".format(k,v))
    
    print("Results saved successfully.")

