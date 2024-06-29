
import pickle
import Train

#LOADING THE TRAINED MODEL AND VECTORIZER

with open('spam_classifier.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    loaded_vectorizer = pickle.load(vec_file)

#TESTING THE MODEL

# Function to predict if an email is spam or not
def predict_spam(text):

    # Preprocess the text
    text = Train.preprocess_text(text)

    # Vectorize the text
    text_vec = loaded_vectorizer.transform([text])

    # Predict
    prediction = loaded_model.predict(text_vec)
    return prediction[0]

# Example usage
print(predict_spam("Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize."))