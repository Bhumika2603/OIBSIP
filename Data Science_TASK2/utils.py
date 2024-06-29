import pickle

# Save the model
def save_model(model):
 with open('spam_classifier.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the vectorizer
def save_vectorizer(vectorizer):
 with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)