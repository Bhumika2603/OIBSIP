from flask import Flask, request, jsonify
import pickle
import Train

app = Flask(__name__)

# Load the trained model and vectorizer
with open('spam_classifier.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    loaded_vectorizer = pickle.load(vec_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    
    # Preprocess the text
    processed_text = Train.preprocess_text(text)
    
    # Vectorize the text
    text_vec = loaded_vectorizer.transform([processed_text])
    
    # Predict
    prediction = loaded_model.predict(text_vec)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

# python app.py
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text":"Congratulations! You have won a $1000 Walmart gift card. Click here to claim your prize."}'
#{
#   "text": "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize."
# }
