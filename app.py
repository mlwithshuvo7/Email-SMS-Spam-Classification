from flask import Flask, render_template, request
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer_email.pkl', 'rb'))
model = pickle.load(open('email_spam_model.pkl', 'rb'))

# Initialize stemmer
ps = PorterStemmer()


# Text preprocessing function
def transform_text(text):
    # Lowercase
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    y = []

    # Remove special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Home route
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    input_sms = request.form['message']

    # Check empty input
    if input_sms.strip() == "":
        return render_template(
            'index.html',
            prediction="Please enter a message."
        )

    # Preprocess text
    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Show result
    if result == 1:
        prediction = "Spam Email "
    else:
        prediction = "Not Spam "

    return render_template(
        'index.html',
        prediction=prediction,
        message=input_sms
    )


# Run app
if __name__ == '__main__':
    app.run(debug=True)