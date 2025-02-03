from flask import Flask, render_template, request
import pickle
import re
# from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# import nltk


app = Flask(__name__)

# Load the model and vectorizer
with open('fake_news_model.pkl', 'rb') as f:
    model, tfidf = pickle.load(f)


import nltk

try:
    from nltk.corpus import stopwords  # Try to import stopwords
    # If it works, the data is there
except LookupError:
    print("Downloading NLTK data (stopwords)...")
    nltk.download('stopwords', download_dir='/opt/render/nltk_data') #specify download directory
    from nltk.corpus import stopwords  # Import again after download
try:
     nltk.word_tokenize("test") # Try to tokenize to check if punkt is present
except LookupError:
     print("Downloading punkt data...")
     nltk.download('punkt', download_dir='/opt/render/nltk_data')  # Download if not present
     nltk.download('averaged_perceptron_tagger', download_dir='/opt/render/nltk_data') #download this as well


     

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    # tokens = word_tokenize(text)  # Remove this line
    tokens = text.split()  # Use Python's split() instead
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        user_input = request.form["news_text"]
        if user_input:
            processed_input = preprocess_text(user_input)
            vectorized_input = tfidf.transform([processed_input])
            prediction = model.predict(vectorized_input)[0]

    return render_template("index.html", prediction=prediction)  # Render the HTML template

if __name__ == "__main__":
    app.run(debug=True)  # debug=True for development; set to False in production