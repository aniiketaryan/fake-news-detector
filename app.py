# app.py
from flask import Flask, render_template, request
import joblib
from preprocessing import clean_text

app = Flask(__name__)

# Load trained pipeline
pipeline = joblib.load("model/fake_news_pipeline.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form["news"]
    cleaned_text = clean_text(news_text)
    prediction = pipeline.predict([cleaned_text])[0]
    confidence = pipeline.predict_proba([cleaned_text]).max() * 100
    result = "Fake News ❌" if prediction == 1 else "Real News ✅"
    return render_template("index.html", prediction_text=f"{result} (Confidence: {confidence:.2f}%)")

if __name__ == "__main__":
    app.run(debug=True)
