# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
from preprocessing import clean_text

# Load dataset
df = pd.read_csv("data/train.csv")

# If no 'text' column, combine title/author/text
if "text" not in df.columns:
    df["text"] = df.fillna("").apply(lambda r: " ".join([str(r.get(c, "")) for c in ["title", "author", "text"] if c in r]), axis=1)

df["clean_text"] = df["text"].apply(clean_text)

X = df["clean_text"]
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save
joblib.dump(pipeline, "model/fake_news_pipeline.joblib")
print("Model saved at model/fake_news_pipeline.joblib")
