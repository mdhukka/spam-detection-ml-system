import pandas as pd
import re, os, joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

data = data[['v1', 'v2']].copy()
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text

data['message'] = data['message'].apply(preprocess)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# Save
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/spam_model.pkl")

print("Model saved")