import re
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

model = joblib.load("model/spam_model.pkl")

SPAM_KEYWORDS = [
    "free","win","winner","prize","cash","claim","urgent","click",
    "offer","limited","congratulations","selected","guaranteed",
    "discount","loan","credit","verify","account","suspend",
    "password","bank","earn","income","work from home",
    "make money","dear friend","you won","lottery","click"
]

class RequestModel(BaseModel):
    message: str

class BatchRequest(BaseModel):
    messages: List[str]

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def confidence_label(p):
    if p >= 80: return "HIGH RISK"
    elif p >= 50: return "MEDIUM RISK"
    elif p >= 30: return "LOW RISK"
    elif p>= 20: return "MIGHT BE RISKY"
    else: return "SAFE"

def risk_words(text):
    return [kw for kw in SPAM_KEYWORDS if kw in text.lower()]

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(req: RequestModel):
    if not req.message.strip():
        raise HTTPException(400, "Empty message")

    text = preprocess(req.message)

    pred = model.predict([text])[0]
    probs = model.predict_proba([text])[0]

    spam_prob = round(probs[1]*100, 2)
    ham_prob = round(probs[0]*100, 2)

    return {
        "prediction": "spam" if pred == 1 else "ham",
        "spam_probability": spam_prob,
        "ham_probability": ham_prob,
        "confidence": confidence_label(spam_prob),
        "risk_factors": risk_words(req.message),
        "length": len(req.message),
        "words": len(req.message.split())
    }

@app.post("/batch")
def batch(req: BatchRequest):
    results = []
    for msg in req.messages:
        if not msg.strip():
            continue

        text = preprocess(msg)
        pred = model.predict([text])[0]
        prob = model.predict_proba([text])[0][1]

        results.append({
            "message": msg[:50],
            "prediction": "spam" if pred == 1 else "ham",
            "spam_probability": round(prob*100,2)
        })

    return {"results": results}