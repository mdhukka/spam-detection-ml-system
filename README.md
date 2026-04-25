# 📩 Spam Detection System (ML + FastAPI + Streamlit)

## 🚀 Overview

This project is a full-stack machine learning application that detects whether a message is spam or not.

It combines:

* Machine Learning (TF-IDF + Logistic Regression)
* FastAPI backend (for prediction API)
* Streamlit frontend (for user interaction)

---

## 🧠 Features

* Real-time spam detection
* Probability-based confidence scoring
* Risk factor keyword detection
* Batch message processing
* Interactive UI

---

## 🏗️ Tech Stack

* Python
* Scikit-learn
* FastAPI
* Streamlit
* Joblib

---

## ⚙️ How It Works

1. Text is preprocessed (lowercase + punctuation removal)
2. Converted into numerical vectors using TF-IDF
3. Logistic Regression predicts spam/ham
4. API returns prediction with probabilities and explanations

---

## ▶️ Run Locally

### 1. Train model

```
python train_model.py
```

### 2. Start API

```
uvicorn main:app --reload
```

### 3. Start UI

```
streamlit run ui.py
```

---

## 📊 Example Output

* Prediction: Spam
* Confidence: HIGH RISK
* Spam Probability: 92%
* Risk Factors: ["free", "win", "offer"]

---

## 🎯 Purpose

Built as an end-to-end ML project to demonstrate:

* Model building
* API development
* Frontend integration

---

## 📌 Future Improvements

* Deploy on cloud (Render / AWS)
* Use deep learning models (BERT)
* Add user authentication
* Improve dataset quality
