# 📧 SMS / Email Spam Detection System

A Machine Learning and NLP-based web application that classifies SMS or Email messages as **Spam** or **Not Spam** in real time.

This project uses TF-IDF vectorization and Multinomial Naive Bayes for accurate spam detection and is deployed using Streamlit.

---

## 📌 Project Overview

Spam messages are a common problem in communication platforms. This project builds a text classification system that:

- Cleans and preprocesses text data
- Converts text into numerical features using TF-IDF
- Uses a trained Naive Bayes classifier
- Provides instant prediction through a web interface

---

## 🧠 Machine Learning Workflow

1. Text Preprocessing
   - Lowercasing
   - Tokenization
   - Stopword removal
   - Stemming (Porter Stemmer)

2. Feature Extraction
   - TF-IDF Vectorization

3. Model Training
   - Multinomial Naive Bayes

4. Model Serialization
   - model.pkl
   - vectorizer.pkl

5. Web Deployment
   - Streamlit Interface

---

## 📂 Project Structure

- `app.py` → Streamlit Web Application
- `model.pkl` → Trained Naive Bayes Model
- `vectorizer.pkl` → TF-IDF Vectorizer
- `main.py` → Sample script file

---

## 🖥️ Tech Stack

- Python
- NLP (NLTK)
- Scikit-learn
- TF-IDF
- Naive Bayes
- Streamlit
- Pickle

---

## ⚙️ How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/sms-spam-detection.git
cd sms-spam-detection
