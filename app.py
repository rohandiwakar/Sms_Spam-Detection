import streamlit as st
import re
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download stopwords if not already present
nltk.download('stopwords')

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


# ------------------ Text Cleaning Function ------------------
def clean_text(text):
    text = text.lower()  # Lowercase
    tokens = re.findall(r'\b\w+\b', text)  # Tokenize
    tokens = [word for word in tokens if word.isalnum()]  # Keep alphanumeric
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [ps.stem(word) for word in tokens]  # Stemming
    return " ".join(tokens)


# ------------------ Fallback Training ------------------
def train_fallback_model():
    # Minimal dataset just for demo – replace with your dataset later
    messages = [
        "Free entry! Win a prize now",
        "Hi, how are you?",
        "Congratulations, you won a lottery!",
        "Let’s meet for project discussion",
        "Claim your free coupon today"
    ]
    labels = [1, 0, 1, 0, 1]  # 1=spam, 0=not spam

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(messages)

    model = MultinomialNB()
    model.fit(X, labels)

    # Save trained model & vectorizer
    pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
    pickle.dump(model, open('model.pkl', 'wb'))

    return tfidf, model


# ------------------ Load Model ------------------
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))

    # Check if model is fitted
    if not hasattr(model, "classes_"):
        raise ValueError("Model not fitted")
except Exception as e:
    st.warning("⚠️ Model not found or not fitted. Training fallback model...")
    tfidf, model = train_fallback_model()
    st.success("✅ New model created and saved!")

# ------------------ Streamlit App ------------------
st.title('📧 Email/SMS Spam Detection')

input_sms = st.text_input('Enter your message:')

if input_sms:  # Only process when input is provided
    transform_sms = clean_text(input_sms)
    vector_input = tfidf.transform([transform_sms])

    try:
        result = model.predict(vector_input)[0]
        if result == 1:
            st.error('🚨 Spam Detected!')
        else:
            st.success('✅ This message is NOT spam.')
    except Exception as e:
        st.error(f"Error during prediction: {e}")
