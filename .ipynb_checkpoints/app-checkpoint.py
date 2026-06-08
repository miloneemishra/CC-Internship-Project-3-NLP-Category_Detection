import streamlit as st
import joblib
from preprocessing import preprocess

st.set_page_config(page_title="Incident Classifier", page_icon="🚚", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #f7f5f2;
    color: #1a1a1a;
}

.stTextArea textarea {
    background-color: #ffffff !important;
    border: 1.5px solid #d4d0c8 !important;
    border-radius: 6px !important;
    color: #1a1a1a !important;
    font-size: 0.95rem !important;
    padding: 0.8rem !important;
}

.stTextArea textarea:focus {
    border-color: #1a1a1a !important;
    box-shadow: none !important;
}

.stButton > button {
    background-color: #1a1a1a;
    color: #f7f5f2;
    font-weight: 600;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 2rem;
    width: 100%;
}

.stButton > button:hover {
    background-color: #333333;
    color: #f7f5f2;
}
</style>
""", unsafe_allow_html=True)

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

st.title("Supply Chain Incident Classifier")
st.write("Enter an incident description below to classify it and get a confidence score.")

text_input = st.text_area("Incident Description", height=150)

if st.button("Predict"):
    if text_input.strip() == "":
        st.warning("Please enter an incident description.")
    else:
        cleaned = preprocess(text_input)
        vectorized = tfidf.transform([cleaned])
        category = model.predict(vectorized)[0]
        confidence = round(model.predict_proba(vectorized).max() * 100, 2)

        st.success(f"Predicted Category: {category}")
        st.info(f"Confidence Score: {confidence}%")