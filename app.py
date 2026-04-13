import streamlit as st
import joblib
import re, string
import pandas as pd
import numpy as np

# Load saved models
vectorization = joblib.load("vectorizer.pkl")
LR = joblib.load("logistic_model.pkl")
DT = joblib.load("decisiontree_model.pkl")
GC = joblib.load("gradientboost_model.pkl")
RF = joblib.load("randomforest_model.pkl")

# Text cleaning function
def word(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?/\]', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

st.title("Fake News Prediction App")

st.write("Enter or paste any news article text below to verify whether it's **fake or real**.")

user_input = st.text_area(" Enter News Article Here:")

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some news text!")
    else:
        cleaned_text = word(user_input)
        transformed = vectorization.transform([cleaned_text])

        # Get predictions from each model
        preds = [
            LR.predict(transformed)[0],
            DT.predict(transformed)[0],
            GC.predict(transformed)[0],
            RF.predict(transformed)[0]
        ]

        # Calculate percentage of True and Fake
        true_percent = (np.sum(preds) / len(preds)) * 100
        fake_percent = 100 - true_percent

        st.subheader("🔍 Overall Prediction")
        if true_percent >= 50:
            st.success(f"TRUE NEWS")
        else:
            st.error(f"FAKE NEWS")
