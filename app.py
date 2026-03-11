import streamlit as st
from transformers import pipeline

st.title("AI Sentiment Analyzer")

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

model = load_model()

text = st.text_input("Enter text")

if text:
    result = model(text)[0]

    if result["label"] == "POSITIVE":
        st.success(f"Positive Sentiment 😊 (Confidence: {result['score']:.2f})")
    else:
        st.error(f"Negative Sentiment 😞 (Confidence: {result['score']:.2f})")
