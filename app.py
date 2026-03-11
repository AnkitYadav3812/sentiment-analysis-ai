import streamlit as st
from transformers import pipeline

st.title("AI Sentiment Analyzer")

@st.cache_resource
def load_model():
    model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    return model

model = load_model()

text = st.text_input("Enter text")

if text:
    result = model(text)
    st.write(result)
