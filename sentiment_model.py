from transformers import pipeline
import pandas as pd

def load_model():
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    return classifier


def predict_sentiment(text):

    model = load_model()

    result = model(text)

    return result[0]["label"]


def analyze_dataset(file):

    df = pd.read_csv(file)

    model = load_model()

    df["prediction"] = df["text"].apply(lambda x: model(x)[0]["label"])

    return df
