import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentiment_distribution(df):

    sns.countplot(x="prediction", data=df)

    plt.title("Sentiment Distribution")

    plt.xlabel("Sentiment")

    plt.ylabel("Count")

    plt.show()
