# ---------------------------
# nlp_logic.py
# ---------------------------

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import uuid


def cluster_complaints(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Complaint_Text'])

    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    return df


def plot_clusters(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Cluster', data=df, palette="Set2")
    plt.title("Number of Complaints per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    fig_path = "/mnt/data/cluster_plot.png"
    plt.tight_layout()
    plt.savefig(fig_path)
    return fig_path
