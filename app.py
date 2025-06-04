# Directory structure:
# complaint-nlp-analyzer/
# ├── app.py
# ├── nlp_logic.py
# ├── data/
# │   └── synthetic_complaints.csv (already created)
# ├── requirements.txt
# ├── README.md
# └── .streamlit/
#     └── config.toml

# ---------------------------
# app.py
# ---------------------------

import streamlit as st
import pandas as pd
from nlp_logic import cluster_complaints, plot_clusters

st.set_page_config(page_title="Complaint NLP Analyzer")
st.title("🔍 Complaint Trend Analyzer with NLP")

uploaded_file = st.file_uploader("Upload Complaint CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Complaints", df.head())

    clustered_df = cluster_complaints(df)
    st.write("### Clustered Complaints", clustered_df.head())

    st.write("### Complaint Category Clusters")
    fig_path = plot_clusters(clustered_df)
    st.image(fig_path)

    st.download_button("Download Clustered Data as CSV", clustered_df.to_csv(index=False), "clustered_complaints.csv")



