import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Global Development Clustering", layout="wide")

st.title("🌍 Global Development Measurement - Clustering App")

# File Upload
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)
    st.subheader("Raw Data")
    st.dataframe(df.head())

    # Drop Country Column if exists
    df_model = df.drop(columns=['Country'], errors='ignore')

    # Convert currency columns
    currency_cols = ['GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound']

    for col in currency_cols:
        if col in df_model.columns:
            df_model[col] = (
                df_model[col]
                .astype(str)
                .str.replace('$', '', regex=False)
                .str.replace(',', '', regex=False)
                .replace('nan', np.nan)
                .astype(float)
            )

    # Convert % columns
    percent_cols = [
        col for col in df_model.columns
        if df_model[col].astype(str).str.contains('%').any()
    ]

    for col in percent_cols:
        df_model[col] = (
            df_model[col]
            .astype(str)
            .str.replace('%', '', regex=False)
            .replace('nan', np.nan)
            .astype(float) / 100
        )

    # Handle missing values
    num_cols = df_model.select_dtypes(include=np.number).columns
    df_model[num_cols] = df_model[num_cols].fillna(df_model[num_cols].mean())

    # Select only numeric
    df_num = df_model.select_dtypes(include=np.number)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num)

    st.subheader("Choose Clustering Algorithm")

    algo = st.selectbox(
        "Select Algorithm",
        ["KMeans", "Hierarchical", "DBSCAN", "Gaussian Mixture"]
    )

    n_clusters = st.slider("Number of Clusters", 2, 6, 3)

    if st.button("Run Clustering"):

        if algo == "KMeans":
            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(X_scaled)

        elif algo == "Hierarchical":
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(X_scaled)

        elif algo == "DBSCAN":
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(X_scaled)

        elif algo == "Gaussian Mixture":
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = model.fit_predict(X_scaled)

        df_num["Cluster"] = labels

        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        st.subheader("Cluster Visualization (PCA Reduced)")

        fig, ax = plt.subplots(figsize=(8,6))
        scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=labels)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(f"{algo} Clustering")
        st.pyplot(fig)

        # Silhouette Score
        try:
            if algo == "DBSCAN":
                mask = labels != -1
                score = silhouette_score(X_scaled[mask], labels[mask])
            else:
                score = silhouette_score(X_scaled, labels)

            st.success(f"Silhouette Score: {round(score, 3)}")
        except:
            st.warning("Silhouette score cannot be calculated for this configuration.")

        st.subheader("Clustered Data")
        st.dataframe(df_num.head())

else:
    st.info("Please upload the dataset to begin.")
