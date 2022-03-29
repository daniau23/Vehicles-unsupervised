from pickletools import float8
import streamlit as st
import pandas as pd
import numpy as np 
import warnings
from sklearn.cluster import DBSCAN,KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_score,silhouette_samples,adjusted_rand_score, normalized_mutual_info_score
import time 
warnings.filterwarnings("ignore")


def cluster_dbscan(df):
    col1,col2 = st.columns([3,5])
    with col1:
        with st.form("Done"):
            n_eps = st.slider("Selecet your eps",1.0,2.0,1.7)
            n_min_samples = st.slider("Selecet your min_samples",10,30,25)
            st.form_submit_button("Done!")
    time.sleep(0.9)
    dbscan = DBSCAN(eps=float(n_eps),min_samples=n_min_samples)
    clusters = dbscan.fit_predict(df)
    n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise_ = list(clusters).count(-1)
    st.write(f"Estimated number of clusters: {n_clusters_}")
    st.write(f"Estimated number of noise points: {n_noise_}")
    labels = dbscan.labels_

    return labels



def cluster_kmeans(df):
    col1,col2 = st.columns([3,5])
    with col1:
        with st.form('Done'):
            n_clusters = st.slider("Select the number of clusters",2,10,6)
            st.form_submit_button("Done!")
        time.sleep(0.9)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(df)
        labels = kmeans.predict(df)

        return labels


def cluster_agglomerate(df):
    col1,col2 = st.columns([3,5])
    with col1:
        with st.form('Done'):
            n_clusters = st.slider("Select the number of clusters",2,10,6)
            st.form_submit_button("Done!")
        time.sleep(0.9)
        ac = AgglomerativeClustering(n_clusters=n_clusters)
        labels = ac.fit_predict(df)

        return labels