import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as gos
import plotly.figure_factory as ff
import seaborn as sns
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score,silhouette_samples,adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import GridSearchCV,train_test_split,KFold,StratifiedShuffleSplit
from sklearn.pipeline import  Pipeline
from sklearn.neighbors import KNeighborsClassifier
import pickle
from make_plots import *
warnings.filterwarnings("ignore")


df_uncleaned = pd.read_csv("cardekho_updated.csv.zip")

# from data cleaning
df_cleaned = pd.read_csv("cleaned-data.csv")
df_cleaned_eda = pd.read_csv("cleaned-data-eda.csv")

st.title('Unsupervised Learning App')
st.sidebar.subheader('An App by Daniel Ihenacho')
st.sidebar.write(('''This app  makes use of a vehicle dataset 
for unsupervised learning purposes '''))


# display data
show_data = st.sidebar.checkbox("See the raw data and data description?",key='data')
with st.container():
    if show_data:
        st.write("This is the uncleaned data")
        df_uncleaned
        st.write("This is the cleaned data")
        df_cleaned



# Performing EDA
plot_types= ["Scatter plot",
    "Line plot",
    "Histogram",
    "Box plot",
    "Heat map",
    "Count plot",
    "Bar plot"]

# User choose type
plot_data = st.sidebar.checkbox("Do you want to visualise the data?", key="plot_data")
if plot_data:
    chart_type = st.sidebar.selectbox("Choose your chart type", plot_types)

    def show_plot(chart_type,data):
        if chart_type == "Bar plot":
            plot = plotly_plot(chart_type,data)
            col_1,col_2 = st.columns((10,3))
            with col_1:
                st.plotly_chart(plot)
                plt.tight_layout()

        elif chart_type == "Line plot":
            plot = plotly_plot(chart_type,data)
            col_1,col_2 = st.columns((10,3))
            with col_1:
                st.plotly_chart(plot)
                plt.tight_layout()

        elif chart_type == "Heat map":
            data = df_cleaned
            plot = sns_plot(chart_type,data)
            col_1,col_2 = st.columns((10,3))
            with col_1:
                st.pyplot(plot)
                plt.tight_layout()

        else:
            # This calls the sns_plot method in make_plots
            # Which returns a figure i.e fig
            plot = sns_plot(chart_type,data)
            col_1,col_2 = st.columns((10,3))
            with col_1:
                st.pyplot(plot)
                plt.tight_layout()
    

    show_plot(chart_type,df_cleaned_eda)


# Showing comparison of skewed and corrected data
correct_data = st.sidebar.checkbox("Skewness correction types", key="correct_data")
if correct_data:
    st.write("""Levels of skewness\n
    1. (-0.5,0.5) = lowly skewed\n
    2. (-1,0-0.5) U (0.5,1) = Moderately skewed\n
    3. (-1 & beyond ) U (1 & beyond) = Highly skewed""")

    form = st.sidebar.form("log-p")
    menu = form.selectbox('Skewness display',options=("Yes","No"))
    if menu == 'Yes':
        comparsion_box(menu,df_cleaned_eda)
    form.form_submit_button()