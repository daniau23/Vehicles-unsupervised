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
import time 
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


# definition for sidebar
sidebar = st.sidebar

model_click = sidebar.checkbox("The Model",key='modeling')

# df_cleaned_eda

# # [ 'avg_cost_price', 'vehicle_age', 
# # 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'selling_price', 
# # 'engine_log', 'max_power_log', 'seats_log', 
# # 'selling_price_log', 'avg_cost_price_log', 'vehicle_age_log', 'km_driven_log', 'mileage_log',
# #  'car_name', 'brand', 'model','seller_type', 'fuel_type','transmission_type' ]
X_mix = ['seats','max_power_log','engine_log','mileage','km_driven',
                'vehicle_age_log','avg_cost_price_log','selling_price_log',
                'seller_type','fuel_type','transmission_type','model','brand']

df_new = df_cleaned_eda[X_mix]

if model_click:
    st.subheader("Welcome to the model")
    def user_input_features():
        with sidebar.form('user-inputs'):
            with st.expander('User Inputs'):
                n_samples = st.slider("How many samples do you want",min_value=1,max_value=13474,value=6392)
                n_random = st.slider("Choose your random state",min_value=1,max_value=50,value=42)
                df_user = df_new.sample(n=n_samples,random_state=n_random)
                # df_user =  df_new

            #     avg_cost_price = st.number_input("Average Cost Price",min_value=0,max_value=30)
            #     vehicle_age = st.number_input("Vehicle Age",min_value=1,max_value=40) 
            #     km_driven = st.number_input("Km Driven", min_value=100,max_value=500)
            #     mileage = st.number_input("Milage",min_value=5,max_value=50)
            #     engine = st.number_input("Engine rating", min_value=700,max_value=5000)
            #     max_power = st.number_input("Max power rating", min_value=30,max_value=300) 
            #     seats = st.number_input("No. of Seats",min_value=1,max_value=10) 
            #     selling_price = st.number_input("Selling price",min_value=0,max_value=30)
            #     brand_type = st.selectbox('Brand type',('Maruti','Hyundai','Ford','Renault','Toyota','Volkswagen','Honda',
            #                                         'Mahindra','Datsun','Tata','Kia','MG','Isuzu','Skoda','Nissan','Jeep'))
            #     model_type = st.selectbox("Model type",('Alto','Grand','i20','Ecosport','Wagon R','i10','Venue','Swift','Verna',
            #                                         'Duster','Ciaz','Innova','Baleno','SwiftDzire','Vento','Creta','City',
            #                                         'Bolero','KWID','Amaze','Santro','XUV500','KUV100','Ignis','RediGO',
            #                                         'Scorpio','Marazzo','Aspire','Figo','Vitara','Tiago','Polo','Seltos',
            #                                         'Celerio','GO','KUV','Jazz','Tigor','Ertiga','Safari','Thar','Eeco',
            #                                         'Hector','Civic','D-Max','Rapid','Freestyle','Nexon','XUV300','DzireVXI',
            #                                         'WR-V','XL6','Triber','Elantra','Yaris','S-Presso','DzireLXI','Aura',
            #                                         'Kicks','Harrier','Compass','redi-GO','Glanza','DzireZXI','Altroz',
            #                                         'Tucson'))

            #     seller_type = st.radio("Seller type",('Individual','Dealer','Trustmark Dealer'))
            #     fuel_type = st.radio("Fuel type",('Petrol','Diesel','CNG','LPG'))
            #     transmission_type = st.radio("Transmission type",('Manual','Automatic'))



            st.form_submit_button()

           

            # data = {
            #     # Log inputs 
            # "avg_cost_price": np.log(avg_cost_price+1),
            # "vehicle_age:": np.log(vehicle_age+1),
            # "selling_price": np.log(selling_price+1),
            # "engine": np.log(engine+1), # End of Log inputs
            # "max_power": np.log(max_power+1),
            # "km_driven": km_driven,
            # "mileage": mileage,
            # "seats": seats,
            # "brand": brand_type,
            # "model": model_type,
            # "seller_type": seller_type,
            # "fuel_type": fuel_type,
            # "transmission_type": transmission_type
            # }

            # features = pd.DataFrame(data, index=[0])

            features = df_user
           
            # # Taking the log of chosen inputs
            # log_inputs = ['seats','max_power_log','engine_log','mileage','km_driven','vehicle_age_log','avg_cost_price_log','selling_price_log']
            # for value in log_inputs:
            #     logged = (f'{value}_log')
            #     features[logged] = features[value].apply(lambda x: np.log(x+1))

            return features

    input_df = user_input_features()
    
    input_df
#     # df_cleaned_eda
    
#     # # Ordinal features
#     # encode = ["brand", "model","seller_type","fuel_type","transmission_type"]
    

#     # # for col in encode:
#     # #     dummy = pd.get_dummies(df_cleaned_eda[col], prefix=col)
#     # #     df_unsupervised = pd.concat([input_df,dummy], axis=1)
        
#     # #     # Delete the column because the dummies are used
#     # #     del df_unsupervised[col]
#     # # dummy
#     # # df_unsupervised = df_unsupervised[:1] # Selects only the first row (the user input data)
    
#     # # print(df_unsupervised.columns.tolist())

    df_supervised = pd.get_dummies(input_df)
    # df_supervised

    st.write("Normalisation in progress")
    scaler = RobustScaler()
    
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)
        if percent_complete == 50:
            st.write("Almost there")
        my_bar.progress(percent_complete + 1)
    
    time.sleep(0.01)
    st.success("Fully scaled")
    df_supervised_scaled = scaler.fit_transform(df_supervised)
    df_supervised_scaled

   
    radio = st.radio("Want to use PCA [Principal component Analysis]?", ("Yes","No"))
    col1,col2 = st.columns((1,1))
    form_pca = st.form('form-pca')
    # with col1:
    if radio == "Yes":
        # with form_pca:
            
        n_components_percent = st.slider("Percentage components", min_value=0,max_value=100,value=90)
        time.sleep(0.9)
        n_components= n_components_percent/100
        pca = PCA(n_components=n_components).fit(df_supervised_scaled)
        
        
        my_bar2 = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar2.progress(percent_complete + 1)
            if percent_complete == 60:
                st.write("Almost done")
            my_bar2.progress(percent_complete + 1)
            # st.form_submit_button()


        st.success("You just applied a Feature engineering algorithm")
        pca_data = pca.transform(df_supervised_scaled)

        # plotting the pca components
        radio = st.radio("Do you wish to visualise the PCA explained Variance ratio?",("Yes","No"))
        if radio == "Yes":
            st.write(f"Features are feed in: {pca.n_features_in_}")
            st.write(f"PCA components are: {pca.n_components_}")
            # pca_plotting = pca_plots(pca)
            pca_plots(pca)
            st.write(f"Using {pca.n_components_} components explains {n_components_percent}% of the data! That's awesome")
            time.sleep(0.9)
            st.balloons()
            st.snow()
            # st.pyplot(pca_plotting)
            # pca_plots(pca)
    else:
        pass