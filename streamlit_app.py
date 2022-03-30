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

from streamlit.proto.Markdown_pb2 import Markdown
from make_plots import *
import time 
warnings.filterwarnings("ignore")
from clustering import *

# definition for sidebar
sidebar = st.sidebar

df_uncleaned = pd.read_csv("cardekho_updated.csv.zip")

# from data cleaning
df_cleaned = pd.read_csv("cleaned-data.csv")
df_cleaned_eda = pd.read_csv("cleaned-data-eda.csv")

st.title('Unsupervised Learning App')
sidebar.subheader('An App by Daniel Ihenacho')
sidebar.write('''This app  makes use of a vehicle dataset 
for unsupervised learning purposes ''')


# About me 
about_me = sidebar.checkbox("About Me",key='ME')
if about_me:
    st.markdown('## Daniel Chiebuka Ihenacho', unsafe_allow_html=True)
    st.markdown('### Summary', unsafe_allow_html=True)
    st.info('''
    - Aspiring Data scientist, Analyst and Teacher.
    - TEFL (Teaching English as a Foreign Language) certified.
    - Mandarin proficient; HSK 4 (Hanyu Shuiping Kaoshi) certified.
    - Skilled, disciplined & a team player
    - Motivated by problems/challenges and intrigued in finding solutions.
    I am thrilled and look forward to hearing from you about potential vacancies and opportunities. Please do feel free to message me.
    ''')
    # Custom function for printing text
    def txt(a, b):
        col1, col2 = st.columns([4,1])
        with col1:
            st.markdown(a)
        with col2:
            st.markdown(b)

    def txt2(a, b):
        col1, col2 = st.columns([1,4])
        with col1:
            st.markdown(f'`{a}`')
        with col2:
            st.markdown(b)

    def txt3(a, b):
        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown(a)
        with col2:
            st.markdown(b)
    
    def txt4(a, b, c):
        col1, col2, col3 = st.columns([1.5,2,2])
        with col1:
            st.markdown(f'`{a}`')
        with col2:
            st.markdown(b)
        with col3:
            st.markdown(c)

    st.markdown('''
    ### Education
    ''')

    txt("**M.Eng.** (Software Engineering), *Liaoning Technical University*, China",
    "2019-2022")
    st.markdown('''
    - GPA: ``
    - Research thesis entitled `Design of Gas Cyclone Using Hybrid Particle Swarm Optimization Algorithm`.
    ''')

    txt("**B.Eng.** (Electrical & Electronics Engineering), *Landmark University*, Nigeria",
    "2014-2019")
    st.markdown('''
    - GPA: `4.77`
    - Thesis entitled `Design of Pyramidal Horn Antenna for WLAN Communication in Landmark Farm at 5.8GHz`
    - Graduated with First Class Honors.
    ''')

    st.markdown('''
    ### Work Experience
    ''')

    txt("**Industrial Training (IT) Student**,*Nigerian Communications Satellite Limited (Nigcomsat)*, Nigeria",
    "01-01-2018 To 30-06-2018")
    st.markdown('''
    - Contributed to the successful NNPC (Nigerian National Petroleum Corporation) elections by setting up computers and ensuring a safe zone.
    - Collaborated in a team/group work to design and implement a temperature cooling system for the department of Innovation & Development department (I&D)
    ''')

    txt("**Intern**, *Hotspot Network Limited*, Nigeria",
    "01-07-2017 to 31-07-2017")
    st.markdown("""
    - Contributed in Radio Planning and Deployment of Rural telephony BTS project under the auspices of Universal Service Provision Fund (USPF).
    - Praised for effective writing of executive summaries for the company as an intern in the company, which helped in business negotiations.
    """)

    st.markdown('''
    ### Skills
    ''')
    txt3("Programming", "`Python`")
    txt3("Data processing/wrangling", "`PostgreSQL`, `pandas`, `numpy`")
    txt3("Data visualization", "`matplotlib`, `seaborn`, `plotly`")
    txt3("Data analysis", "`Excel`,`Tableau`")
    txt3("Machine Learning", "`scikit-learn`")
    txt3("Deep Learning", '``')
    txt3("Model deployment", "`streamlit`")

    #####################
    st.markdown('''
    ### Social Media
    ''')
    txt2("LinkedIn", "http://www.linkedin.com/in/daniel-ihenacho-637467223")
    txt2("Indeed", "https://my.indeed.com/p/danielchiebukai-hz1szfb")
    txt2("GitHub", "https://github.com/daniau23")
    txt2("Kaggle", "https://www.kaggle.com/danielihenacho")
    txt2("ORCID", "https://orcid.org/0000-0003-3043-9201")
    
# display data
show_data = sidebar.checkbox("See the raw data and data description?",key='data')
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
plot_data = sidebar.checkbox("Do you want to visualise the data?", key="plot_data")
if plot_data:
    chart_type = sidebar.selectbox("Choose your chart type", plot_types)
   
    
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
correct_data = sidebar.checkbox("Skewness correction types", key="correct_data")
if correct_data:
    st.write("""Levels of skewness\n
    1. (-0.5,0.5) = lowly skewed\n
    2. (-1,0-0.5) U (0.5,1) = Moderately skewed\n
    3. (-1 & beyond ) U (1 & beyond) = Highly skewed""")

    form = sidebar.form("log-p")
    menu = form.selectbox('Skewness display',options=("Yes","No"))
    if menu == 'Yes':
        comparsion_box(menu,df_cleaned_eda)
    form.form_submit_button()




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
                n_samples = st.slider("How many samples do you want",min_value=300,max_value=13474,value=6392)
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
    
    # @st.cache(suppress_st_warning=True)
    def feature_eng(df):
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
            while radio == "Yes":    
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

                # Here
                # plotting the pca components
                radio2 = st.radio("Do you wish to visualise the PCA explained Variance ratio?",("Yes","No"))
                if radio2 == "Yes":
                    while radio2 == "Yes":
                        st.write(f"Features fed in were: {pca.n_features_in_}")
                        st.write(f"PCA components are: {pca.n_components_}")
                        # pca_plotting = pca_plots(pca)
                        pca_plots(pca)
                        st.warning(f"Using {pca.n_components_} components explains {n_components_percent}% of the data! That's awesome")
                        
                        # returns the pca dataframe and the components after selecting
                        # the number of components needed for the PCA
                        return df_supervised, pca_data, pca.n_components_

                elif radio2 == "No":
                    st.error(f"No Visualisation was used. Please try it")
                    return df_supervised, pca_data, pca.n_components_
                    # end here
        else:
            nothing = st.error("No PCA was applied. Please try it")
            return nothing

    # supervised contains my dummies data for the
    # supervsied classification model that I will use
    # try:
    supervised, pca_data, components_ = feature_eng(input_df)
    time.sleep(0.9)
    print(components_)
    # st.balloons()
    # st.snow()
    
    def clusters(df):

        cluster_click = sidebar.selectbox("Clustering",("DBSCAN","K-Means","Agglomerate",))

        if cluster_click == "DBSCAN":
            st.write("DBSCAN")
            
            # Returns the dbscan labels 
            dbscan_labels = cluster_dbscan(pca_data)
            final_labels = dbscan_labels
            col1,col2 = st.columns([5,10])
            if components_ >= 3:
                with col1:
                    with st.form("Visualise"):
                        componets_1 = st.number_input("Choose your component 1",0,components_-1,0)
                        componets_2 = st.number_input("Choose your component 2",0,components_-1,1)
                        componets_3 = st.number_input("Choose your component 3",0,components_-1,2)
                        st.form_submit_button("Visualise")    
                with col2:
                    time.sleep(0.9)
                    cluster_plots(pca_data,components_,dbscan_labels,componets_1,componets_2,componets_3)
            else:
                st.error("Number of components msust be greater than 2 to visualise clusters")

        elif cluster_click == "K-Means":
            st.write("K-Means")
            kmeans_labels = cluster_kmeans(pca_data)
            final_labels = kmeans_labels
            col1,col2 = st.columns([5,10])
            with col1:
                with st.form("Visualise"):
                    componets_1 = st.number_input("Choose your component 1",0,components_-1,0)
                    componets_2 = st.number_input("Choose your component 2",0,components_-1,1)
                    componets_3 = st.number_input("Choose your component 3",0,components_-1,2)
                    st.form_submit_button("Visualise")    
            with col2:
                time.sleep(0.9)
                cluster_plots(pca_data,components_,kmeans_labels,componets_1,componets_2,componets_3)

        elif cluster_click == "Agglomerate":
            st.write("Agglomerate")
            agglomerate_labels = cluster_agglomerate(pca_data)
            final_labels = agglomerate_labels
            col1,col2 = st.columns([5,10])
            with col1:
                with st.form("Visualise"):
                    componets_1 = st.number_input("Choose your component 1",0,components_-1,0)
                    componets_2 = st.number_input("Choose your component 2",0,components_-1,1)
                    componets_3 = st.number_input("Choose your component 3",0,components_-1,2)
                    st.form_submit_button("Visualise")    
            with col2:
                time.sleep(0.9)
                cluster_plots(pca_data,components_,agglomerate_labels,componets_1,componets_2,componets_3)

        # returning the labels of each clustering algorithm
        return final_labels

    # Dsiplays the PCA visualisation and returns the clusters labels
    clustering = clusters(pca_data)
    # print(clustering)
    # print(isinstance(clustering, list))
    # print(type(clustering) == type(list))
    # print("Labels: {}".format(clustering))

    # print(supervised.columns)

    # put this in another python script for usage
    # Use of supervised algorithm
    df_dummies1_pca_dbscan = pd.concat([supervised.reset_index(drop = True), pd.DataFrame(pca_data)], axis=1)
    
    
    # # '''No need for this part'''
    # # df_dummies1_pca_dbscan.columns.values[-16:] = ['component_1',
    # #                                             'component_2',
    # #                                             'component_3',
    # #                                             'component_4',
    # #                                             'component_5',
    # #                                             'component_6',
    # #                                             'component_7',
    # #                                             'component_8',
    # #                                             'component_9',
    # #                                             'component_10',
    # #                                             'component_11',
    # #                                             'component_12',
    # #                                             'component_13',
    # #                                             'component_14',
    # #                                             'component_15',
    # #                                             'component_16',]



    df_dummies1_pca_dbscan['segement_pca'] = clustering
    i = df_dummies1_pca_dbscan.loc[df_dummies1_pca_dbscan['segement_pca'] == -1,'segement_pca'].index
    # print(i)
    df_dummies1_pca_dbscan.drop(i, inplace=True)

    # Getting the range of components needed 
    comp = list(range(components_))
    df_knn_set = df_dummies1_pca_dbscan[comp]
    df_knn_set['segement_pca'] = df_dummies1_pca_dbscan['segement_pca']

    X = df_knn_set.drop(['segement_pca'],axis=1)
    y = df_knn_set['segement_pca']

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=.2,train_size=.8, stratify=y)
    knn = KNeighborsClassifier(n_neighbors=5,p=1,weights='uniform')
    knn.fit(X_train,y_train)

    prediction =  knn.predict(X_test)
    
    
    print(list(range(components_)))
    print(df_knn_set.columns.tolist())

    # Model predictions pre-trained model
    # load_vehicle = pickle.load(open('vehicle.pkl', 'rb'))
    # prediction = load_vehicle.predict(X)

    
    # Comparing Observed to predicted
    st.subheader("Predicted vs Observed")
    df_prd_tst = pd.DataFrame({'Predicted':prediction.astype('int8'), 'Observed':y_test})
    # df_prd_tst.to_csv('prediction.csv')
    st.write(df_prd_tst)
    # except Exception:
    #     st.error("PCA was not selected and no model can be applied")


pre_model_click = sidebar.checkbox("Pre-trained model",key='pre-modeling')

loggers = ['max_power','engine','vehicle_age','avg_cost_price','selling_price']

if pre_model_click:
    st.subheader("This is still under development")
    # uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    # if uploaded_file is not None:
    #     input_file = pd.read_csv(uploaded_file)
    #     input_file[loggers] = input_file[loggers].apply(lambda x: np.log(x+1))
    #     dummies_input = pd.get_dummies(input_file)
       
    #     st.write(dummies_input)
    #     scaler = RobustScaler()

    #     dummies_input_scaled = scaler.fit_transform(dummies_input)

    #     st.write(dummies_input_scaled)
        
    #     # PCA
    #     pca_data_file = PCA(n_components=.90)
    #     pca_data_file.fit(dummies_input_scaled)
    #     pca_data_file_ = pca_data_file.transform(dummies_input_scaled)

    #     st.write(pca_data_file.n_components_)

    #     # Clustering
    #     dbscan = DBSCAN(eps=1.7,min_samples=25)
    #     clusters = dbscan.fit_predict(pca_data_file_)
    #     n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
    #     n_noise_ = list(clusters).count(-1)
    #     st.write(f"Estimated number of clusters: {n_clusters_}")
    #     st.write(f"Estimated number of noise points: {n_noise_}")
    #     labels = dbscan.labels_


    #     # dummies_input_scaled

    #     df_dummies1_pca_dbscan = pd.concat([dummies_input_scaled.reset_index(drop = True),pd.DataFrame(pca_data_file_)], axis=1)
    #     df_dummies1_pca_dbscan.columns.values[-16:] = ['component_1',
    #                                                     'component_2',
    #                                                     'component_3',
    #                                                     'component_4',
    #                                                     'component_5',
    #                                                     'component_6',
    #                                                     'component_7',
    #                                                     'component_8',
    #                                                     'component_9',
    #                                                     'component_10',
    #                                                     'component_11',
    #                                                     'component_12',
    #                                                     'component_13',
    #                                                     'component_14',
    #                                                     'component_15',
    #                                                     'component_16',]

    #     df_dummies1_pca_dbscan['segement_dbscan_pca'] = labels

    #     i = df_dummies1_pca_dbscan.loc[df_dummies1_pca_dbscan['segment'] == 'noise','segment'].index
    #     df_dummies1_pca_dbscan.drop(i, inplace=True)

    #     df_knn_set = df_dummies1_pca_dbscan[['component_1',
    #                                             'component_2',
    #                                             'component_3',
    #                                             'component_4',
    #                                             'component_5',
    #                                             'component_6',
    #                                             'component_7',
    #                                             'component_8',
    #                                             'component_9',
    #                                             'component_10',
    #                                             'component_11',
    #                                             'component_12',
    #                                             'component_13',
    #                                             'component_14',
    #                                             'component_15',
    #                                             'component_16',
    #                                             'segement_dbscan_pca']]

    #     X = df_knn_set.drop(['segement_dbscan_pca'],axis=1)
    #     y = df_knn_set['segement_dbscan_pca']

    #     load_vehicle = pickle.load(open('vehicle.pkl', 'rb'))
    #     prediction = load_vehicle.predict(X)
        
    #     df_prd_tst_file = pd.DataFrame({'Predicted':prediction.astype('int8'), 'Observed':y})
    #     # df_prd_tst.to_csv('prediction.csv')
    #     st.write(df_prd_tst_file)

    #     # data = {
    #     #     "predicted": prediction
    #     # }
    #     # predicted = pd.DataFrame(data)
    #     # st.write(predicted)

    # else:
    #     st.write('Awaiting CSV file to be uploaded.')