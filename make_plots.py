import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# [ 'avg_cost_price', 'vehicle_age', 
# 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'selling_price', 
# 'engine_log', 'max_power_log', 'seats_log', 
# 'selling_price_log', 'avg_cost_price_log', 'vehicle_age_log', 'km_driven_log', 'mileage_log',
#  'car_name', 'brand', 'model','seller_type', 'fuel_type','transmission_type' ]


def sns_plot(chart_type, df):
    '''Plot types are 
    Scatter plot,
    Histogram,
    Box plot,
    Boxen plot,
    Count plot,
    Bar plot,
    Line plot
    '''
    
    sns.set_style(style='dark')
    
    # Scatter plot
    fig,ax = plt.subplots(figsize=(10,10))
    with st.sidebar.form("select-form"):
        if chart_type == "Scatter plot":
            selected_x_var = st.selectbox('''What do want the x variable to
                        be?''',
                    ['avg_cost_price', 
                    'vehicle_age', 'km_driven', 
                    'mileage', 'engine', 
                    'max_power', 'seats', 
                    'selling_price'])
                        # y variable
            selected_y_var = st.selectbox('What about the y?',
                    ['avg_cost_price', 
                    'vehicle_age', 'km_driven', 
                    'mileage', 'engine', 
                    'max_power', 'seats', 
                    'selling_price'])
            
            sns.scatterplot(data=df,x=selected_x_var,y=selected_y_var)
            ax.set_title(f"Scatter plot of {selected_x_var} vs {selected_y_var}")
    
    
    # # Line plot
    #     elif chart_type == "Line plot":
    #         selected_x_var = st.selectbox('''What do want the x variable to
    #                     be?''',
    #                 ['avg_cost_price', 
    #                 'vehicle_age', 'km_driven', 
    #                 'mileage', 'engine', 
    #                 'max_power', 'seats', 
    #                 'selling_price'])

    #         # y variable
    #         selected_y_var = st.selectbox('What about the y?',
    #                 ['avg_cost_price', 
    #                 'vehicle_age', 'km_driven', 
    #                 'mileage', 'engine', 
    #                 'max_power', 'seats', 
    #                 'selling_price'])
            
    #         sns.lineplot(data=df,x=selected_x_var,y=selected_y_var)
    #         ax.set_title(f"Line plot of {selected_x_var} vs {selected_y_var}")
            
        
        # Histogram plot
        elif chart_type =="Histogram":
            selected_x_var = st.selectbox('''What do want the x variable to
                        be?''',
                        ['avg_cost_price', 
                    'vehicle_age', 'km_driven', 
                    'mileage', 'engine', 
                    'max_power', 'seats', 
                    'selling_price'])

            sns.histplot(data=df,x=selected_x_var,kde=True)
            # ax.set_title(f"Histogram plot of {selected_x_var}")
            ax.set_title(f"Skewness of {selected_x_var}: {np.around(df[selected_x_var].skew(axis=0),2)}")


        # Categorical plot 1: Box
        elif chart_type == "Box plot":
            selected_y_var = st.selectbox('''What do want the y variable to
                        be?''',
                        ['brand','seller_type', 'fuel_type','transmission_type'])
                        #  'model',

            selected_x_var = 'selling_price'
            sns.boxplot(x=selected_x_var,y=selected_y_var, data=df,palette='nipy_spectral',hue=selected_y_var)
            ax.set_title(f"The Box plot of {selected_y_var}")
        
        # Categorical plot 2: Boxen
        # elif chart_type == "Boxen plot":
        #     selected_y_var = st.selectbox('''What do want the y variable to
        #                 be?''',
        #                 ['brand', 'seller_type', 'fuel_type','transmission_type'])

        #     selected_x_var = 'selling_price'
        #     sns.boxenplot(x=selected_x_var,y=selected_y_var, data=df,palette='nipy_spectral',hue=selected_y_var)
        #     ax.set_title(f"The Boxen plot of {selected_y_var}")
        
        # Categorical plot 3: Count
        elif chart_type == "Count plot":
            selected_x_var = st.selectbox('''What do want the x variable to
                        be?''',
                        [ 'seller_type', 'fuel_type','transmission_type'])
                        #   'car_name', 'brand', 'model'

            sns.countplot(data=df,x=selected_x_var,hue=selected_x_var)
            ax.set_title(f"The Count plot of {selected_x_var}")

        # Categorical plot 4: Bar 
        # elif chart_type == "Bar plot":
        #     selected_x_var = st.selectbox('''What do want the x variable to
        #                 be?''',
        #                 ['seller_type', 'fuel_type','transmission_type'])

        #     selected_y_var = 'selling_price'
        #     sns.barplot(x=selected_x_var,y=selected_y_var, data=df,palette='nipy_spectral',hue=selected_x_var)
        #     ax.set_title(f"The Bar plot of {selected_x_var}")

        
        st.form_submit_button()
            

    return fig







'''Make a function to be integrated in the sidebar for Plotly to display bar plot'''
def plotly_plot(chart_type, df):
    with st.sidebar.form("select-form"):
        if chart_type == "Bar plot":
                    selected_x_var = st.selectbox('''What do want the x variable to
                                be?''',
                                ['seller_type', 'fuel_type','transmission_type'])

                    selected_y_var = 'selling_price'
                    fig = px.bar(df,x= selected_x_var
                            ,y= selected_y_var
                            ,color='brand'
                            ,barmode='group'
                            # ,title= f"Bar chart of {x} group by {}"
                            ,height=600,color_discrete_sequence=px.colors.qualitative.Vivid)

        elif chart_type == "Line plot":
            selected_x_var = st.selectbox('''What do want the x variable to
                        be?''',
                    ['avg_cost_price', 
                    'vehicle_age', 'km_driven', 
                    'mileage', 'engine', 
                    'max_power', 'seats', 
                    'selling_price'])

            # y variable
            selected_y_var = st.selectbox('What about the y?',
                    ['avg_cost_price', 
                    'vehicle_age', 'km_driven', 
                    'mileage', 'engine', 
                    'max_power', 'seats', 
                    'selling_price'])
            # Using Plotly for plotting
            fig = px.line(df.sort_values(selected_x_var), x=selected_x_var,
            y=selected_y_var,color='transmission_type') #,markers=True
            # fig.show()
        st.form_submit_button()
    return fig







# Log vs normal plot
def comparsion(df_1):
    col1,col2=st.columns([10,10])
    with st.container():
        with col1:
            fig1,ax1 = plt.subplots()
            x_var = 'price'
            ax1 = sns.boxplot(df_1[x_var],palette='nipy_spectral')
            ax1.set_title(f"Skewness: {np.around(df_1[x_var].skew(),3)}")
            st.pyplot(fig1)

        with col2:
            fig2,ax2 = plt.subplots()
            x_var = 'price_log'
            ax2 = sns.boxplot(df_1[x_var],palette='nipy_spectral')
            ax2.set_title(f"Skewness: {np.around(df_1[x_var].skew(),3)}")
            st.pyplot(fig2)

        with col1:
            fig3,ax3 = plt.subplots()
            x_var = 'area'
            ax3 = sns.boxplot(df_1[x_var],palette='nipy_spectral')
            ax3.set_title(f"Skewness: {np.around(df_1[x_var].skew(),3)}")
            st.pyplot(fig3)

        with col2:
            fig4,ax4 = plt.subplots()
            x_var = 'area_log'
            ax4 = sns.boxplot(df_1[x_var],palette='nipy_spectral')
            ax4.set_title(f"Skewness: {np.around(df_1[x_var].skew(),3)}")
            st.pyplot(fig4)






