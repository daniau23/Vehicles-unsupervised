from ast import withitem
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
key_values = {"Average Cost":'avg_cost_price', 
            "Vehicle age":'vehicle_age', 
            "Km driven":'km_driven', 
            "Milage":'mileage', "Engine":'engine', 
            "Max power":'max_power', "Seats":'seats', 
            "Selling price":'selling_price'}

key_values_cat = {"Brand":'brand',
                "Seller":'seller_type',
                "Fuel":'fuel_type',
                "Transmission":'transmission_type'}

key_values_cat_count = {
                            "Seller":'seller_type',
                            "Fuel":'fuel_type',
                            "Transmission":'transmission_type'}

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
    # key_values = {"Average Cost":'avg_cost_price', 
    #             "Vehicle age":'vehicle_age', 
    #             "Km driven":'km_driven', 
    #             "Milage":'mileage', "Engine":'engine', 
    #             "Max power":'max_power', "Seats":'seats', 
    #             "Selling price":'selling_price'}

    # key_values_cat = {"Brand":'brand',
    #                 "Seller":'seller_type',
    #                 "Fuel":'fuel_type',
    #                 "Transmission":'transmission_type'}

    # key_values_cat_count = {
    #                             "Seller":'seller_type',
    #                             "Fuel":'fuel_type',
    #                             "Transmission":'transmission_type'}

    sns.set_style(style='dark')
    # x_var = key_values.get(select_x_var)
    # Scatter plot
    fig,ax = plt.subplots(figsize=(10,10))
    with st.sidebar.form("select-form"):
        if chart_type == "Scatter plot":
            selected_x_var = st.selectbox('''What do want the x variable to
                        be?''', options=(key_values)
                    # ['avg_cost_price', 
                    # 'vehicle_age', 'km_driven', 
                    # 'mileage', 'engine', 
                    # 'max_power', 'seats', 
                    # 'selling_price']
                    )
            x_var = key_values.get(selected_x_var)
                        # y variable
            selected_y_var = st.selectbox('What about the y?',
                    options=(key_values))
            
            y_var = key_values.get(selected_y_var)

            sns.scatterplot(data=df,x=x_var,y=y_var)
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
                        options=key_values)
            x_var = key_values.get(selected_x_var)

            sns.histplot(data=df,x=x_var,kde=True)
            # ax.set_title(f"Histogram plot of {selected_x_var}")
            ax.set_title(f"Skewness of {x_var}: {np.around(df[x_var].skew(axis=0),2)}")


        # THE USE OF DICTIONARY DID NOT WORK FOR THE BOX PLOT
        # Categorical plot 1: Box
        # elif chart_type == "Box plot":
        #     selected_y_var = st.selectbox('''What do want the y variable to
        #                 be?''',
        #                 options=key_values_cat)
        #                 #  'model',
        #     y_var = key_values.get(selected_y_var)
            
        #     selected_x_var = 'selling_price'
        #     sns.boxplot(x=selected_x_var,y=y_var, data=df,palette='nipy_spectral', hue=df['brand']) # 
        #     ax.set_title(f"The Box plot of {y_var}")

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
                        options=key_values_cat_count)
                        #   'car_name', 'brand', 'model'
            x_var = key_values_cat_count.get(selected_x_var)

            sns.countplot(data=df,x=x_var,hue=x_var)
            ax.set_title(f"The Count plot of {x_var}")

        # Categorical plot 4: Bar 
        # elif chart_type == "Bar plot":
        #     selected_x_var = st.selectbox('''What do want the x variable to
        #                 be?''',
        #                 ['seller_type', 'fuel_type','transmission_type'])

        #     selected_y_var = 'selling_price'
        #     sns.barplot(x=selected_x_var,y=selected_y_var, data=df,palette='nipy_spectral',hue=selected_x_var)
        #     ax.set_title(f"The Bar plot of {selected_x_var}")
        
        # Heatmap  
        elif chart_type == "Heat map":
                sns.set_theme()
                grid_kws = {"height_ratios": (1,.05),"hspace":.3}
                fig,(ax, cbar_ax) =plt.subplots(2,figsize=(15, 9),gridspec_kw=grid_kws)
                sns.heatmap(df.corr(),
                ax=ax,
                cbar_ax=cbar_ax,
                cmap="rainbow",
                annot=True,
                vmin=-1,
                vmax=1,
                cbar_kws={"orientation":"horizontal",'label':"Christmas colorbar"},
                linewidths=1
                )

        
        st.form_submit_button()
            

    return fig







'''Make a function to be integrated in the sidebar for Plotly to display bar plot'''
def plotly_plot(chart_type, df):
    with st.sidebar.form("select-form"):
        if chart_type == "Bar plot":
                    selected_x_var = st.selectbox('''What do want the x variable to
                                be?''',
                                options=key_values_cat_count)
                    
                    x_var = key_values_cat_count.get(selected_x_var)

                    selected_y_var = 'selling_price'
                    fig = px.bar(df,x= x_var
                            ,y= selected_y_var
                            ,color='brand'
                            ,barmode='group'
                            # ,title= f"Bar chart of {x} group by {}"
                            ,height=600,color_discrete_sequence=px.colors.qualitative.Vivid)

        elif chart_type == "Line plot":
            selected_x_var = st.selectbox('''What do want the x variable to
                        be?''',
                    options=key_values)
            x_var = key_values.get(selected_x_var)
            # y variable
            selected_y_var = st.selectbox('What about the y?',
                      options=key_values)

            y_var = key_values.get(selected_y_var)
            # Using Plotly for plotting
            fig = px.line(df.sort_values(x_var), x=x_var,
            y=y_var,color='transmission_type') #,markers=True
            # fig.show()
        st.form_submit_button()
    return fig


# ['avg_cost_price', 'vehicle_age', 
#         'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'selling_price', 
#         'engine_log', 'max_power_log', 'seats_log', 
#         'selling_price_log', 'avg_cost_price_log', 
#         'vehicle_age_log', 'km_driven_log', 'mileage_log']

def comparsion_box(chart_type_com, df):
    # with st.sidebar.form('select-form-com'):
    # key_values = {"Average Cost":'avg_cost_price', 
    #             "Vehicle age":'vehicle_age', 
    #             "Km driven":'km_driven', 
    #             "Milage":'mileage', "Engine":'engine', 
    #             "Max power":'max_power', "Seats":'seats', 
    #             "Selling price":'selling_price'}
    col1,col2=st.columns([10,10])
    col3,col4=st.columns([10,10])
    # if chart_type_com == "Boxplot":
        # select_x_var = st.sidebar.radio('''Select the value to be feature to be 
        # displayed''', options=('avg_cost_price', 'vehicle_age', 
        # 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'selling_price')
        
        # Example code
        # key_value = {'Hello':1, 'Jacob':2}
        # input_value = "Hello"
        # key_value.get(input_value)

        # Using key-values
    select_x_var = st.sidebar.radio('''Select the value to be feature to be 
    displayed''', options=(key_values)
    )
    with st.container():
        with col1:
            fig1,ax1 = plt.subplots()
            x_var = key_values.get(select_x_var)
            ax1 = sns.boxplot(df[x_var],palette='nipy_spectral')
            ax1.set_title(f"Skewness: {np.around(df[x_var].skew(),3)}")
            st.pyplot(fig1)

        with col2:
            fig2,ax2 = plt.subplots()
            x_var_log = (f'{x_var}_log')
            ax2 = sns.boxplot(df[x_var_log],palette='nipy_spectral')
            ax2.set_title(f"Skewness of {x_var_log}: {np.around(df[x_var_log].skew(),3)}")
            st.pyplot(fig2)


        # An extension
        with col3:
            fig3,ax3 = plt.subplots()
            x_var = key_values.get(select_x_var)
            # sns.histplot(data=df,x=x_var,kde=True)
            # ax.set_title(f"Histogram plot of {selected_x_var}")
            # ax.set_title(f"Skewness of {x_var}: {np.around(df[x_var].skew(axis=0),2)}")

            ax3 = sns.histplot(data=df,x=x_var,kde=True) #,palette='nipy_spectral'
            ax3.set_title(f"Skewness of {x_var}: {np.around(df[x_var].skew(axis=0),2)}")
            st.pyplot(fig3)

            # sns.histplot(x=log, data=df, ax=ax[index], kde=True,bins=bins_spec[index])
            # ax[index].set_title(f"Skewness in log10: {np.around(df[log].skew(axis=0),2)}",fontsize=14)
        
        with col4:
            fig4,ax4 = plt.subplots()
            x_var_log = (f'{x_var}_log')
            ax4 = sns.histplot(data=df,x=x_var_log,palette='nipy_spectral',kde=True)
            ax4.set_title(f"Skewness of {x_var_log} : {np.around(df[x_var_log].skew(),3)}")
            st.pyplot(fig4)
    
    
    # # """Histogram section"""
    # elif chart_type_com == "Histo-plot":
    #     select_x_var = st.sidebar.radio('''Select the value to be feature to be 
    #     displayed''', options=(key_values))

    #     with st.container():
    #         with col1:
    #             fig1,ax1 = plt.subplots()
    #             x_var = key_values.get(select_x_var)
                
    #             # sns.histplot(data=df,x=x_var,kde=True)
    #             # ax.set_title(f"Histogram plot of {selected_x_var}")
    #             # ax.set_title(f"Skewness of {x_var}: {np.around(df[x_var].skew(axis=0),2)}")

    #             ax1 = sns.histplot(data=df,x=x_var,kde=True) #,palette='nipy_spectral'
    #             ax1.set_title(f"Skewness of {x_var}: {np.around(df[x_var].skew(axis=0),2)}")
    #             st.pyplot(fig1)

    #             # sns.histplot(x=log, data=df, ax=ax[index], kde=True,bins=bins_spec[index])
    #             # ax[index].set_title(f"Skewness in log10: {np.around(df[log].skew(axis=0),2)}",fontsize=14)
            
    #         with col2:
    #             fig2,ax2 = plt.subplots()
    #             x_var_log = (f'{x_var}_log')
    #             ax2 = sns.histplot(data=df,x=x_var_log,palette='nipy_spectral',kde=True)
    #             ax2.set_title(f"Skewness of {x_var_log} : {np.around(df[x_var_log].skew(),3)}")
    #             st.pyplot(fig2)

"""Not Required"""
# Log vs normal plot
# def comparsion_box(df):
#     col1,col2=st.columns([10,10])
#     with st.container():
#         with col1:
#             fig1,ax1 = plt.subplots()
#             x_var = 'price'
#             ax1 = sns.boxplot(df[x_var],palette='nipy_spectral')
#             ax1.set_title(f"Skewness: {np.around(df[x_var].skew(),3)}")
#             st.pyplot(fig1)

#         with col2:
#             fig2,ax2 = plt.subplots()
#             x_var = 'price_log'
#             ax2 = sns.boxplot(df[x_var],palette='nipy_spectral')
#             ax2.set_title(f"Skewness: {np.around(df[x_var].skew(),3)}")
#             st.pyplot(fig2)

#         with col1:
#             fig3,ax3 = plt.subplots()
#             x_var = 'area'
#             ax3 = sns.boxplot(df[x_var],palette='nipy_spectral')
#             ax3.set_title(f"Skewness: {np.around(df[x_var].skew(),3)}")
#             st.pyplot(fig3)

#         with col2:
#             fig4,ax4 = plt.subplots()
#             x_var = 'area_log'
#             ax4 = sns.boxplot(df[x_var],palette='nipy_spectral')
#             ax4.set_title(f"Skewness: {np.around(df[x_var].skew(),3)}")
#             st.pyplot(fig4)
