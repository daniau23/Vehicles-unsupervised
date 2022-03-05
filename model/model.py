import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as gos
import plotly.figure_factory as ff
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score,silhouette_samples,adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import GridSearchCV,train_test_split,KFold,StratifiedShuffleSplit
from sklearn.pipeline import  Pipeline
from sklearn.neighbors import KNeighborsClassifier
import pickle

df = pd.read_csv("cardekho_updated.csv.zip")

df.sample(random_state=30,n=5)


df.info(memory_usage='deep')

## Data Cleaning
# Checking for Null values
df.isna().sum()
df.columns
df.dropna(axis='rows',subset=['full_name', 'selling_price', 'year', 'seller_type',
       'km_driven', 'owner_type', 'fuel_type', 'transmission_type', 'mileage',
       'engine', 'max_power', 'seats'],how='any',inplace=True)
# default how=any, inplace = False

# df.shape
# The dataset has 19542 rows and 13 columns

df.sample(random_state=30,n=5)

# **Creating an age value from the year**
current_year = 2021
df['vehicle_age'] = current_year-df.year
df.drop(['year'],axis='columns',inplace=True)

df.sample(random_state=30,n=5)

# **Creating column "brand" & "model" from "full_name**
df.full_name.sample(n=5,random_state=30)

# Total number of unique vehicles
df.full_name.nunique()

# Total number of vehicles
df.full_name.value_counts().sum()
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html?highlight=replace#pandas.DataFrame.replace
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.replace.html?highlight=replace#pandas.Series.str.replace
# https://docs.python.org/3/library/stdtypes.html#str.replace
df['full_name'] = df["full_name"].str.replace(" New", " ")
df.full_name.sample(n=5,random_state=30)

# Reference
# a = "hello gegete"
# a.split(' ')
# # Output
# # ['hello', 'gegete']
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.split.html?highlight=split
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.split.html?highlight=split#pandas.Series.str.split
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.get.html?highlight=str%20get#pandas.Series.str.get

# Obtaining the brand value
df["brand"] = df.full_name.str.split(' ').str.get(0)

# Example
# df.loc[:,['brand','full_name']]
# df.iloc[[0,4,5],[0,4]]
# df.iloc[0,4]
# End of example

# replacing all values in the brand column that has the value "Land" with "Land Rover"
df.loc[(df.brand == 'Land'),'brand']='Land Rover'



# Example
# a = "hello gegete hegee hfhfhfh"
# a.split(' ')
# # 3 the stopping index is not included
# " ".join(a.split(' ')[0:3]) # Output --> 'hello gegete hegee'
# # " ".join(a[1:3].split(' ')) # Output --> 'el'
# End of Example

# Creating the model 
df["model"] = df["full_name"].apply(lambda x: "".join(x.split(' ')[1:3]) if "Dzire" in x else ''.join(x.split(' ')[1]))
df.loc[:5,["full_name",'brand',"model"]]

# Renaming car models 
df.loc[(df.model == "Wagon"),'model'] = "Wagon R"
df.loc[(df.model == "E"),'model'] = 'E Verito'
df.loc[(df.model == "Land"),'model'] = 'Land Cruiser'
df.loc[:5,["full_name",'brand',"model"]]
# Dropping full_name
df.drop('full_name',axis=1,inplace=True)
# or 
# df.drop(columns=['full_name'])

# or
# df.drop(['B', 'C'], axis=1)
# Creating car_name column

df["car_name"] = df['brand'] + " "+df['model']
df_unique = pd.DataFrame(df['car_name'].value_counts())
df.loc[:,['brand','model','car_name']].sample(n=5,random_state=30)

# **Converting "new_price" into "min_price" & "max_price**
# df_unique
# df
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.lstrip.html?highlight=lstrip#pandas.Series.str.lstrip
df["new_price1"] = df['new_price'].str.lstrip('New Car (On-Road Price) : Rs.')

df.new_price1 = df.new_price1.str.replace('[*,,]', '')
df[['new_price1','unit']] = df.new_price1.str.split(' ',expand=True)
df[['min_cost_price','max_cost_price']] = df.new_price1.str.split("-",expand=True)

df.min_cost_price = df.min_cost_price.str.replace('[A-Za-z]', '')
df.max_cost_price = df.max_cost_price.str.replace('[A-Za-z]', '')
df[['new_price1','unit','min_cost_price','max_cost_price']]
df.drop(['new_price','new_price1'],axis=1,inplace=True)
df.head()
df.info(memory_usage='deep')
# Converting data type to float
df.max_cost_price = df.max_cost_price.astype('float64')
df.min_cost_price = df.min_cost_price.astype('float64')

# **Converting cost price to appropriate units**
# Converting cost price to appropriate units
# 1 Lakh = 100000 units
df.loc[df.unit == "Lakh", 'min_cost_price'] = df.min_cost_price * 100000.0
df.loc[df.unit == "Lakh", 'max_cost_price'] = df.max_cost_price * 100000.0

# 1 Cr = 10000000.0 units
df.loc[df.unit == "Cr", 'min_cost_price'] = df.min_cost_price * 10000000.0
df.loc[df.unit == "Cr", 'max_cost_price'] = df.max_cost_price * 10000000.0

df.isna().sum()
df.drop(['unit'],axis='columns', inplace=True)

# Filling cars whose "max_cost_price" is missing with "min_cost_price"
df.max_cost_price = df.max_cost_price.fillna(df.min_cost_price)
same_as_max_cost = df.max_cost_price == df.min_cost_price
df[same_as_max_cost]
df.loc[[377,296],['max_cost_price','min_cost_price']]
df.shape
df.drop(df[same_as_max_cost].index, inplace=True)
df.shape
df.head()
df.groupby('car_name')
# Filling missing cost price of cars with the mean of their respective car models
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.DataFrameGroupBy.transform.html?highlight=transform#pandas.core.groupby.DataFrameGroupBy.transform
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.SeriesGroupBy.transform.html?highlight=transform#pandas.core.groupby.SeriesGroupBy.transform
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html?highlight=fillna#pandas.DataFrame.fillna
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html?highlight=groupby#pandas.DataFrame.groupby
# https://pbpython.com/pandas_transform.html

groupby_transform = df.groupby('car_name')

df.min_cost_price = df.min_cost_price.fillna(groupby_transform['min_cost_price'].transform('mean'))
df.max_cost_price = df.max_cost_price.fillna(groupby_transform['max_cost_price'].transform('mean'))
df.head()

# **Converting selling price to appropriate units**
# Converting selling price to appropriate units
# Removing the asterisk
df.selling_price = df.selling_price.str.replace('[*,,]', '')
df[["selling_price","unit"]] = df.selling_price.str.split(expand=True)
df.selling_price = df.selling_price.astype('float64')

# # Some values of selling price and cost price have different units
# 1 Lakh = 100000 units
df.loc[df.unit == "Lakh", 'selling_price'] = df.selling_price * 100000.0

# 1 Cr = 10000000.0 units
df.loc[df.unit == "Cr", 'selling_price'] = df.selling_price * 10000000.0

df.head()
df.drop(['unit'],axis='columns', inplace=True)
df.head()

# **Removing unwanted non-numeric data from columns**
cols = [ "mileage","km_driven","engine","max_power","seats"]
df[cols] = df[cols].replace(r'[^\d.]+','',regex=True)
df[cols] = df[cols].replace('','0')

df.head()
# Dropping null values
df.dropna(how='any',axis=0,inplace=True)



df= df.astype({'km_driven': 'float64', 
                'mileage': 'float64', 
                'engine': 'float64', 
                'max_power': 'float64', 
                'seats': 'float64',
                'min_cost_price': 'float64',
                'max_cost_price': 'float64'
                })

df.info(memory_usage='deep')
col_order=['car_name','brand','model',
            'min_cost_price',
            'max_cost_price',
            'vehicle_age',
            'km_driven',
            'seller_type','fuel_type',
            'transmission_type',
            'mileage',
            'engine',
            'max_power',
            'seats',
            'selling_price'
        ]
df=df[col_order]
df.head()
df.describe().T

# Dropping zero valued cells
df.drop(df[df['seats'] == 0].index, inplace = True)
df.drop(df[df['mileage'] == 0].index, inplace = True)
df.drop(df[df['km_driven'] == 0].index, inplace = True)
df.drop(df[df['vehicle_age'] == 0].index, inplace = True)
df.drop(df[df['max_power'] == 0].index, inplace = True)
df.info(memory_usage='deep')
df_copy = df.copy()

# Dropping out of boundary values
vehicle_ge_20 = df_copy['vehicle_age'] > 20
km_driven_300000 = df_copy['km_driven'] > 300000

df_copy.drop(df_copy[vehicle_ge_20].index, inplace=True)
df_copy.drop(df_copy[km_driven_300000].index, inplace = True)
def remove_outliers(data,col):
    Q_1 = np.quantile(data[col],0.25)
    Q_3 = np.quantile(data[col],0.75)
    IQR = Q_3 - Q_1 
    lower_bound = Q_1 - (1.5* IQR)
    upper_bound = Q_3 + (1.5* IQR)
    print(f"IQR value for column {col} is: {IQR}")
    print(f"Lower bound value for column {col} is: {lower_bound}")
    print(f"Upper bound value for column {col} is: {upper_bound}\n")
    # Creating a general bound(range) for the removal of outliers
    outlier_free_list = [x for x in data[col] if ((x > lower_bound) & (x < upper_bound))]
    filetered_data1 = data[col].isin(outlier_free_list)
    global filetered_data2
    filetered_data2 = data.loc[filetered_data1]

out_columns = df_copy[['km_driven',
                'vehicle_age',
                'mileage',
                'engine',
                'max_power',
                'seats',
                'selling_price',
                'min_cost_price',
                'max_cost_price'
                ]]  
for i in out_columns:
    remove_outliers(df_copy, i)

# Assigning filtered data back to our original variable'
df_copy = filetered_data2

df_copy.shape

# **Converting "min_cost_price" and "max_cost_price" to "avg_cost_price" using mean**
# 1 Lakh = 100,000 units
df_copy['avg_cost_price'] =(df_copy.min_cost_price + df_copy.max_cost_price)/2
col_order=['car_name','brand','model',
            'min_cost_price',
            'max_cost_price',
            'avg_cost_price',
            'vehicle_age',
            'km_driven',
            'seller_type','fuel_type',
            'transmission_type',
            'mileage',
            'engine',
            'max_power',
            'seats',
            'selling_price'
        ]
df_copy=df_copy[col_order]
df_copy.head()
df_copy.drop(['min_cost_price','max_cost_price'], axis=1, inplace=True)
df_copy['avg_cost_price'] = df_copy['avg_cost_price']/100000
df_copy['selling_price'] = df_copy['selling_price']/100000
df_copy.head()

# For another purpose
df_copy_2 = df_copy.copy()
df_copy.head(20)
df_copy.select_dtypes(include='O').columns
unique = ['car_name', 'brand', 'model', 'seller_type', 'fuel_type',
       'transmission_type']

for x in unique:
    # Accessing the unique labels in each  column
    print(f"In column {x}: {df_copy[x].unique()}")
    # Total number of unique labels in each column
    print(f"Total number of unique values: {df_copy[x].nunique()}\n")


# **Memory management**
df_copy.info(memory_usage='deep')
df_copy.select_dtypes(include='O').columns
df_copy_2 = df_copy.copy()
for col in df_copy_2.columns:
    if df_copy_2[col].dtype == np.object:
        df_copy_2[col] = df_copy_2[col].astype('category')
df_copy_2.dtypes
df_copy_2.info(memory_usage='deep')
df_copy_2.describe().T


# | Data-Type | Precision |
# | ----------- | ----------- |
# float16   | 3
# float32   | 6
# float64   | 15
# float128  | 18

# --------------------------------- 


# |Data type |min|max|
# | ----------- | ----------- |----------- |
# |int8|-128|127|
# |int16|-32768|32767|
# |int32|-2147483648|2147483647|
# |int64|-9223372036854775808|9223372036854775807|
# np.iinfo(np.int64)
# np.finfo(np.float16)
for col in df_copy_2.columns:
    if df_copy_2[col].dtype == np.float64:
        df_copy_2[col] = df_copy_2[col].astype(np.float32)
df_copy_2.describe().T
df_copy_2.info(memory_usage='deep')
for col in df_copy_2.columns:
    if df_copy_2[col].dtype == np.int64:
        df_copy_2[col] = df_copy_2[col].astype(np.int8)
df_copy_2.info(memory_usage='deep')
df_copy_2.describe().T
# df_copy_2[df_copy_2.duplicated()].head(10)

## Performing EDA
# The target variable is the selling price
df_copy_2.dtypes
df_copy_2['seats'] = df_copy_2.seats.astype('int8')
df_copy_2.dtypes

# **Final Histogram plot**
nums = df_copy_2.select_dtypes(include=np.number)
nums

# **Using Log to correct skewness**

# Levels of skewness
# 1. (-0.5,0.5) = lowly skewed
# 2. (-1,0-0.5) U (0.5,1) = Moderately skewed
# 3. (-1 & beyond ) U (1 & beyond) = Highly skewed

fig, ax = plt.subplots(2,4, figsize=(30,20), constrained_layout=True)
ax = ax.ravel()
sns.set_theme(rc=None)
df_copy_log = df_copy_2.copy()
# row, column - (1,1),-> (1,'hdlngth'), (1,2) -> (1 ,'skullw')
# It's accessing the list
# bins_spec = [15,'auto',50,15,20,20,'auto',10]
for index, value in enumerate(nums):
    log = (f'{value}_log')
    df_copy_log[log] = df_copy_log[value].apply(lambda x: np.log(x+1))
    # sns.histplot(x=log, data=df_copy_log, ax=ax[index], kde=True,bins=bins_spec[index])
    # ax[index].set_title(f"Skewness in log10: {np.around(df_copy_log[log].skew(axis=0),2)}",fontsize=14)


for value in df_copy_log.select_dtypes(include=np.number):
    print(f'Skewness of {value} is: {np.around(df_copy_log[value].skew(axis=0),3)}')



# **Box plot**
# Creating a list of obejcts
objects = df_copy_2.select_dtypes(include='category').columns.tolist()
# model is index 1
# brand & car_name is index 0 
# All were popped
objects.pop(1)
objects.pop(0)
objects.pop(0)
objects


X_mix = ['seats','max_power_log','engine_log','mileage','km_driven','vehicle_age_log','avg_cost_log']
y_regular = ['selling_price']
y_log = ['selling_price_log']

# Log then scale
# https://stats.stackexchange.com/questions/402470/how-can-i-use-scaling-and-log-transforming-together

## Model building 
# + In this section different algorithms will be applied such as;  Clustering,KNN, PCA and regression methods.
# + Before the application of each model a scaler shall be applied to it to ensure accuracy and precision.

# - Columns to be explored;
#     + Brand
#     + Fuel type
#     + Seller type
#     + Transmission type
#     + Selling price

df_copy_log.columns

X_mix = ['seats','max_power_log','engine_log','mileage','km_driven','vehicle_age_log','avg_cost_price_log']
y_regular = ['selling_price']
y_log = ['selling_price_log']

### **Use of Clustering Algorithms**

X_mix_kmean = ['seats','max_power_log','engine_log','mileage','km_driven','vehicle_age_log','avg_cost_price_log','selling_price_log']
df_kmeans = df_copy_log[X_mix_kmean]
df_kmeans[['seller_type','fuel_type','transmission_type']] = df_copy_2[objects]
df_kmeans
df_kmeans[['model','brand']] = df_copy_2[['model','brand']]
df_kmeans
df_kmeans.info(memory_usage='deep')
df_kmeans.select_dtypes(include=np.number).columns
df_kmeans.select_dtypes(exclude=np.number).columns

# Use get dummies to encode the categorical part of the data
dummies = pd.get_dummies(df_kmeans)
dummies

dummies_1 = dummies.copy()

# Making use of Robust Scaler and PCA
scaler3 = RobustScaler()
dummies1_scaled = scaler3.fit_transform(dummies_1)
pca = PCA(n_components=.90).fit(dummies1_scaled)
x_pca = pca.transform(dummies1_scaled)

# Using DBSCAN
dbscan = DBSCAN(eps=1.7,min_samples=10) 
# eps=2,min_samples=10
clusters = dbscan.fit_predict(x_pca)
print("Cluster memberships:\n{}".format(clusters))

# This stores the clusters labels 
listing = clusters.tolist()


### **Analysing the Results of PCA and Clustering (DBSCAN)**
# Making copies
x_pca2 = x_pca.copy()
dummies_2 = dummies_1.copy()
dummies_2
# x_pca2_frame = pd.DataFrame(x_pca2)
# x_pca2_frame
df_dummies1_pca_dbscan = pd.concat([dummies_2.reset_index(drop = True), pd.DataFrame(x_pca2)], axis=1)
df_dummies1_pca_dbscan.columns.values[-16:] = ['component_1',
                                                'component_2',
                                                'component_3',
                                                'component_4',
                                                'component_5',
                                                'component_6',
                                                'component_7',
                                                'component_8',
                                                'component_9',
                                                'component_10',
                                                'component_11',
                                                'component_12',
                                                'component_13',
                                                'component_14',
                                                'component_15',
                                                'component_16',]

df_dummies1_pca_dbscan['segement_dbscan_pca'] = dbscan.labels_
df_dummies1_pca_dbscan.head()
df_dummies1_pca_dbscan["segment"] = df_dummies1_pca_dbscan['segement_dbscan_pca'].map({0:'first',
                                                                                1:'second', 
                                                                                -1:'noise'})

# **This is an example for dropping specific rows in a dataset**
# df_dummies1_pca_dbscan_mycopy = df_dummies1_pca_dbscan.copy()
# i = df_dummies1_pca_dbscan_mycopy.loc[df_dummies1_pca_dbscan['segment'] == 'noise','segment'].index
# i
# df_dummies1_pca_dbscan_mycopy.drop(i, inplace=True)
# df_dummies1_pca_dbscan_mycopy.loc[df_dummies1_pca_dbscan['segment'] == 'noise','segment'].value_counts()
# **End of example**

# Selecting  components 1-16 and segment_dbscan_pca
df_dummies1_pca_dbscan[['component_1',
                                                'component_2',
                                                'component_3',
                                                'component_4',
                                                'component_5',
                                                'component_6',
                                                'component_7',
                                                'component_8',
                                                'component_9',
                                                'component_10',
                                                'component_11',
                                                'component_12',
                                                'component_13',
                                                'component_14',
                                                'component_15',
                                                'component_16',
                                                'segement_dbscan_pca']]
# https://stackoverflow.com/questions/43136137/drop-a-specific-row-in-pandas
i = df_dummies1_pca_dbscan.loc[df_dummies1_pca_dbscan['segment'] == 'noise','segment'].index
i
df_dummies1_pca_dbscan.drop(i, inplace=True)

# df.loc[df.unit == "Lakh", 'min_cost_price'] = df.min_cost_price * 100000.0
# # The code above serves as a reference to the codes below:
# df_dummies1_pca_dbscan.loc[df_dummies1_pca_dbscan['segment'] == 'noise','segment'].value_counts()
# df_dummies1_pca_dbscan.loc[df_dummies1_pca_dbscan['segment'] == 'first','segment'].value_counts()
# df_dummies1_pca_dbscan.loc[df_dummies1_pca_dbscan['segment'] == 'second','segment'].value_counts()

df_dummies1_pca_dbscan.segement_dbscan_pca.dtype
df_knn_set = df_dummies1_pca_dbscan[['component_1',
                                                'component_2',
                                                'component_3',
                                                'component_4',
                                                'component_5',
                                                'component_6',
                                                'component_7',
                                                'component_8',
                                                'component_9',
                                                'component_10',
                                                'component_11',
                                                'component_12',
                                                'component_13',
                                                'component_14',
                                                'component_15',
                                                'component_16',
                                                'segement_dbscan_pca']]
df_knn_set.isnull().sum()
X = df_knn_set.drop(['segement_dbscan_pca'],axis=1)
y = df_knn_set['segement_dbscan_pca']


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
clf = KNeighborsClassifier()
params_grid = {'n_neighbors':[5,6,7],
                "weights":['uniform','distance'],
                "p":[1,2]}

my_cv = StratifiedShuffleSplit(n_splits=10,test_size=.2,train_size=.8)

# Try to specify the scoring for the grid
mygrid = GridSearchCV(clf,param_grid=params_grid,cv=my_cv,scoring='roc_auc')
mygrid.fit(X_train,y_train)


# Exporting the algorithm
pickle.dump(mygrid, open('vehicle.pkl', 'wb'))