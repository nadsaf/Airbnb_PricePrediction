import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('data-airbnbsg_cleaned.csv')
host_features = ['total_price','host_is_superhost','scores_rating', 'host_response_rate','scores_communication', 'scores_cleanliness','scores_checkin' , 'scores_communication','scores_location', 'number_of_reviews', 'reviews_per_month']
listings_features = ['total_price', 'number_of_reviews', 'bathrooms', 'bedrooms', 'beds', 'accommodates', 'area','region','property_type', 'room_type', 'scores_location', 'scores_cleanliness', 'scores_rating']
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
num = ['bedrooms', 'accommodates', 'total_price', 'host_is_superhost']
categories = ['property_type', 'room_type','area']
df = df.dropna(subset= ['bedrooms', 'accommodates','area', 'property_type', 'room_type', 'host_is_superhost'])
dfdum = pd.get_dummies(df[categories])
dfnew = df[num]
dfnew = pd.concat([dfnew, dfdum], axis=1)
dfX = dfnew.drop(['total_price'], axis=1)
dfY = dfnew['total_price']
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
    dfX, dfY, 
    test_size= .1,
    random_state = 0
    )

from sklearn.ensemble import GradientBoostingRegressor
modelGB = GradientBoostingRegressor(max_depth=5)
modelGB.fit(xtrain, ytrain)
skorGB = round(modelGB.score(xtest, ytest),2)
dict_col = dict(zip(xtrain.columns,range(len(xtrain))))
df_fi = pd.DataFrame({'coefficient' : modelGB.feature_importances_, 'features' : xtrain.columns.values})
print(df_fi.sort_values(by='coefficient', ascending=False))
print(modelGB.feature_importances_)
print(xtrain.columns.values)
import joblib
joblib.dump([modelGB, skorGB, dict_col], 'modelGB')