import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#====================================== Data Airbnb Singapore ============================================

df = pd.read_csv('data-airbnbsg_cleaned.csv')
host_features = ['total_price','host_is_superhost','scores_rating', 'host_response_rate','scores_communication', 'scores_cleanliness','scores_checkin' , 'scores_communication','scores_location', 'number_of_reviews', 'reviews_per_month']
listings_features = ['total_price', 'number_of_reviews', 'bathrooms', 'bedrooms', 'beds', 'accommodates', 'area','region','property_type', 'room_type', 'scores_location', 'scores_cleanliness', 'scores_rating']


#================================================= Machine Learning ==============================================
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score
'''
Features yang digunakan :
1 numeric = 'bedrooms', 'accommodates'
2 categories = 'area', 'property_type', 'room_type'
3 target = total_price
'''
num = ['bedrooms', 'accommodates', 'total_price', 'host_is_superhost']
categories = ['property_type', 'room_type','area']

#================================================= Data Preprocessing ============================================
df = df.dropna(subset= ['bedrooms', 'accommodates','area', 'property_type', 'room_type', 'host_is_superhost'])

# Labelling categorical features -----------------------------------------------------
dfdum = pd.get_dummies(df[categories])

dfnew = df[num]
dfnew = pd.concat([dfnew, dfdum], axis=1)
dfX = dfnew.drop(['total_price'], axis=1)
dfY = dfnew['total_price']

# splitting train & test data -----------------------------------------------------
from sklearn.model_selection import train_test_split, KFold
xtrain, xtest, ytrain, ytest = train_test_split(
    dfX, dfY, 
    test_size= .1,
    random_state = 0
    )

#================================================= Lasso Regression =================================================
from sklearn.linear_model import Lasso, LinearRegression

modelLin = LinearRegression()
modelLin.fit(xtrain, ytrain)
predLin = modelLin.predict(xtest)
skorLin = modelLin.score(xtest, ytest)
df_Lin = pd.DataFrame({'coefficient' : modelLin.coef_, 'features' : xtrain.columns.values})
print('Skor Linear :', round(skorLin, 2)*100, '%')
print('Skor Linear R2 : ', r2_score(ytest, predLin))
print('RMSE Linear: ', mean_squared_error(ytest, predLin)** 0.5)
print(df_Lin.sort_values(by='coefficient', ascending=False))


modelL = Lasso(alpha=.01)
modelL.fit(xtrain, ytrain)
predL = modelL.predict(xtest)
skorL = modelL.score(xtest, ytest)
df_L = pd.DataFrame({'coefficient' : modelL.coef_, 'features' : xtrain.columns.values})
print('Skor Lasso :', round(skorL, 2)*100, '%')
print('Skor Lasso R2 : ', r2_score(ytest, predL))
print('RMSE Lasso: ', mean_squared_error(ytest, predL)** 0.5)
print(df_L.sort_values(by='coefficient', ascending=False))


#================================================= DT Regression =================================================
from sklearn.tree import DecisionTreeRegressor

modelDT = DecisionTreeRegressor(max_depth=5)
modelDT.fit(xtrain, ytrain)

skorDT = modelDT.score(xtest, ytest)
predDT = modelDT.predict(xtest)
df_DT = pd.DataFrame({'coefficient' : modelDT.feature_importances_, 'features' : xtrain.columns.values})
print('Skor DT :', round(skorDT, 2)*100, '%')
print('Skor DT R2 : ', r2_score(ytest, predDT))
print('RMSE DT: ', mean_squared_error(ytest, predDT)** 0.5)
print(df_DT.sort_values(by='coefficient', ascending=False))

#================================================= Random Tree Forest Regression =================================================
from sklearn.ensemble import RandomForestRegressor
modelRF = RandomForestRegressor(n_estimators=100)
modelRF.fit(xtrain, ytrain)
df_RF = pd.DataFrame({'coefficient' : modelRF.feature_importances_, 'features' : xtrain.columns.values})
predRF = modelRF.predict(xtest)
skorRF =modelRF.score(xtest, ytest)
print('Skor RTF :', round(skorRF, 2)*100, '%')
print('Skor RTF R2:', r2_score(ytest, predRF))
print('RMSE RTF: ', mean_squared_error(ytest, predRF)** 0.5)
print(df_RF.sort_values(by='coefficient', ascending=False))

#================================================= Gradient Boosting Regressor =================================================
from sklearn.ensemble import GradientBoostingRegressor
modelGB = GradientBoostingRegressor(max_depth=5)
modelGB.fit(xtrain, ytrain)
df_GB = pd.DataFrame({'coefficient' : modelGB.feature_importances_, 'features' : xtrain.columns.values})
predGB = modelGB.predict(xtest)
skorGB = modelGB.score(xtest, ytest)
print('Skor GBR :', round(skorGB, 2)*100, '%')
print('Skor GBR R2:', r2_score(ytest, predGB))
print('RMSE GBR: ', mean_squared_error(ytest, predGB)** 0.5)
print(df_GB.sort_values(by='coefficient', ascending=False))

#================================================= Visualization ==============================================
from matplotlib.colors import ListedColormap 
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid", {'font.family':'serif'})
plt.style.use("seaborn-darkgrid")
color = ['#fccbcb','#F99797', '#F76C6C', '#ABD0E6', '#23305E']
cmap = ListedColormap(color)
fig = plt.figure(figsize=(10,8))
ax = sns.barplot(
    x= 'coefficient',
    y= 'features', data= df_Lin.head(10), palette=color)

plt.ylabel('Features')
plt.title('Linear \n score : {}%'.format(round(skorLin, 2)*100))

plt.tight_layout()
plt.subplots_adjust(left= .26)

# plt.show() 
# plt.close()

fig = plt.figure(figsize=(10,8))
ax = sns.barplot(
    x= 'coefficient',
    y= 'features', data= df_L.head(10), palette=color)

plt.ylabel('Features')
plt.title('Lasso \n score : {}%'.format(round(skorL, 2)*100))

plt.tight_layout()
plt.subplots_adjust(left= .26)

# plt.show() 
# plt.close()

fig = plt.figure(figsize=(10,8))
sns.barplot(
    x= 'coefficient',
    y= 'features',data= df_DT.head(10), palette=color)

plt.ylabel('Features')

plt.title('Decision Tree \n score : {}%'.format(round(skorDT, 2)*100))
plt.tight_layout()
plt.subplots_adjust(left= .26)

# plt.show() 
# plt.close()

fig = plt.figure(figsize=(10,8))
sns.barplot(
    x= 'coefficient',
    y= 'features',data= df_RF.head(10), palette=color)

plt.ylabel('Features')
plt.title('Random Forest Tree \n score : {}%'.format(round(skorRF, 2)*100))
plt.tight_layout()
plt.subplots_adjust(left= .26)


# plt.show() 
# plt.close()

fig = plt.figure(figsize=(10,8))
sns.barplot(
    x= 'coefficient',
    y= 'features',data= df_GB.head(10), palette=color)

plt.ylabel('Features')
plt.title('Gradient Boosting \n score : {}%'.format(round(skorGB, 2)*100))
plt.tight_layout()
plt.subplots_adjust(left= .26)
plt.show() 
plt.close()