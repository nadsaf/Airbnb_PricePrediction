import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#====================================== Data Airbnb Singapore ============================================

df = pd.read_csv('data-airbnbsg_cleaned.csv')

host_features = ['total_price','host_is_superhost','scores_rating', 'host_response_rate','scores_communication', 'scores_cleanliness','scores_checkin' , 'scores_communication','scores_location', 'number_of_reviews', 'reviews_per_month']
listings_features = ['total_price', 'number_of_reviews', 'bathrooms', 'bedrooms', 'beds', 'accommodates', 'area','region','property_type', 'room_type', 'scores_location', 'scores_cleanliness', 'scores_rating']

num = ['host_is_superhost', 'total_price', 'accommodates', 'bedrooms', 'host_response_rate', 'scores_rating','scores_communication', 'scores_cleanliness','scores_checkin' , 'scores_communication','scores_location', 'number_of_reviews', 'reviews_per_month', 'instant_bookable', 'has_availability']
categories = ['cancellation_policy']

dfnew = pd.concat([df[num], df[categories]], axis=1)
# print(dfnew.isnull().sum() / len(df))
dfnew = dfnew.fillna(0)

#================================================= Machine Learning ==============================================
from sklearn.metrics import mean_squared_error,r2_score, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
'''
Question :
Predicting qualities affect the most to becoming superhost listings
'''
#================================================= Data Preprocessing ============================================

# Labelling categorical feautures -----------------------------------------------------
dfdum = pd.get_dummies(dfnew[categories])
dfnew = dfnew.drop(categories, axis=1)

dfnew = pd.concat([dfnew, dfdum], axis=1)
dfX = dfnew.drop(['host_is_superhost'], axis=1)
dfY = dfnew['host_is_superhost']

# splitting train & test data -----------------------------------------------------
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
    dfX, dfY, 
    test_size= .1,
    random_state = 0
    )
# #================================================= Logistic Regression =================================================
from sklearn.linear_model import LogisticRegression

modelL = LogisticRegression(solver='liblinear')

modelL.fit(xtrain, ytrain)
df_log = pd.DataFrame({'coefficient' : modelL.coef_.ravel(), 'features' : xtrain.columns.values})

skorL = modelL.score(xtest, ytest)
predL = modelL.predict(xtest)
print('Skor Linear Logistic :', round(skorL, 2)*100, '%')
print('Accuracy Linear Logistic :', accuracy_score(ytest,  predL))
# print(df_log)
#================================================= DT Classifier =================================================
from sklearn.tree import DecisionTreeClassifier

modelDT = DecisionTreeClassifier(max_depth=5)
modelDT.fit(xtrain, ytrain)
df_DT = pd.DataFrame({'coefficient' : modelDT.feature_importances_, 'features' : xtrain.columns.values})

skorDT = modelDT.score(xtest, ytest)
predDT = modelDT.predict(xtest)
print('Skor DT :', round(skorDT, 2)*100, '%')
print('Accuracy DT :', accuracy_score(ytest,  predDT))

#================================================= Random Tree Forest Classifier =================================================
from sklearn.ensemble import RandomForestClassifier
modelRF = RandomForestClassifier()
modelRF.fit(xtrain, ytrain)
df_RF= pd.DataFrame({'coefficient' : modelRF.feature_importances_, 'features' : xtrain.columns.values})
skorRF = modelRF.score(xtest, ytest)
predRF = modelRF.predict(xtest)
print('Skor RF :', round(skorRF, 2)*100, '%')
print('Accuracy DT :', accuracy_score(ytest,  predRF))

#================================================= Gradient Boosting Classifier =================================================
from sklearn.ensemble import GradientBoostingClassifier
modelGB = GradientBoostingClassifier()
modelGB.fit(xtrain, ytrain)
df_GB = pd.DataFrame({'coefficient' : modelGB.feature_importances_, 'features' : xtrain.columns.values})
skorGB = modelGB.score(xtest, ytest)
predGB = modelGB.predict(xtest)
print('Skor GBR :', round(skorGB, 2)*100, '%')
print('Accuracy GBR :', accuracy_score(ytest,  predGB))

#================================================= Visualization ==============================================
from matplotlib.colors import ListedColormap 
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid", {'font.family':'serif'})
plt.style.use("seaborn-darkgrid")
color = ['#fccbcb','#F99797', '#F76C6C', '#ABD0E6', '#23305E']
cmap = ListedColormap(color)

ax = sns.barplot(
    x= 'coefficient',
    y= 'features', data= df_log, palette=color)

plt.ylabel('Features')
plt.title('Logistic Regression \n score : {}%'.format(round(skorL, 2)*100))

plt.tight_layout()
plt.subplots_adjust(left= .26)

plt.show() 
plt.close()

sns.barplot(
    x= 'coefficient',
    y= 'features',data= df_DT, palette=color)

plt.ylabel('Features')

plt.title('Decision Tree Classifier \n score : {}%'.format(round(skorDT, 2)*100))
plt.tight_layout()
plt.subplots_adjust(left= .26)

plt.show() 
plt.close()

sns.barplot(
    x= 'coefficient',
    y= 'features',data= df_RF, palette=color)

plt.ylabel('Features')
plt.title('Random Forest Tree Classifier \n score : {}%'.format(round(skorRF, 2)*100))
plt.tight_layout()
plt.subplots_adjust(left= .26)


# plt.show() 
plt.close()

sns.barplot(
    x= 'coefficient',
    y= 'features',data= df_GB, palette=color)

plt.ylabel('Features')
plt.title('Gradient Boosting Classifier \n score : {}%'.format(round(skorGB, 2)*100))
plt.tight_layout()
plt.subplots_adjust(left= .26)
plt.show() 
plt.close()