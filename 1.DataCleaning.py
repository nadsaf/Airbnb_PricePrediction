import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid", {'font.family':'serif'})
plt.style.use("seaborn-darkgrid")
color = ['#fccbcb','#F99797', '#F76C6C', '#ABD0E6', '#23305E']
cmap = ListedColormap(color)
color2 = ['#90113f', '#afb5c0', '#e39292', '#61749f', '#b4d5ea', '#c26f6d']
#----------------------------------------------------------------------------------------------------
df = pd.read_csv(
    './Airbnb Singapore/listings.csv.gz', 
    low_memory=False, 
    parse_dates=['host_since', 'first_review', 'last_review'], 
    na_values=['na', 'nan']
    )

# print(df.columns.tolist())
# print(df.info())

#================================================= Drop missing value in listings dataframe ===========================================

missing_data = df.isnull().sum() / len(df)
# print(missing_data)
missing_half = df.columns[missing_data > 0.5]
df = df.drop(missing_half, axis= 1)
df['host_response_rate'] = df['host_response_rate'].apply(lambda x: float(str(x).replace("%", "")))


#================================================= Grouping the Non-numerical features =================================================

# for i in df.columns:
#     print(i, type(df[i][0]), '//', df[i][0])

group_categorical = ['neighbourhood_cleansed','neighbourhood_group_cleansed','property_type', 'room_type', 'bed_type', 'cancellation_policy']
group_full_text = ['name', 'summary', 'space','description', 'experiences_offered', 'neighborhood_overview', 'notes','smart_location', 'transit', 'access','interaction', 'house_rules', 'host_about', 'host_location', 'host_neighbourhood', 'street', 'city','neighbourhood']
group_dropping = ['host_response_time','listing_url', 'scrape_id', 'picture_url', 'host_url', 'host_thumbnail_url', 'host_picture_url', 'last_scraped', 'host_name', 'calendar_last_scraped', 'calendar_updated', 'host_verifications', 'amenities','zipcode','market', 'country', 'requires_license', 'is_business_travel_ready', 'require_guest_profile_picture', 'require_guest_phone_verification', 'is_location_exact',  'host_has_profile_pic', 'host_identity_verified', 'country_code']
group_date = ['host_since', 'first_review', 'last_review']
group_bool = ['host_is_superhost', 'instant_bookable','has_availability']
group_money = ['extra_people', 'cleaning_fee', 'security_deposit', 'price']

df[group_bool] = df[group_bool].apply(lambda x: pd.Series(x).map({'t' : 1, 'f': 0}))
# print(df[group_bool])

df[group_money] = df[group_money].apply(lambda x: x.str.replace("$", "").str.replace(",","").astype('float'))
df[['cleaning_fee', 'security_deposit']] = df[['cleaning_fee', 'security_deposit']].fillna(round(df[['cleaning_fee', 'security_deposit']].mean()))  

# print(df[group_money])

df = df.drop(group_full_text, axis=1)
df = df.drop(group_dropping, axis =1)

a = df.index[df['price'] == 0].tolist()
df = df.drop(df.index[a])

df = df[df['bed_type'] == 'Real Bed']

#================================================= Grouping numerical features =================================================

group_numeric = df.select_dtypes(include=['int64', 'float64']).columns.values
# print(group_numeric)

df = df.dropna(subset=['number_of_reviews', 'bathrooms', 'bedrooms', 'beds', 'accommodates'])

df['bedrooms'] = df['bedrooms'].clip(lower=1)

df = df[df['minimum_nights_avg_ntm'] <30]

df.loc[df['accommodates'] < df['guests_included'], 'accommodates'] = df.loc[df['accommodates'] < df['guests_included'], 'guests_included'].values
# print(len(df[df['accommodates'] >= df ['guests_included' ]])/len(df))

df['total_price'] = df['price'] + (df['extra_people'] * (df['accommodates'] - df['guests_included']))
# print(df[df['price'] == df['total_price']][['total_price', 'price']])
df = df.rename(columns={"neighbourhood_cleansed": "area", "neighbourhood_group_cleansed" : "region"})
df.columns = df.columns.str.replace('review_scores', 'scores')

#=================================================================================
from scipy import stats

Q1 = df['total_price'].quantile(0.25)
Q3 = df['total_price'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
print(len(df['area'].unique()))

index_o = df.index[(df['total_price'] < (Q1 - 1.5 * IQR)) |(df['total_price'] > (Q3 + 1.5 * IQR)) & (df['area'] != 'Southern Islands') & (df['area'] != 'Tuas')]

print(df['total_price'].describe())
print(len(df))
df = df.drop(index=index_o, axis=1)

print(len(df['area'].unique()))
print(len(df))

print(df['total_price'].describe())


# save to csv
df.to_csv('data-airbnbsg.csv', header=True, index=False) 