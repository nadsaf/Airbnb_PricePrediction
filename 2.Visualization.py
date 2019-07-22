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

#====================================== Data Airbnb Singapore ============================================
df = pd.read_csv('data-airbnbsg.csv')

host_features = ['total_price','host_is_superhost','scores_rating', 'host_response_rate','scores_communication', 'scores_cleanliness','scores_checkin' , 'scores_communication','scores_location', 'number_of_reviews', 'reviews_per_month']
listings_features = ['total_price', 'number_of_reviews', 'bathrooms', 'bedrooms', 'beds', 'accommodates', 'area','region','property_type', 'room_type', 'scores_location', 'scores_cleanliness', 'scores_rating']

# Property type data  --------------------------------------------------------------------------------
a = df['property_type'].value_counts().reset_index()
fig = plt.figure(figsize=(10,10))
ax =sns.barplot(y='index', x='property_type', data=a, palette=color)
plt.xlabel('Property Type')
plt.ylabel('Count')
plt.title('Airbnb Listings by Property Type', fontsize=15, weight='bold')
for p in ax.patches:
    ax.text(p.get_width()+50, p.get_height()/2 + p.get_y(), int(p.get_width()) ,ha="center")
plt.show()
plt.close()

a['persen'] = a['property_type'] / len(df) *100
# print(a)
a = a[a['persen'] > 4]['index']
a = a.tolist()
df = df[df['property_type'].isin(a)]

#================================================= Exploring the Data =================================================

# Figure : Listings by Region
a = df['region'].value_counts().reset_index().sort_values(by='index').reset_index(drop=True)
f, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 8))
sns.barplot(y='index', x='region', data=a, palette=color, ax=ax[0])
ax[0].set(xlabel='Count of Property Type', ylabel='')
ax[0].set_title('Airbnb Listings by Region', fontsize=15, weight='bold')
numListing =[]
for p in ax[0].patches:
    ax[0].text(p.get_width()+170, p.get_height()/2 + p.get_y(), int(p.get_width()), ha="center")
    numListing.append(int(p.get_width()))
# Figure : Bookings by Region    
a = df.groupby(by=['region'], as_index=False).number_of_reviews.sum()
sns.barplot(y='region', x='number_of_reviews', data=a, palette=color, ax=ax[1])
ax[1].set(xlabel='Numbers of Booking', ylabel='')
ax[1].set_title('Airbnb Bookings in Singapore', fontsize=15, weight='bold')
for p in ax[1].patches:
    ax[1].text(p.get_width()+2600, p.get_height()/2 + p.get_y(), int(p.get_width()) ,ha="center")
plt.tight_layout()
plt.show()
a['numListing'] = numListing
a['supply-demand'] = numListing/a['number_of_reviews'] *100
print(a)


# Figure : Number of Listings by sub-district in Singapore ----------------------------------------
f, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))
a = df.groupby(['area', 'property_type']).size().unstack()
# print(a)
a.plot(kind='barh', stacked=True, ax=ax[0], color=color)
ax[0].set_title('Number of Listings by Subdistrict', fontsize=12)
ax[0].set(xlabel='Count of Property', ylabel='')
ax[0].yaxis.grid(True)
ax[0].legend()
# Figure : Number of Property_type ----------------------------------------------------------------------
a = df['property_type'].value_counts().reset_index().sort_values(by='index').reset_index(drop=True)
sns.barplot(
    x='property_type', y='index', data=a,
    label="Property type", palette=color, ax= ax[1])
ax[1].set(xlabel='Count of Property Type', ylabel='')
ax[1].set_title('Number of Listings by Property', fontsize=12)
ax[1].yaxis.grid(True)
plt.suptitle('Airbnb in Central Singapore Area', fontsize=18, weight='bold')
plt.subplots_adjust(top=.88, wspace=0.3)
# plt.show()
plt.close()

# Figure : Property_type vs price ---------------------------------------------------------------------
f, ax = plt.subplots(figsize=(10, 8), ncols=2, nrows=1)
ax[0].set_xscale("log")

sns.boxplot(x="total_price", y="property_type", data=df,
            whis="range", palette=color, ax=ax[0])

ax[0].set_title("Price by Property Type", fontsize=15, weight='bold')
ax[0].xaxis.grid(True)
ax[0].set(xlabel="Price",ylabel="")
# Figure : Room_type vs price -----------------------------------------------------------------------------
ax[1].set_xscale("log")

sns.boxplot(x="total_price", y="room_type", data=df,
            whis="range", palette=color, ax=ax[1])

ax[1].set_title("Price by Room Type", fontsize=15, weight='bold')
ax[1].xaxis.grid(True)
ax[1].set(xlabel="Price",ylabel="")
plt.tight_layout()
plt.show()
plt.close()

# Figure : Price per guest ------------------------------------------------------------------------------
f, ax = plt.subplots(figsize=(14, 10))

sns.heatmap(df.groupby([
        'area', 'accommodates']).total_price.median().unstack(),
            annot=True, 
            fmt=".0f",
           cmap=cmap)
ax.set(xlabel="Maximum Number of Guest",ylabel="")           
ax.set_title('Price (USD) per Guest', fontsize=15, weight='bold')
plt.show()
plt.close()
# Figure : Area vs price -----------------------------------------------------------------------------
df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,c="total_price", cmap=plt.get_cmap("gnuplot2_r"), colorbar=True, figsize=(10,8))
plt.title('Airbnb Singapore Price', fontsize=12)
plt.show()
plt.close()
# Figure : Area vs price -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize = (10,8))
sns.boxplot(x="area", y="total_price", data=df,
            whis="range", palette=color)
ax.set_yscale("log")
ax.set_title("Price by Area", fontsize=15, weight='bold')
ax.xaxis.grid(True)
ax.set(xlabel="",ylabel="Price")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.subplots_adjust(bottom= .28)
plt.show()
plt.close()

# Figure : Number of Super Host Listings  ----------------------------------------------------------------
f, ax = plt.subplots(figsize=(10, 8))
a = df['area'].value_counts().reset_index()
sns.barplot(
    x='area', y='index', data=a,
    label="Total Listings", color=color[0])

a = df['area'][df['host_is_superhost'] == 1].value_counts().reset_index()

sns.barplot(x="area", y="index", data=a,
            label="Superhost Listings", color=color[-1])

ax.legend(ncol=2, loc="best", frameon=True)
ax.set(ylabel="", xlabel="Airbnb Listings")
sns.despine(left=True, bottom=True)
ax.set_title("Number of Superhost Listings in Central Singapore", fontsize=15, weight='bold')
for p in ax.patches:
    ax.text(p.get_width()+23, p.get_height()/2 + p.get_y(), int(p.get_width()) ,ha="center")
plt.show()
plt.close()

# Correlation Price - Host Features ----------------------------------------------------------------------------

f, ax = plt.subplots(figsize=(12, 10))
corr = df[host_features].corr(method= 'spearman')
sns.heatmap(corr, cmap= cmap, annot=True, fmt=".2f")
ax.set_xticklabels(ax.get_xticklabels())
plt.tight_layout()
plt.subplots_adjust(bottom= .25)
plt.show()
plt.close()

# Correlation Price - Listings Features ------------------------------------------------------------------------
f, ax = plt.subplots(figsize=(12, 10))
corr = df[listings_features].corr(method= 'spearman')
sns.heatmap(corr, cmap= cmap, annot=True, fmt=".2f")
ax.set_xticklabels(ax.get_xticklabels())
plt.tight_layout()
plt.subplots_adjust(bottom= .21)
plt.show()
plt.close()

# save to csv
df.to_csv('data-airbnbsg_cleaned.csv', header=True, index=False) 