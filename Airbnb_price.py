from flask import Flask, abort, render_template, request, send_from_directory, url_for, redirect, jsonify
import joblib
import pandas as pd
import numpy as np
import requests
import requests
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/map')
def map():
    return render_template('map.html')

@app.route('/price')
def price():
    return render_template('price.html')

@app.route('/priceprediction', methods=['POST', 'GET'])
def priceprediction():
    if request.method == 'POST':
        area = request.form['area']
        guests = int(request.form['guests'])
        bedrooms = int(request.form['bedrooms'])
        propertytype = request.form['propertytype']
        roomtype = request.form['room']
        form_all = request.form.to_dict()

        pred_value = np.zeros(len(dict_col))        
        pred_value[dict_col['area_'+str(area)]] = 1
        pred_value[dict_col['property_type_'+str(propertytype)]] = 1
        pred_value[dict_col['bedrooms']] = bedrooms
        pred_value[dict_col['accommodates']] = guests
        pred_value[dict_col['room_type_'+str(roomtype)]] = 1
        
        try:
            if int(request.form.get('superhost')) == 1:
                superhost = 1
                form_all['superhost'] = 'Superhost'
        except:
            superhost = 0
            form_all['superhost'] = 'Not Superhost'
        pred_value[dict_col['host_is_superhost']] = superhost
                       
        price = round(model.predict(pred_value.reshape(1, -1))[0], 2)
        # Change currency
        key = 'f8868785f8186c4b59ce'
        url = 'https://free.currconv.com/api/v7/convert?q=USD_SGD,USD_IDR&compact=ultra&apiKey={}'.format(key)

        raw = requests.get(url).json()
       
        price_SGD = round(raw['USD_SGD'] * price, 2)
        price_IDR = round(raw['USD_IDR'] * price, 2)
        result = {
            'price' : abs(price), 'price_IDR' : abs(price_IDR), 'price_SGD' : abs(price_SGD), 'skorGB' : skorGB
        }

        # Recommendation based on Area
        df = pd.read_csv('data-airbnbsg_cleaned.csv')
        df =  df[['area','accommodates','bedrooms', 'property_type', 'room_type','total_price']]
        if df['area'].str.contains(str(area)).count() > 0:
            df = df[df['area'] == str(area)]
            if df[df['accommodates'] == guests]['accommodates'].count() > 0:
                df = df[df['accommodates'] == guests]
            elif df[df['accommodates'] > guests]['accommodates'].count() > 0:
                df = df[df['accommodates'] > guests]
            else:
                df = df[df['accommodates'] < guests]
        else:
            df = df

        df = df.head()
        print(df)

        return render_template('result.html', result=result, form_all=form_all, df=df)
    else:
        return redirect(url_for('home'))



#========================================================================================================
# TEMP IMAGE
@app.route('/filetemp/<path:path>')                           
def filetemp(path):
    return send_from_directory('./templates/image', path)

@app.errorhandler(404)                                              
def notFound(error) :                                               
    return render_template('error.html')

if __name__ == '__main__':
    model, skorGB, dict_col = joblib.load('modelGB')
    app.run(debug = True)  