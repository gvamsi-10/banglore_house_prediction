from flask import Flask, request, jsonify , render_template

import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None


app = Flask(__name__)

@app.route('/',methods = ['GET','POST'])
def index():
    return render_template("index.html")

@app.route('/get_location_names',methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': get_location_names()
    })

    response.headers.add('Access-Control-Allow-Origin','*')
    return response

@app.route('/predict_home_price',methods=['GET','POST'])
def predict_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price': get_estimated_price(location, total_sqft, bhk, bath)
    })
    response.headers.add('Access-Control-Allow-Origin','*')

    return response

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    a = np.zeros(len(__data_columns))
    a[0] = sqft
    a[1] = bath
    a[2] = bhk
    if loc_index >= 0:
        a[loc_index] = 1

    return round(__model.predict([a])[0],2)

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    with open("artifacts/columns.json", 'r') as f:
        __data_columns=json.load(f)['data_columns']
        __locations = __data_columns[3:]
    global __model
    with open("artifacts/banglore_home_prices_model.pickle", 'rb') as f:
        __model = pickle.load(f)
    print("loading saved artifacts...done")


def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns


if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    load_saved_artifacts()
    app.run()
