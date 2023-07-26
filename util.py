import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

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
    with open("./artifacts/columns.json",'r') as f:
        __data_columns=json.load(f)['data_columns']
        __locations = __data_columns[3:]
    global __model
    with open("./artifacts/banglore_home_prices_model.pickle",'rb') as f:
        __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())