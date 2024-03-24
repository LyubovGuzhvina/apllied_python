from fastapi import FastAPI, HTTPException
import uvicorn
import urllib.request

from fastapi.responses import FileResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
import hashlib

from redis import asyncio as aioredis
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse
import pandas as pd
import json
import pickle
import sklearn
from fastapi import UploadFile
from typing import Annotated
from fastapi import File
import redis
import numpy as np
from fastapi.responses import StreamingResponse
from redis import asyncio as aioredis

app = FastAPI()

redis_url = 'rediss://red-co06gvi0si5c73fi6r80:LXRfrjxT6TEX3WE0J8cxsuUiMe1BJbex@frankfurt-redis.render.com:6379'
redis = redis.Redis.from_url(redis_url)

@app.on_event("startup")
async def startup_event():
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

def cache_key_generation(text):
    return hashlib.md5(str(text).encode('utf-8')).hexdigest()


@app.get("/hello_world")
@cache(expire=60)
async def index():
    return dict(hello = "world")

class Item(BaseModel):
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: int
    max_power: float
    seats: int

class Items(BaseModel):
    objects: List[Item]

cars_model = pickle.load(open("cars_model.pkl", "rb"))
scaler = pickle.load(open("standard_scaler.pkl", "rb"))
transformer = pickle.load(open("transformer.pkl", "rb"))

@app.get("/get_train_data")
@cache(expire=60)
#download original dataset (train)
def get_data():
    data_train = urllib.request.urlretrieve('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_train.csv', 'cars_train.csv')
    return FileResponse(path = 'cars_train.csv', filename='cars_train.csv')

@app.get("/get_test_data")
#download original dataset (test)
def get_data():
    data_test = urllib.request.urlretrieve('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_test.csv', 'cars_test.csv')
    return FileResponse(path = 'cars_test.csv', filename='cars_test.csv')

@app.post("/predict_item")
#obtain a model prediction for single user item
def predict_item(item: Item) -> float:

    cache_key = cache_key_generation(item)

    if redis.exists(cache_key):
        prediction = pickle.loads(redis.get(cache_key))
        return {"prediction": prediction, 'From cash': True}
    else:
        input_data = item.json()
        input_dictionary = json.loads(input_data)
        year = input_dictionary["year"]
        km_driven = input_dictionary["km_driven"]
        fuel = input_dictionary["fuel"]
        seller_type = input_dictionary["seller_type"]
        transmission = input_dictionary["transmission"]
        owner = input_dictionary["owner"]
        mileage = input_dictionary["mileage"]
        engine = input_dictionary["engine"]
        max_power = input_dictionary["max_power"]
        seats = input_dictionary["seats"]
        my_list = [year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]
        my_list.append(max_power / engine)
        my_list.append(year ** 2)
        my_list = pd.DataFrame([my_list], columns=["year", "km_driven", "fuel", "seller_type", "transmission",
                                                   "owner", "mileage", "engine", "max_power", "seats",
                                                   "max_power_engine", "year_squared"])
        tramsformed_list = transformer.transform(my_list)
        scaled_list = scaler.transform(tramsformed_list)
        prediction = cars_model.predict(scaled_list)
        redis.set(cache_key, pickle.dumps(prediction))
        return {"prediction": prediction, 'From cash': False}

@app.post("/predict_item1")
def predict_item(item: Item) -> float:
    input_data = item.json()
    input_dictionary = json.loads(input_data)
    # name = input_dictionary["name"]
    year = input_dictionary["year"]
    # selling_price = input_dictionary["selling_price"]
    km_driven = input_dictionary["km_driven"]
    fuel = input_dictionary["fuel"]
    seller_type = input_dictionary["seller_type"]
    transmission = input_dictionary["transmission"]
    owner = input_dictionary["owner"]
    mileage = input_dictionary["mileage"]
    engine = input_dictionary["engine"]
    max_power = input_dictionary["max_power"]
    seats = input_dictionary["seats"]
    my_list = [year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]
    my_list.append(max_power/engine)
    my_list.append(year**2)
    my_list = pd.DataFrame([my_list], columns = ["year", "km_driven", "fuel", "seller_type", "transmission",
                                               "owner", "mileage", "engine", "max_power", "seats", "max_power_engine", "year_squared"])
    tramsformed_list = transformer.transform(my_list)
    scaled_list = scaler.transform(tramsformed_list)
    prediction = cars_model.predict(scaled_list)
    return prediction








@app.post("/uploadcsv/")
#obtain a model prediction for user items listed in csv file
def upload_csv(csv_file: UploadFile = File(...)) -> List[float]:
    df = pd.read_csv(csv_file.file)
    tramsformed_list = transformer.transform(df)
    scaled_list = scaler.transform(tramsformed_list)
    df["selling_price_pred"] = cars_model.predict(scaled_list)
    return df["selling_price_pred"]

@app.get("/ping")
async def ping():
    return {"message": "pong"}




