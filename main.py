# main.py
from fastapi import FastAPI
from joblib import load
import requests
import numpy as np

app = FastAPI()

# Load model
model = load("btc_model.joblib")

# Coinbase fetch function
def fetch_latest_feature():
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=300"
    data = requests.get(url).json()
    data.sort(key=lambda x: x[0])
    if len(data) < 2:
        return [0]
    prev_close = data[-2][4]
    curr_open = data[-1][3]
    return [(curr_open - prev_close) / prev_close]

@app.get("/")
def predict():
    feature = fetch_latest_feature()
    pred = model.predict([feature])[0]
    return {"action": "BUY" if pred == 1 else "SELL"}
