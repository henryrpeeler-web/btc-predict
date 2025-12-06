from flask import Flask, jsonify
from joblib import load
import numpy as np
import requests

app = Flask(__name__)

# Load model
model = load("btc_model.joblib")

# Fetch latest BTC 5-min data
def fetch_latest_btc():
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    params = {"granularity": 300, "limit": 2}  # last 2 candles
    r = requests.get(url, params=params)
    data = r.json()
    data.sort(key=lambda x: x[0])
    last = data[-2:]
    return np.array([[last[0][3], last[0][4]]])  # previous open, close

@app.route("/")
def predict():
    features = fetch_latest_btc()
    pred = model.predict(features)[0]
    action = "BUY" if pred == 1 else "SELL"
    return jsonify({"action": action})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
