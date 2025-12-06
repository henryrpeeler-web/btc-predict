from flask import Flask, jsonify
from joblib import load
import requests
import numpy as np

app = Flask(__name__)

# Load model
model = load("btc_model.pkl")

def fetch_latest_features():
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=300&limit=3"
    data = requests.get(url).json()
    data.sort(key=lambda x: x[0])
    features = []
    for candle in data:
        features.extend(candle[1:5])
    return np.array(features).reshape(1, -1)

@app.route('/')
def index():
    features = fetch_latest_features()
    prediction = model.predict(features)[0]
    action = "BUY" if prediction == 1 else "SELL"
    return f"<h1>BTC-USD 5min Signal: {action}</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
