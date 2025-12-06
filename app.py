from flask import Flask
from joblib import load
import requests
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)
model = load('btc_model.joblib')

def fetch_latest_btc():
    end = datetime.utcnow()
    start = end - timedelta(minutes=5)
    url = 'https://api.exchange.coinbase.com/products/BTC-USD/candles'
    params = {'start': start.isoformat(), 'end': end.isoformat(), 'granularity': 300}
    data = requests.get(url, params=params).json()
    latest = data[-1]
    return np.array([latest[3], latest[2], latest[1], latest[4], latest[5]]).reshape(1, -1)

@app.route('/')
def index():
    try:
        latest = fetch_latest_btc()
        pred = model.predict(latest)
        action = 'BUY' if pred[0] == 1 else 'SELL'
        return f"<h1>Next 5-min BTC/USD action: {action}</h1>"
    except Exception as e:
        return f"<h1>Error fetching prediction: {e}</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
