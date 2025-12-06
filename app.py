from flask import Flask
import numpy as np
from joblib import load

app = Flask(__name__)

# Load trained model
model = load("btc_model.joblib")

# Endpoint for prediction
@app.route("/")
def predict():
    # Fetch latest BTC-USD price
    COINBASE_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    params = {"granularity": 300}
    import requests
    data = np.array(requests.get(COINBASE_URL, params=params).json())
    
    last_close = data[-1][4]  # close price
    pred = model.predict([[last_close]])[0]
    
    return "BUY" if pred == 1 else "SELL"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
