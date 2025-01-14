import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
from sklearn.preprocessing import MinMaxScaler
from stock_fetcher import fetch_stock_data

def train_model(ticker="AAPL", model_path="stock_model.h5", scaler_path="scaler.pkl"):
    """
    Train an LSTM model on stock price data.
    """
    # ✅ Fetch stock data dynamically
    data = fetch_stock_data(ticker)

    if data is None or data.empty:
        print(f"Error: No data found for {ticker}!")
        return

    # ✅ Ensure data is numerical
    data = data[['Close']].astype(float)

    # ✅ Data Preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # ✅ Prepare training data
    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):  # Use last 60 days for prediction
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # ✅ Define LSTM Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    # ✅ Compile & Train
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=10)

    # ✅ Save model & scaler
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"✅ Model trained and saved at {model_path}")

if __name__ == "__main__":
    train_model("AAPL")  # ✅ Correct: Fetches data instead of passing a string

