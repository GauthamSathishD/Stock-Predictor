import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import yfinance as yf
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox
from PyQt6.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from stock_fetcher import fetch_stock_data

class StockPredictorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Stock Predictor")
        self.setGeometry(100, 100, 800, 600)

        # ✅ Layout & Widgets
        self.layout = QVBoxLayout()

        self.label = QLabel("Enter Stock Symbols (comma-separated):")
        self.layout.addWidget(self.label)

        self.input_field = QLineEdit()
        self.layout.addWidget(self.input_field)

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict_stocks)
        self.layout.addWidget(self.predict_button)

        self.result_label = QLabel("Predictions will appear here")
        self.layout.addWidget(self.result_label)

        # ✅ Timeframe Selector (Static vs. Live Updates)
        self.timeframe_selector = QComboBox()
        self.timeframe_selector.addItems(["Static Data", "Live Updates"])
        self.layout.addWidget(self.timeframe_selector)

        # ✅ Add Matplotlib Figure for Multi-Stock Graph
        self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.setLayout(self.layout)

        # ✅ Set up a timer for real-time updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_live_prices)

    def predict_stocks(self):
        """ Fetch stock data, make predictions, and update graph for multiple stocks. """
        stock_symbols = self.input_field.text().upper().replace(" ", "").split(",")
        if not stock_symbols or stock_symbols == [""]:
            self.result_label.setText("Please enter at least one valid stock symbol!")
            return

        self.predictions = {}  # Store predicted prices
        self.stock_data = {}  # Store stock history

        # ✅ Clear previous graph
        self.ax.clear()

        for stock_symbol in stock_symbols:
            # ✅ Fetch stock data
            data = fetch_stock_data(stock_symbol)
            if data is None or data.empty:
                self.result_label.setText(f"Error: No stock data found for {stock_symbol}!")
                continue

            self.stock_data[stock_symbol] = data  # Save data for live updates

            # ✅ Load pre-trained model and scaler
            try:
                model = tf.keras.models.load_model("stock_model.h5")
                scaler = joblib.load("scaler.pkl")
            except Exception as e:
                self.result_label.setText("Error loading model!")
                print(e)
                return

            # ✅ Prepare last 60 days data for prediction
            last_60_days = data['Close'].values[-60:].reshape(-1, 1)
            last_60_days_scaled = scaler.transform(last_60_days)
            X_test = np.array([last_60_days_scaled])
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            # ✅ Predict next day's price
            predicted_price = model.predict(X_test)
            predicted_price = scaler.inverse_transform(predicted_price)[0][0]

            self.predictions[stock_symbol] = predicted_price

            # ✅ Update Multi-Stock Graph
            self.ax.plot(data.index, data['Close'], label=f"{stock_symbol} (Actual)", linestyle="solid")
            self.ax.scatter(data.index[-1], predicted_price, label=f"{stock_symbol} (Predicted)", marker="o")

        # ✅ Update UI
        self.result_label.setText(
            "Predicted Prices:\n" + "\n".join([f"{stock}: ${price:.2f}" for stock, price in self.predictions.items()])
        )

        # ✅ Finalize graph updates
        self.ax.set_title("Stock Price Trends & Predictions")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Closing Price")
        self.ax.legend()
        self.canvas.draw()

        # ✅ If "Live Updates" is selected, start real-time tracking
        if self.timeframe_selector.currentText() == "Live Updates":
            self.timer.start(10000)  # Update every 10 seconds
        else:
            self.timer.stop()  # Stop live updates for static mode

    def update_live_prices(self):
        """ Fetch latest stock price and update graph in real-time. """
        if not self.stock_data:
            return

        for stock_symbol, data in self.stock_data.items():
            live_data = yf.Ticker(stock_symbol).history(period="1d", interval="1m")
            if live_data.empty:
                continue

            latest_price = live_data['Close'].iloc[-1]  # Get latest closing price
            data.loc[data.index[-1]] = latest_price  # Append to existing data

        self.update_graph()

    def update_graph(self):
        """ Refresh the graph with live stock price updates. """
        self.ax.clear()
        for stock_symbol, data in self.stock_data.items():
            self.ax.plot(data.index, data['Close'], label=f"{stock_symbol} (Updated)", linestyle="solid")
            if stock_symbol in self.predictions:
                self.ax.scatter(data.index[-1], self.predictions[stock_symbol], label=f"{stock_symbol} (Predicted)", marker="o")

        self.ax.set_title("Stock Price Trends & Predictions (Live)")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Closing Price")
        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockPredictorApp()
    window.show()
    sys.exit(app.exec())



