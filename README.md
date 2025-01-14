# **📈 Stock Market Predictor**

A **PyQt6-based desktop application** that predicts stock prices using an **LSTM (Long Short-Term Memory) model** trained on historical stock data. It supports **multi-stock comparisons** and offers **real-time stock price updates**.

---

## ** Features**
✅ **Stock Price Prediction** using a trained LSTM model
✅ **Multi-Stock Comparison** (enter multiple stock symbols)
✅ **Live Stock Price Updates** (fetches new prices every 10 seconds)
✅ **Static vs. Live Mode Toggle**
✅ **Matplotlib Graph Integration** (visualize stock trends & predictions)
✅ **Modern PyQt6 UI**

---

## **📌 Installation**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/GauthamSathishD/StockPredictor.git
cd StockPredictor
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```
If you don’t have `requirements.txt`, install manually:
```bash
pip install yfinance PyQt6 matplotlib joblib numpy pandas tensorflow requests
```

### **3️⃣ Train the Model**
Run the following command to fetch stock data and train the LSTM model:
```bash
python predictor.py
```

### **4️⃣ Run the Application**
```bash
python stock_predictor.py
```

---

## **🛠️ Project Structure**
```
StockPredictor/
│── stock_predictor.py   # Main PyQt6 GUI Application
│── predictor.py         # LSTM Model Training & Prediction
│── stock_fetcher.py     # Fetch Historical Stock Data (Yahoo Finance API)
│── requirements.txt     # Dependencies
│── README.md            # Project Documentation
```

---

## **📊 How It Works**
1. **User enters stock symbols** (e.g., `AAPL, TSLA, GOOGL`).
2. **Stock data is fetched** using Yahoo Finance API.
3. **Pre-trained LSTM model predicts stock prices.**
4. **Graph displays actual stock prices & predictions.**
5. **Live mode updates stock prices every 10 seconds.**

---

## **📈 Example Usage**
### **🔹 Static Mode (Historical Data)**
- Enter a stock ticker (`AAPL`) and click `Predict`.
- The graph displays historical data & predicted price.

### **🔹 Live Mode (Real-Time Updates)**
- Select **Live Updates** from the dropdown.
- The app fetches real-time stock prices every **10 seconds**.

---


## **👨‍💻 Author**
Developed by **Gautham Sathish**  



