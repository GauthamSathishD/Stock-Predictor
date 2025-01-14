import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, period="5y", save_to_csv=True):
    """
    Fetch historical stock price data from Yahoo Finance.
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)

    if data.empty:
        raise ValueError(f"Error: No data found for {ticker}!")

    # ✅ Ensure "Close" column is numerical
    data = data[['Close']].astype(float)

    if save_to_csv:
        csv_filename = f"{ticker}.csv"
        data.to_csv(csv_filename)
        print(f"✅ Stock data saved to {csv_filename}")

    return data

