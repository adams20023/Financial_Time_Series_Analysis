import yfinance as yf
import pandas as pd

def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(f'data/{ticker}.csv')
    print(f'Data for {ticker} downloaded successfully.')

if __name__ == "__main__":
    ticker = 'AAPL'  # Apple Inc.
    start_date = '2010-01-01'
    end_date = '2023-10-01'
    download_data(ticker, start_date, end_date)

