# src/yfinance_ingestion.py
import pandas as pd
import yfinance as yf
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class YFinanceIngestionData:
    def __init__(self, symbol, start_date, end_date, output_dir):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir

    def fetch_data(self):
        try:
            data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            if data is None or data.empty:
                raise ValueError(f"No data found for symbol: {self.symbol}")
            return data
        except Exception as e:
            raise Exception(f"Failed to fetch data for {self.symbol} using yfinance: {e}")

    def process_data(self, data):
        df = data.copy()
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'Open Time', 'Close': 'Close', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Volume': 'Volume'}, inplace=True)
        return df

    def save_to_csv(self, df):
        file_path = f"{self.output_dir}/{self.symbol}_2Y.csv"
        df.to_csv(file_path, index=False)
        print(f"Data for {self.symbol} saved to {file_path}")