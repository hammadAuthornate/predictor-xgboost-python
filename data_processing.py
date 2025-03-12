import pandas as pd
import numpy as np

class DataProcessing:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path, parse_dates=["Open Time"], index_col="Open Time")
        self.df = self.df.apply(pd.to_numeric, errors='coerce')

    def add_features(self):
        self.df = self.df.rename(columns={"Close": "y"})
        self.df = self.df.drop(columns=["Ignore"], errors="ignore")
        self.df["High_Low_Diff"] = self.df["High"] - self.df["Low"]
        self.df["Open_Close_Diff"] = self.df["Open"] - self.df["y"]
        self.df["Average_Price"] = (self.df["High"] + self.df["Low"] + self.df["y"]) / 3
        self.df["Volume"] = self.df["Volume"]  # Keep the volume column as is.

    def clean_data(self):
        # Remove rows with NaN or infinity in the target column (y)
        self.df = self.df.replace([np.inf, -np.inf], np.nan)  # Replace infinity with NaN
        self.df = self.df.dropna(subset=["y"])  # Drop rows where 'y' is NaN
        return self.df

    def process_data(self):
        self.add_features()
        self.clean_data()  # Clean the data before processing
        self.df.index = pd.to_datetime(self.df.index)
        self.df.reset_index(inplace=True)
        return self.df