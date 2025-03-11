import pandas as pd

class DataProcessing:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path, parse_dates=["Open Time"])

    def add_features(self):
        self.df = self.df.rename(columns={"Open Time": "ds", "Close": "y"})
        self.df = self.df.drop(columns=["Ignore"], errors="ignore")
        self.df["High_Low_Diff"] = self.df["High"] - self.df["Low"]
        self.df["Open_Close_Diff"] = self.df["Open"] - self.df["y"]
        self.df["Average_Price"] = (self.df["High"] + self.df["Low"] + self.df["y"]) / 3
        self.df["Volume_Weighted_Price"] = self.df["Quote Asset Volume"] / self.df["Volume"]

    def process_data(self):
        self.add_features()
        self.df["ds"] = pd.to_datetime(self.df["ds"])
        return self.df
