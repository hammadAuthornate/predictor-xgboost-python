# src/xgboost_forecasting.py
import os
import xgboost as xgb
import pandas as pd
from datetime import timedelta

class XGBoostForecasting:
    def __init__(self, data, date_column, target_column, config):
        self.data = data
        self.date_column = date_column
        self.target_column = target_column
        self.config = config

    def preprocess_data(self):
        self.data = self.data.rename(columns={self.date_column: "ds", self.target_column: "y"})
        self.data["ds"] = pd.to_datetime(self.data["ds"])
        self.data["day"] = self.data["ds"].dt.day
        self.data["month"] = self.data["ds"].dt.month
        self.data["weekday"] = self.data["ds"].dt.weekday

    def prepare_features(self, training_period=730):
        last_date = self.data["ds"].max()
        start_date = last_date - timedelta(days=training_period)
        training_data = self.data[self.data["ds"] >= start_date]
        features = training_data[["day", "month", "weekday"]]
        target = training_data["y"]
        return features, target, training_data

    def train_model(self, training_period=730):
        features, target, _ = self.prepare_features(training_period)
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror", n_estimators=1000, learning_rate=0.01, max_depth=6, subsample=0.8, colsample_bytree=0.8
        )
        self.model.fit(features, target)

    def forecast(self, future_periods=180):
        last_date = self.data["ds"].max()
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, future_periods + 1)]
        future_features = pd.DataFrame(
            {"day": [d.day for d in forecast_dates], "month": [d.month for d in forecast_dates], "weekday": [d.weekday() for d in forecast_dates]}
        )
        forecast_values = self.model.predict(future_features)
        last_training_value = self.data["y"].iloc[-1]
        first_forecast_value = forecast_values[0]
        adjustment_factor = last_training_value - first_forecast_value
        forecast_values += adjustment_factor
        return pd.DataFrame({"ds": forecast_dates, "yhat": forecast_values})

    def save_forecast(self, forecast, coin_name):
        forecast_dir = self.config["paths"]["forecast_dir"]
        os.makedirs(forecast_dir, exist_ok=True)
        forecast_path = os.path.join(forecast_dir, f"{coin_name}_Forecast.csv")
        forecast.to_csv(forecast_path, index=False)
        print(f"Forecast saved at: {forecast_path}")

    def get_forecast_csv_data(self, forecast, coin_name):
        forecast_dir = self.config["paths"]["forecast_dir"]
        forecast_path = os.path.join(forecast_dir, f"{coin_name}_Forecast.csv")
        if os.path.exists(forecast_path):
            return open(forecast_path, "rb").read()
        else:
            return None