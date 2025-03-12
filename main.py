# src/main.py
import os
import io
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse

from yfinance_ingestion import YFinanceIngestionData
from data_processing import DataProcessing
from xgboost_forecasting import XGBoostForecasting
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configuration (replace with your actual config loading if needed)
config = {
    "paths": {"artifacts_dir": "/tmp/artifacts", "processed_dir": "/tmp/processed", "forecast_dir": "/tmp/forecast"},
}

# Ensure directories exist (Vercel's /tmp is writable)
for path in config["paths"].values():
    os.makedirs(path, exist_ok=True)


@app.get("/forecast/{symbol}")
async def forecast_symbol(symbol: str):
    """
    Endpoint to trigger the forecasting pipeline for a specific symbol and return the forecast as JSON.
    """
    try:
        # Stage 1: Data Ingestion
        print("Data Ingestion started")
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=1800)).strftime("%Y-%m-%d") # default for 5 years

        output_dir = config["paths"]["artifacts_dir"]

        yfinance_data = YFinanceIngestionData(symbol, start_date, end_date, output_dir)
        raw_data = yfinance_data.fetch_data()
        processed_ingestion_data = yfinance_data.process_data(raw_data)
        yfinance_data.save_to_csv(processed_ingestion_data)

        # Stage 2: Data Processing
        print("Data Processing started")
        csv_path = os.path.join(output_dir, f"{symbol}_2Y.csv")
        data_processor = DataProcessing(csv_path)
        processed_data = data_processor.process_data()

        processed_file_path = os.path.join(config["paths"]["processed_dir"], f"{symbol}_Featured.csv")
        processed_data.to_csv(processed_file_path, index=False)

        # Stage 3: Model Training and Forecasting
        print("Model Training started")
        xgboost_forecasting = XGBoostForecasting(processed_data, "Open Time", "y", config)
        xgboost_forecasting.preprocess_data()
        xgboost_forecasting.train_model(training_period=730)

        # Dynamic forecast period
        historical_days = (processed_data['Open Time'].max() - processed_data['Open Time'].min()).days
        # Calculate the forecast period based on the available data duration
        if historical_days >= 365 * 5:  # 5 years
            forecast_period = 365 * 2  # 2 years
        elif historical_days >= 365 * 2:  # 1 year
            forecast_period = 365  # 1 year
        elif historical_days >= 365:  # 1 year
            forecast_period = 365  # 1 year
        elif historical_days >= 180:  # 6 months
            forecast_period = 180 # 6 months
        elif historical_days >= 90:  # 3 months
            forecast_period = 90 # 3 months
        elif historical_days >= 30:  # 1 month
            forecast_period = 30 # 1 month
        elif historical_days >= 14:  # 2 weeks
            forecast_period = 14 # 2 weeks
        elif historical_days >= 7:  # 1 week
            forecast_period = 7  # 1 week
        else:
            forecast_period = 1  # 1 day

        forecast = xgboost_forecasting.forecast(future_periods=forecast_period)
        xgboost_forecasting.save_forecast(forecast, symbol)

        # Create JSON response
        historical_data = processed_data.set_index("Open Time")["y"].to_dict()
        prediction_data = forecast.set_index("ds")["yhat"].to_dict()

        # Convert datetime keys to string (YYYY-MM-DD) for JSON serialization
        historical_data_str = {k.strftime('%Y-%m-%d'): v for k, v in historical_data.items() if pd.notna(k)} #Filter out NaT
        prediction_data_str = {k.strftime('%Y-%m-%d'): v for k, v in prediction_data.items() if pd.notna(k)} #Filter out NaT

        response_data = {
            "symbol": symbol,
            "predictedAt": datetime.now().strftime("%Y-%m-%d"),
            "historical": historical_data_str,
            "prediction": prediction_data_str,
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        return {"error": str(e)}