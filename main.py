# src/main.py
import os
import io
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse

from binance_ingestion import BinanceIngestionData
from data_processing import DataProcessing
from xgboost_forecasting import XGBoostForecasting
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configuration (replace with your actual config loading if needed)
config = {
    "symbols": {"currencies": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOGEUSDT", "LINKUSDT"]},  # Example symbols
    "paths": {"artifacts_dir": "/tmp/artifacts", "processed_dir": "/tmp/processed", "foresast_dir": "/tmp/forecast"},
    "forecast_period": 180,
}

# Ensure directories exist (Vercel's /tmp is writable)
for path in config["paths"].values():
    os.makedirs(path, exist_ok=True)


@app.get("/forecast/{symbol}")
async def forecast_symbol(symbol: str):
    """
    Endpoint to trigger the forecasting pipeline for a specific symbol and return the forecast as a CSV.
    """
    if symbol not in config["symbols"]["currencies"]:
        return {"error": "Invalid symbol"}

    try:
        # Stage 1: Data Ingestion
        print("Data Ingestion started")
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        output_dir = config["paths"]["artifacts_dir"]

        binance_data = BinanceIngestionData(symbol, "1d", start_date, end_date, output_dir)
        raw_data = binance_data.fetch_data()
        processed_ingestion_data = binance_data.process_data(raw_data)
        binance_data.save_to_csv(processed_ingestion_data)

        # Stage 2: Data Processing
        print("Data Processing started")
        csv_path = os.path.join(output_dir, f"{symbol}_2Y.csv")
        data_processor = DataProcessing(csv_path)
        processed_data = data_processor.process_data()

        processed_file_path = os.path.join(config["paths"]["processed_dir"], f"{symbol}_Featured.csv")
        processed_data.to_csv(processed_file_path, index=False)

        # Stage 3: Model Training and Forecasting
        print("Model Training started")
        xgboost_forecasting = XGBoostForecasting(processed_data, "ds", "y", config)
        xgboost_forecasting.preprocess_data()
        xgboost_forecasting.train_model(training_period=730)
        forecast = xgboost_forecasting.forecast(future_periods=config["forecast_period"])
        xgboost_forecasting.save_forecast(forecast, symbol)

        # Create JSON response
        historical_data = processed_data.set_index("ds")["y"].to_dict()
        prediction_data = forecast.set_index("ds")["yhat"].to_dict()

        # Convert datetime keys to string (YYYY-MM-DD) for JSON serialization
        historical_data_str = {k.strftime('%Y-%m-%d'): v for k, v in historical_data.items()}
        prediction_data_str = {k.strftime('%Y-%m-%d'): v for k, v in prediction_data.items()}

        response_data = {
            "historical": historical_data_str,
            "prediction": prediction_data_str,
        }

        return JSONResponse(content=response_data)

        # # Return the forecast as a CSV file
        # csv_data = xgboost_forecasting.get_forecast_csv_data(forecast, symbol)

        # if csv_data:
        #     return Response(content=csv_data, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={symbol}_Forecast.csv"})
        # else:
        #     return {"error": "Forecast file not found."}

    except Exception as e:
        return {"error": str(e)}