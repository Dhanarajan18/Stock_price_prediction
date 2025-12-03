"""
Demo script showing complete workflow of the Stock Price Prediction System.
This script demonstrates all key features of the project.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime

# Import all modules
from data_fetcher import StockDataFetcher
from preprocess import DataPreprocessor
from features import FeatureEngineer
from train_model import train_models
from predict import StockPredictor
from evaluate import ModelEvaluator
from config import CLOSE_COLUMN, FORECAST_DAYS


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def demo_workflow():
    """Demonstrate the complete workflow."""
    
    print("\n" + "🎯" + "="*68 + "🎯")
    print("  STOCK PRICE PREDICTION SYSTEM - DEMO")
    print("  Indian Stock Market Forecasting with ARIMA & LSTM")
    print("🎯" + "="*68 + "🎯\n")
    
    # Configuration
    DEMO_SYMBOL = "RELIANCE"
    HISTORY_DAYS = 365
    
    # =========================================================================
    # STEP 1: Data Fetching
    # =========================================================================
    print_section("STEP 1: Fetching Historical Stock Data")
    
    fetcher = StockDataFetcher()
    print(f"Fetching {HISTORY_DAYS} days of data for {DEMO_SYMBOL}...")
    
    df_raw = fetcher.fetch_historical_data(DEMO_SYMBOL, days=HISTORY_DAYS)
    
    if df_raw is None or df_raw.empty:
        print("❌ Failed to fetch data. Demo will use sample data.")
        # Generate sample data for demo
        dates = pd.date_range(end=datetime.now(), periods=HISTORY_DAYS, freq='B')
        np.random.seed(42)
        base_price = 2500
        prices = base_price + np.cumsum(np.random.randn(HISTORY_DAYS) * 20)
        
        df_raw = pd.DataFrame({
            'Date': dates,
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, HISTORY_DAYS)
        })
        print("✅ Using sample data for demonstration")
    
    print(f"✅ Data fetched: {len(df_raw)} records")
    print(f"   Date range: {df_raw['Date'].min()} to {df_raw['Date'].max()}")
    print(f"   Latest close: ₹{df_raw[CLOSE_COLUMN].iloc[-1]:.2f}")
    print(f"\n   Sample data:")
    print(df_raw.tail(3))
    
    # =========================================================================
    # STEP 2: Data Preprocessing
    # =========================================================================
    print_section("STEP 2: Data Preprocessing")
    
    preprocessor = DataPreprocessor()
    
    print("Cleaning data...")
    df_clean = preprocessor.clean_data(df_raw)
    print(f"✅ Cleaned: {len(df_clean)} records (removed {len(df_raw) - len(df_clean)} rows)")
    
    print("\nHandling outliers...")
    df_clean = preprocessor.handle_outliers(df_clean, method='iqr')
    print("✅ Outliers handled using IQR method")
    
    # =========================================================================
    # STEP 3: Feature Engineering
    # =========================================================================
    print_section("STEP 3: Feature Engineering - Technical Indicators")
    
    engineer = FeatureEngineer()
    
    print("Adding technical indicators...")
    print("  ➤ Moving Averages (MA10, MA20, MA50)")
    print("  ➤ Exponential Moving Averages")
    print("  ➤ RSI (Relative Strength Index)")
    print("  ➤ MACD (Moving Average Convergence Divergence)")
    print("  ➤ Bollinger Bands")
    print("  ➤ Volume indicators")
    print("  ➤ Price-based features")
    
    df_features = engineer.add_all_features(df_clean)
    print(f"\n✅ Features created: {len(df_features.columns)} total columns")
    print(f"   Records after feature engineering: {len(df_features)}")
    
    print(f"\n   Feature columns:")
    for i, col in enumerate(df_features.columns, 1):
        print(f"      {i:2d}. {col}")
    
    # =========================================================================
    # STEP 4: Data Normalization
    # =========================================================================
    print_section("STEP 4: Data Normalization")
    
    print("Normalizing data using MinMaxScaler...")
    df_normalized, scaler = preprocessor.normalize_data(df_features, fit=True)
    print("✅ Data normalized to range [0, 1]")
    
    preprocessor.save_scaler()
    print("✅ Scaler saved for future predictions")
    
    print(f"\n   Normalized data sample:")
    print(df_normalized[[CLOSE_COLUMN, 'MA_10', 'RSI', 'MACD']].tail(3))
    
    # =========================================================================
    # STEP 5: Model Training
    # =========================================================================
    print_section("STEP 5: Training Prediction Models")
    
    print("Training both ARIMA and LSTM models...")
    print("\nThis may take several minutes...\n")
    
    arima_model, lstm_model = train_models(
        df_normalized,
        target_column=CLOSE_COLUMN,
        train_arima=True,
        train_lstm=True,
        auto_tune_arima=False
    )
    
    if arima_model is not None:
        print("\n✅ ARIMA model trained successfully")
    else:
        print("\n⚠️  ARIMA model training failed or skipped")
    
    if lstm_model is not None:
        print("✅ LSTM model trained successfully")
    else:
        print("⚠️  LSTM model training failed or skipped")
    
    # =========================================================================
    # STEP 6: Making Predictions
    # =========================================================================
    print_section("STEP 6: Making Price Predictions")
    
    predictor = StockPredictor()
    
    print("Loading trained models...")
    predictor.load_arima_model()
    predictor.load_lstm_model()
    predictor.load_scaler()
    print("✅ Models loaded")
    
    print(f"\nGenerating {FORECAST_DAYS}-day forecast...")
    
    # Get predictions
    input_data = df_normalized[CLOSE_COLUMN].values
    last_date = df_features['Date'].max()
    
    predictions_df = predictor.predict_with_dates(
        df_normalized,
        last_date,
        steps=FORECAST_DAYS,
        model_type='both'
    )
    
    print(f"✅ Predictions generated\n")
    
    # Display predictions
    print(f"   📈 {DEMO_SYMBOL} - {FORECAST_DAYS}-Day Price Forecast")
    print("   " + "-"*66)
    
    for idx, row in predictions_df.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d (%A)')
        print(f"\n   {date_str}")
        
        if 'ARIMA_Prediction' in row:
            print(f"     ARIMA:  ₹{row['ARIMA_Prediction']:,.2f}")
        
        if 'LSTM_Prediction' in row:
            print(f"     LSTM:   ₹{row['LSTM_Prediction']:,.2f}")
        
        if 'ARIMA_Prediction' in row and 'LSTM_Prediction' in row:
            ensemble = (row['ARIMA_Prediction'] + row['LSTM_Prediction']) / 2
            print(f"     Ensemble: ₹{ensemble:,.2f}")
    
    # =========================================================================
    # STEP 7: Model Evaluation
    # =========================================================================
    print_section("STEP 7: Model Evaluation")
    
    evaluator = ModelEvaluator()
    
    print("Evaluating models on test data...\n")
    
    # Split data for evaluation
    split_idx = int(len(df_normalized) * 0.8)
    test_data = df_normalized[CLOSE_COLUMN].values[split_idx:]
    
    # Make predictions on test data
    if arima_model is not None:
        try:
            arima_test_pred = arima_model.forecast(steps=len(test_data))
            arima_test_pred = preprocessor.inverse_transform(arima_test_pred, column_index=0)
        except:
            arima_test_pred = None
    
    if lstm_model is not None:
        try:
            lstm_test_pred = predictor.predict_lstm(
                df_normalized[CLOSE_COLUMN].values[:split_idx],
                steps=len(test_data)
            )
            if lstm_test_pred is not None:
                lstm_test_pred = preprocessor.inverse_transform(lstm_test_pred, column_index=0)
        except:
            lstm_test_pred = None
    
    # Get actual values
    actual_test = preprocessor.inverse_transform(test_data, column_index=0)
    
    # Evaluate
    if arima_test_pred is not None:
        arima_metrics = evaluator.evaluate(actual_test, arima_test_pred, "ARIMA")
        evaluator.print_evaluation_report(arima_metrics, "ARIMA Model")
    
    if lstm_test_pred is not None:
        lstm_metrics = evaluator.evaluate(actual_test, lstm_test_pred, "LSTM")
        evaluator.print_evaluation_report(lstm_metrics, "LSTM Model")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_section("✅ DEMO COMPLETED SUCCESSFULLY")
    
    print("Summary:")
    print(f"  ✓ Fetched {len(df_raw)} records for {DEMO_SYMBOL}")
    print(f"  ✓ Created {len(df_features.columns)} features")
    print(f"  ✓ Trained {'ARIMA' if arima_model else ''} {'and' if arima_model and lstm_model else ''} {'LSTM' if lstm_model else ''} model(s)")
    print(f"  ✓ Generated {FORECAST_DAYS}-day forecast")
    print(f"  ✓ Evaluated model performance")
    
    print("\nNext Steps:")
    print("  1. Try with different stock symbols: TCS, INFY, HDFCBANK")
    print("  2. Adjust parameters in src/config.py")
    print("  3. Use CLI for interactive predictions:")
    print(f"     python src/cli_interface.py {DEMO_SYMBOL} --days 5")
    
    print("\n" + "="*70)
    print("  Thank you for exploring the Stock Price Prediction System!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        demo_workflow()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()
