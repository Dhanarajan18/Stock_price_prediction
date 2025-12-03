"""
Command-Line Interface for stock price prediction.
Provides an interactive interface for users to get stock forecasts.
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Optional

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import StockDataFetcher
from preprocess import DataPreprocessor
from features import FeatureEngineer
from train_model import train_models
from predict import StockPredictor
from evaluate import ModelEvaluator
from config import FORECAST_DAYS, CLOSE_COLUMN, DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockPredictionCLI:
    """
    Command-line interface for stock price prediction.
    """
    
    def __init__(self):
        """Initialize the CLI."""
        self.fetcher = StockDataFetcher()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.predictor = StockPredictor()
        self.evaluator = ModelEvaluator()
    
    def fetch_and_prepare_data(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """
        Fetch and prepare data for a stock symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data
            
        Returns:
            Prepared DataFrame or None
        """
        print(f"\n{'='*60}")
        print(f"Fetching data for {symbol}...")
        print(f"{'='*60}\n")
        
        # Fetch data
        df = self.fetcher.fetch_historical_data(symbol, days=days, save_to_csv=True)
        
        if df is None or df.empty:
            print(f"❌ Failed to fetch data for {symbol}")
            return None
        
        print(f"✅ Fetched {len(df)} records")
        print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"   Latest close price: ₹{df[CLOSE_COLUMN].iloc[-1]:.2f}")
        
        return df
    
    def prepare_for_training(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepare data for model training.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Prepared DataFrame or None
        """
        print(f"\n{'='*60}")
        print("Preprocessing data...")
        print(f"{'='*60}\n")
        
        # Clean data
        df_clean = self.preprocessor.clean_data(df)
        print(f"✅ Data cleaned: {len(df_clean)} records")
        
        # Check if we have enough data
        if len(df_clean) < 100:
            print(f"\n⚠️  Warning: Only {len(df_clean)} records available.")
            print(f"   Recommended: At least 100 records for reliable predictions.")
            print(f"   Try increasing --history parameter (e.g., --history 365)\n")
        
        # Add technical indicators
        df_features = self.feature_engineer.add_all_features(df_clean)
        print(f"✅ Technical indicators added: {len(df_features.columns)} features")
        print(f"   Records after feature engineering: {len(df_features)}")
        
        # Check if we have any data left after feature engineering
        if len(df_features) == 0:
            print(f"\n❌ Error: No data remaining after feature engineering!")
            print(f"   Technical indicators require historical windows (e.g., MA50 needs 50 days).")
            print(f"   Solution: Fetch more historical data.")
            print(f"   Try: --history 365 (for 1 year of data)\n")
            return None
        
        if len(df_features) < 60:
            print(f"\n⚠️  Warning: Only {len(df_features)} records after feature engineering.")
            print(f"   LSTM model requires at least 60 records.")
            print(f"   You may only be able to train ARIMA model.\n")
        
        # Normalize data
        try:
            df_normalized, scaler = self.preprocessor.normalize_data(df_features, fit=True)
            print(f"✅ Data normalized")
        except Exception as e:
            print(f"\n❌ Error during normalization: {str(e)}")
            print(f"   This usually happens when there's insufficient data.")
            return None
        
        # Save scaler
        self.preprocessor.save_scaler()
        print(f"✅ Scaler saved")
        
        return df_normalized
    
    def train_models_for_stock(self, df: pd.DataFrame, symbol: str, original_df: Optional[pd.DataFrame] = None) -> bool:
        """
        Train models for a stock.
        
        Args:
            df: Prepared DataFrame (normalized)
            symbol: Stock symbol
            original_df: Original DataFrame with date information
            
        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"Training models for {symbol}...")
        print(f"{'='*60}\n")
        
        try:
            # Train models
            arima_model, lstm_model = train_models(
                df,
                target_column=CLOSE_COLUMN,
                train_arima=True,
                train_lstm=True,
                auto_tune_arima=False
            )
            
            if arima_model is None and lstm_model is None:
                print("❌ Failed to train models")
                return False
            
            if arima_model is not None:
                print("✅ ARIMA model trained and saved")
            
            if lstm_model is not None:
                print("✅ LSTM model trained and saved")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            print(f"❌ Error training models: {str(e)}")
            return False
    
    def make_forecast(
        self,
        symbol: str,
        days: int = FORECAST_DAYS,
        model_type: str = 'both'
    ) -> Optional[pd.DataFrame]:
        """
        Make forecast for a stock.
        
        Args:
            symbol: Stock symbol
            days: Number of days to forecast
            model_type: Model to use ('arima', 'lstm', or 'both')
            
        Returns:
            DataFrame with predictions or None
        """
        print(f"\n{'='*60}")
        print(f"Making {days}-day forecast for {symbol}...")
        print(f"{'='*60}\n")
        
        # Load data
        csv_path = os.path.join(DATA_DIR, f"{symbol}_historical.csv")
        
        if not os.path.exists(csv_path):
            print(f"❌ No data found for {symbol}. Please fetch data first.")
            return None
        
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Load models and scaler
        print("Loading models and scaler...")
        
        if model_type in ['arima', 'both']:
            if not self.predictor.load_arima_model():
                print("⚠️  ARIMA model not available")
        
        if model_type in ['lstm', 'both']:
            if not self.predictor.load_lstm_model():
                print("⚠️  LSTM model not available")
        
        if not self.predictor.load_scaler():
            print("⚠️  Scaler not available")
        
        # Prepare data for prediction
        self.preprocessor.load_scaler()
        df_clean = self.preprocessor.clean_data(df)
        df_features = self.feature_engineer.add_all_features(df_clean)
        df_normalized, _ = self.preprocessor.normalize_data(df_features, fit=False)
        
        # Get last date
        last_date = df_clean['Date'].max()
        
        # Make predictions
        predictions_df = self.predictor.predict_with_dates(
            df_normalized,
            last_date,
            steps=days,
            model_type=model_type
        )
        
        return predictions_df
    
    def display_forecast(self, forecast_df: pd.DataFrame, symbol: str) -> None:
        """
        Display forecast results.
        
        Args:
            forecast_df: DataFrame with predictions
            symbol: Stock symbol
        """
        print(f"\n{'='*60}")
        print(f"📈 {symbol} - {FORECAST_DAYS}-Day Price Forecast")
        print(f"{'='*60}\n")
        
        # Try to fetch current/latest actual price for comparison
        try:
            latest_price = self.fetcher.get_latest_price(symbol)
            if latest_price:
                print(f"  💰 Current Market Price: ₹{latest_price:,.2f}")
                print(f"  📅 As of: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                
                # Compare with first prediction
                if 'ARIMA_Prediction' in forecast_df.columns:
                    arima_diff = ((forecast_df.iloc[0]['ARIMA_Prediction'] - latest_price) / latest_price) * 100
                    print(f"  📊 ARIMA vs Current: {arima_diff:+.2f}%")
                
                if 'LSTM_Prediction' in forecast_df.columns:
                    lstm_diff = ((forecast_df.iloc[0]['LSTM_Prediction'] - latest_price) / latest_price) * 100
                    print(f"  📊 LSTM vs Current: {lstm_diff:+.2f}%")
                
                print()
        except:
            pass
        
        # Display predictions
        for idx, row in forecast_df.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d (%A)')
            print(f"  {date_str}")
            
            if 'ARIMA_Prediction' in row:
                print(f"    ARIMA:  ₹{row['ARIMA_Prediction']:,.2f}")
            
            if 'LSTM_Prediction' in row:
                print(f"    LSTM:   ₹{row['LSTM_Prediction']:,.2f}")
            
            print()
        
        # Calculate average predictions
        if 'ARIMA_Prediction' in forecast_df.columns and 'LSTM_Prediction' in forecast_df.columns:
            forecast_df['Ensemble'] = (forecast_df['ARIMA_Prediction'] + forecast_df['LSTM_Prediction']) / 2
            print(f"\n  📊 Ensemble Forecast (Average):")
            for idx, row in forecast_df.iterrows():
                date_str = row['Date'].strftime('%Y-%m-%d')
                print(f"    {date_str}: ₹{row['Ensemble']:,.2f}")
        
        # Add accuracy note
        print(f"\n  ⚠️  Note: Stock predictions are estimates based on historical patterns.")
        print(f"     Actual prices may vary due to market conditions, news, and events.")
        print(f"     For best accuracy, retrain models weekly with fresh data.")
        
        print(f"\n{'='*60}\n")
    
    def run_interactive(self) -> None:
        """Run interactive CLI mode."""
        print("\n" + "="*60)
        print("📈 Stock Price Prediction System")
        print("   Indian Stock Market Forecasting")
        print("="*60)
        
        while True:
            print("\nOptions:")
            print("  1. Fetch data and train models")
            print("  2. Make forecast (using existing models)")
            print("  3. Full pipeline (fetch, train, forecast)")
            print("  4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                symbol = input("Enter stock symbol (e.g., RELIANCE, TCS): ").strip().upper()
                days = input(f"Enter number of days of history (default 365): ").strip()
                try:
                    days = int(days) if days else 365
                    if days < 100:
                        print(f"\n⚠️  Warning: {days} days may be insufficient.")
                        print(f"   Recommended minimum: 100 days")
                        print(f"   For best results: 365+ days\n")
                        proceed = input("Continue anyway? (y/n): ").strip().lower()
                        if proceed != 'y':
                            continue
                except ValueError:
                    print("❌ Invalid number. Using default 365 days.")
                    days = 365
                
                # Fetch data
                df_raw = self.fetch_and_prepare_data(symbol, days)
                if df_raw is None:
                    continue
                
                # Prepare data
                df_prepared = self.prepare_for_training(df_raw)
                if df_prepared is None:
                    print("\n⚠️  Data preparation failed. Please try again with more historical data.")
                    continue
                
                # Train models (df_raw is already saved in fetch_and_prepare_data)
                self.train_models_for_stock(df_prepared, symbol, df_raw)
            
            elif choice == '2':
                symbol = input("Enter stock symbol: ").strip().upper()
                days = input(f"Enter forecast days (default {FORECAST_DAYS}): ").strip()
                days = int(days) if days else FORECAST_DAYS
                
                model_type = input("Model type (arima/lstm/both, default both): ").strip().lower()
                model_type = model_type if model_type in ['arima', 'lstm', 'both'] else 'both'
                
                # Make forecast
                forecast_df = self.make_forecast(symbol, days, model_type)
                
                if forecast_df is not None and not forecast_df.empty:
                    self.display_forecast(forecast_df, symbol)
            
            elif choice == '3':
                symbol = input("Enter stock symbol (e.g., RELIANCE, TCS): ").strip().upper()
                
                # Full pipeline
                df_raw = self.fetch_and_prepare_data(symbol, days=365)
                if df_raw is None:
                    continue
                
                df_prepared = self.prepare_for_training(df_raw)
                if df_prepared is None:
                    continue
                
                if self.train_models_for_stock(df_prepared, symbol, df_raw):
                    forecast_df = self.make_forecast(symbol, FORECAST_DAYS, 'both')
                    if forecast_df is not None:
                        self.display_forecast(forecast_df, symbol)
            
            elif choice == '4':
                print("\n👋 Thank you for using Stock Price Prediction System!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Stock Price Prediction System for Indian Stock Market'
    )
    
    parser.add_argument(
        'symbol',
        nargs='?',
        help='Stock symbol (e.g., RELIANCE, TCS, INFY)'
    )
    
    parser.add_argument(
        '-d', '--days',
        type=int,
        default=FORECAST_DAYS,
        help=f'Number of days to forecast (default: {FORECAST_DAYS})'
    )
    
    parser.add_argument(
        '-m', '--model',
        choices=['arima', 'lstm', 'both'],
        default='both',
        help='Model to use for prediction (default: both)'
    )
    
    parser.add_argument(
        '-t', '--train',
        action='store_true',
        help='Train models before prediction'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--history',
        type=int,
        default=365,
        help='Days of historical data to fetch (default: 365)'
    )
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = StockPredictionCLI()
    
    # Interactive mode
    if args.interactive or args.symbol is None:
        cli.run_interactive()
        return
    
    # Command-line mode
    symbol = args.symbol.upper()
    
    if args.train:
        # Full pipeline: fetch, prepare, train
        df_raw = cli.fetch_and_prepare_data(symbol, args.history)
        if df_raw is not None:
            df_prepared = cli.prepare_for_training(df_raw)
            if df_prepared is not None:
                success = cli.train_models_for_stock(df_prepared, symbol, df_raw)
                if not success:
                    print("\n❌ Training failed. Cannot proceed with forecast.")
                    return
    
    # Make forecast
    forecast_df = cli.make_forecast(symbol, args.days, args.model)
    
    if forecast_df is not None and not forecast_df.empty:
        cli.display_forecast(forecast_df, symbol)
    else:
        print(f"\n❌ Unable to generate forecast for {symbol}")
        print("   Try running with --train flag to train models first")


if __name__ == "__main__":
    main()
