"""
Stock Price Prediction System
Indian Stock Market Forecasting using ARIMA and LSTM

This package provides a complete solution for predicting stock prices
in the Indian stock market using both traditional statistical methods (ARIMA)
and deep learning (LSTM) approaches.

Author: Dhanarajan K
"""

__version__ = "1.0.0"
__author__ = "Dhanarajan K"

# Import main classes for easier access
from .data_fetcher import StockDataFetcher, fetch_stock_data
from .preprocess import DataPreprocessor, preprocess_data
from .features import FeatureEngineer, create_features
from .train_model import ARIMATrainer, LSTMTrainer, train_models
from .predict import StockPredictor, make_predictions
from .evaluate import ModelEvaluator, evaluate_model, evaluate_predictions

__all__ = [
    'StockDataFetcher',
    'fetch_stock_data',
    'DataPreprocessor',
    'preprocess_data',
    'FeatureEngineer',
    'create_features',
    'ARIMATrainer',
    'LSTMTrainer',
    'train_models',
    'StockPredictor',
    'make_predictions',
    'ModelEvaluator',
    'evaluate_model',
    'evaluate_predictions',
]
