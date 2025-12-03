"""
Configuration file for stock price prediction project.
Contains all constants, model parameters, and settings.
"""

import os

# Project directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Data fetching settings
DEFAULT_DAYS_HISTORY = 365 * 3  # 3 years of historical data
MIN_DATA_POINTS = 100  # Minimum data points required for training
MIN_DATA_POINTS_LSTM = 60  # Minimum for LSTM (sequence length)
RECOMMENDED_DAYS_HISTORY = 365  # Recommended minimum for good predictions

# Feature engineering settings
MOVING_AVERAGES = [10, 20, 50]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# ARIMA model settings
ARIMA_ORDER = (5, 1, 0)  # (p, d, q) - can be tuned using auto_arima
ARIMA_SEASONAL_ORDER = (0, 0, 0, 0)

# LSTM model settings
LSTM_SEQUENCE_LENGTH = 60  # Use 60 days of data to predict next day
LSTM_UNITS = [50, 50]  # Two LSTM layers with 50 units each
LSTM_DROPOUT = 0.2
LSTM_BATCH_SIZE = 32
LSTM_EPOCHS = 50
LSTM_VALIDATION_SPLIT = 0.2
LSTM_LEARNING_RATE = 0.001

# Train-test split
TRAIN_TEST_SPLIT = 0.8  # 80% training, 20% testing

# Prediction settings
FORECAST_DAYS = 5  # Number of days to forecast

# Model file names
ARIMA_MODEL_FILE = 'arima_model.pkl'
LSTM_MODEL_FILE = 'lstm_model.h5'
SCALER_FILE = 'scaler.pkl'

# Model maintenance
RECOMMENDED_RETRAIN_DAYS = 7  # Retrain models weekly for best accuracy
MODEL_ACCURACY_WARNING_THRESHOLD = 5.0  # Warn if prediction differs > 5% from current price

# Column names
DATE_COLUMN = 'Date'
CLOSE_COLUMN = 'Close'
OPEN_COLUMN = 'Open'
HIGH_COLUMN = 'High'
LOW_COLUMN = 'Low'
VOLUME_COLUMN = 'Volume'

# Logging
LOG_LEVEL = 'INFO'
