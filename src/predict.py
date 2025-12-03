"""
Prediction module for forecasting stock prices using trained models.
Supports both ARIMA and LSTM models.
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow import keras
except ImportError:
    keras = None

from config import (
    MODELS_DIR, ARIMA_MODEL_FILE, LSTM_MODEL_FILE, SCALER_FILE,
    LSTM_SEQUENCE_LENGTH, FORECAST_DAYS, CLOSE_COLUMN
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockPredictor:
    """
    Makes predictions using trained ARIMA and LSTM models.
    """
    
    def __init__(self):
        """Initialize the stock predictor."""
        self.arima_model = None
        self.lstm_model = None
        self.scaler = None
        self.sequence_length = LSTM_SEQUENCE_LENGTH
    
    def load_arima_model(self, filepath: Optional[str] = None) -> bool:
        """
        Load trained ARIMA model.
        
        Args:
            filepath: Path to model file
            
        Returns:
            True if successful, False otherwise
        """
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, ARIMA_MODEL_FILE)
        
        if not os.path.exists(filepath):
            logger.error(f"ARIMA model not found: {filepath}")
            return False
        
        try:
            import pickle
            with open(filepath, 'rb') as f:
                self.arima_model = pickle.load(f)
            logger.info(f"ARIMA model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading ARIMA model: {str(e)}")
            return False
    
    def load_lstm_model(self, filepath: Optional[str] = None) -> bool:
        """
        Load trained LSTM model.
        
        Args:
            filepath: Path to model file
            
        Returns:
            True if successful, False otherwise
        """
        if keras is None:
            logger.error("TensorFlow/Keras not available")
            return False
        
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, LSTM_MODEL_FILE)
        
        if not os.path.exists(filepath):
            logger.error(f"LSTM model not found: {filepath}")
            return False
        
        try:
            self.lstm_model = keras.models.load_model(filepath)
            logger.info(f"LSTM model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading LSTM model: {str(e)}")
            return False
    
    def load_scaler(self, filepath: Optional[str] = None) -> bool:
        """
        Load data scaler.
        
        Args:
            filepath: Path to scaler file
            
        Returns:
            True if successful, False otherwise
        """
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, SCALER_FILE)
        
        if not os.path.exists(filepath):
            logger.error(f"Scaler not found: {filepath}")
            return False
        
        try:
            import pickle
            with open(filepath, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Scaler loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            return False
    
    def predict_arima(
        self, 
        steps: int = FORECAST_DAYS,
        return_conf_int: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions using ARIMA model.
        
        Args:
            steps: Number of steps to forecast
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        if self.arima_model is None:
            logger.error("ARIMA model not loaded")
            return None, None
        
        try:
            # Forecast
            forecast = self.arima_model.forecast(steps=steps)
            
            if return_conf_int:
                # Get forecast with confidence intervals
                forecast_obj = self.arima_model.get_forecast(steps=steps)
                conf_int = forecast_obj.conf_int()
                return forecast, conf_int
            
            return forecast, None
            
        except Exception as e:
            logger.error(f"Error making ARIMA prediction: {str(e)}")
            return None, None
    
    def predict_lstm(
        self,
        input_data: np.ndarray,
        steps: int = FORECAST_DAYS
    ) -> np.ndarray:
        """
        Make predictions using LSTM model.
        
        Args:
            input_data: Input sequence data
            steps: Number of steps to forecast
            
        Returns:
            Array of predictions
        """
        if self.lstm_model is None:
            logger.error("LSTM model not loaded")
            return None
        
        try:
            predictions = []
            
            # Use the last sequence_length points as input
            current_sequence = input_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            
            for _ in range(steps):
                # Predict next value
                next_pred = self.lstm_model.predict(current_sequence, verbose=0)
                predictions.append(next_pred[0, 0])
                
                # Update sequence with new prediction
                current_sequence = np.append(current_sequence[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error making LSTM prediction: {str(e)}")
            return None
    
    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        feature_index: int = 0
    ) -> np.ndarray:
        """
        Inverse transform predictions to original scale.
        
        Args:
            predictions: Normalized predictions
            feature_index: Index of the feature (0 for Close price)
            
        Returns:
            Predictions in original scale
        """
        if self.scaler is None:
            logger.warning("Scaler not loaded. Returning predictions as-is.")
            return predictions
        
        try:
            # Create array with correct shape for inverse transform
            pred_array = np.zeros((len(predictions), self.scaler.n_features_in_))
            pred_array[:, feature_index] = predictions
            
            # Inverse transform
            inverse_pred = self.scaler.inverse_transform(pred_array)
            
            return inverse_pred[:, feature_index]
            
        except Exception as e:
            logger.error(f"Error inverse transforming: {str(e)}")
            return predictions
    
    def predict(
        self,
        data: np.ndarray,
        model_type: str = 'both',
        steps: int = FORECAST_DAYS,
        inverse_transform: bool = True
    ) -> dict:
        """
        Make predictions using specified model(s).
        
        Args:
            data: Input data (normalized)
            model_type: 'arima', 'lstm', or 'both'
            steps: Number of steps to forecast
            inverse_transform: Whether to inverse transform predictions
            
        Returns:
            Dictionary with predictions from each model
        """
        results = {}
        
        # ARIMA predictions
        if model_type in ['arima', 'both']:
            if self.arima_model is not None:
                logger.info(f"Making {steps}-step ARIMA forecast...")
                arima_pred, arima_conf = self.predict_arima(steps=steps)
                
                if arima_pred is not None:
                    if inverse_transform and self.scaler is not None:
                        arima_pred = self.inverse_transform_predictions(arima_pred)
                    
                    results['arima'] = {
                        'predictions': arima_pred,
                        'confidence_intervals': arima_conf
                    }
                    logger.info(f"ARIMA predictions: {arima_pred}")
            else:
                logger.warning("ARIMA model not available")
        
        # LSTM predictions
        if model_type in ['lstm', 'both']:
            if self.lstm_model is not None:
                logger.info(f"Making {steps}-step LSTM forecast...")
                lstm_pred = self.predict_lstm(data, steps=steps)
                
                if lstm_pred is not None:
                    if inverse_transform and self.scaler is not None:
                        lstm_pred = self.inverse_transform_predictions(lstm_pred)
                    
                    results['lstm'] = {
                        'predictions': lstm_pred,
                        'confidence_intervals': None
                    }
                    logger.info(f"LSTM predictions: {lstm_pred}")
            else:
                logger.warning("LSTM model not available")
        
        return results
    
    def predict_with_dates(
        self,
        data: pd.DataFrame,
        last_date: pd.Timestamp,
        steps: int = FORECAST_DAYS,
        model_type: str = 'both'
    ) -> pd.DataFrame:
        """
        Make predictions and return DataFrame with dates.
        
        Args:
            data: Historical data
            last_date: Last date in historical data
            steps: Number of steps to forecast
            model_type: Model to use
            
        Returns:
            DataFrame with predictions and dates
        """
        # Get predictions
        if CLOSE_COLUMN in data.columns:
            input_data = data[CLOSE_COLUMN].values
        else:
            input_data = data.iloc[:, 0].values
        
        predictions = self.predict(input_data, model_type=model_type, steps=steps)
        
        # Generate future dates (business days)
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)
        
        # Create DataFrame
        results_df = pd.DataFrame({'Date': future_dates})
        
        if 'arima' in predictions:
            results_df['ARIMA_Prediction'] = predictions['arima']['predictions']
        
        if 'lstm' in predictions:
            results_df['LSTM_Prediction'] = predictions['lstm']['predictions']
        
        return results_df


def make_predictions(
    data: np.ndarray,
    steps: int = FORECAST_DAYS,
    model_type: str = 'both'
) -> dict:
    """
    Convenience function to make predictions.
    
    Args:
        data: Input data
        steps: Number of steps to forecast
        model_type: Model to use ('arima', 'lstm', or 'both')
        
    Returns:
        Dictionary with predictions
    """
    predictor = StockPredictor()
    
    # Load models and scaler
    if model_type in ['arima', 'both']:
        predictor.load_arima_model()
    
    if model_type in ['lstm', 'both']:
        predictor.load_lstm_model()
    
    predictor.load_scaler()
    
    # Make predictions
    return predictor.predict(data, model_type=model_type, steps=steps)


if __name__ == "__main__":
    # Test the predictor
    print("Testing Stock Predictor...")
    
    # Create sample data
    sample_data = np.random.uniform(0, 1, 200)  # Normalized data
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Try to make predictions
    print("\nAttempting to make predictions...")
    predictions = make_predictions(sample_data, steps=5, model_type='both')
    
    print(f"\nPredictions: {predictions}")
