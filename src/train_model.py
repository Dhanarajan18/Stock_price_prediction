"""
Model training module for ARIMA and LSTM models.
Trains both models on historical stock data.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import os
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ARIMA imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from pmdarima import auto_arima
except ImportError:
    ARIMA = None
    auto_arima = None
    logging.warning("statsmodels or pmdarima not installed")

# LSTM imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    tf = None
    keras = None
    logging.warning("TensorFlow not installed")

from sklearn.model_selection import train_test_split

from config import (
    ARIMA_ORDER, LSTM_SEQUENCE_LENGTH, LSTM_UNITS, LSTM_DROPOUT,
    LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_VALIDATION_SPLIT, LSTM_LEARNING_RATE,
    MODELS_DIR, ARIMA_MODEL_FILE, LSTM_MODEL_FILE, CLOSE_COLUMN,
    TRAIN_TEST_SPLIT
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ARIMATrainer:
    """
    Trains ARIMA model for time series forecasting.
    """
    
    def __init__(self, order: tuple = ARIMA_ORDER):
        """
        Initialize ARIMA trainer.
        
        Args:
            order: ARIMA order (p, d, q)
        """
        self.order = order
        self.model = None
        self.fitted_model = None
    
    def auto_tune(self, data: np.ndarray, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> tuple:
        """
        Automatically find the best ARIMA parameters using auto_arima.
        
        Args:
            data: Time series data
            max_p: Maximum p value
            max_d: Maximum d value
            max_q: Maximum q value
            
        Returns:
            Best ARIMA order (p, d, q)
        """
        if auto_arima is None:
            logger.warning("auto_arima not available. Using default order.")
            return self.order
        
        logger.info("Auto-tuning ARIMA parameters...")
        
        try:
            model = auto_arima(
                data,
                max_p=max_p,
                max_d=max_d,
                max_q=max_q,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            
            best_order = model.order
            logger.info(f"Best ARIMA order found: {best_order}")
            return best_order
            
        except Exception as e:
            logger.error(f"Auto-tuning failed: {str(e)}. Using default order.")
            return self.order
    
    def train(
        self, 
        train_data: np.ndarray,
        auto_tune: bool = False
    ) -> object:
        """
        Train ARIMA model.
        
        Args:
            train_data: Training data
            auto_tune: Whether to auto-tune parameters
            
        Returns:
            Fitted ARIMA model
        """
        if ARIMA is None:
            logger.error("ARIMA not available. Install statsmodels.")
            return None
        
        logger.info(f"Training ARIMA model with {len(train_data)} data points")
        
        try:
            # Auto-tune if requested
            if auto_tune:
                self.order = self.auto_tune(train_data)
            
            # Fit ARIMA model
            logger.info(f"Fitting ARIMA{self.order}")
            self.model = ARIMA(train_data, order=self.order)
            self.fitted_model = self.model.fit()
            
            logger.info("ARIMA model trained successfully")
            logger.info(f"AIC: {self.fitted_model.aic:.2f}")
            
            return self.fitted_model
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {str(e)}")
            return None
    
    def save_model(self, filepath: Optional[str] = None) -> None:
        """
        Save the trained ARIMA model.
        
        Args:
            filepath: Path to save the model
        """
        if self.fitted_model is None:
            logger.warning("No fitted model to save")
            return
        
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, ARIMA_MODEL_FILE)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.fitted_model, f)
        
        logger.info(f"ARIMA model saved to {filepath}")
    
    def load_model(self, filepath: Optional[str] = None) -> object:
        """
        Load a trained ARIMA model.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded ARIMA model
        """
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, ARIMA_MODEL_FILE)
        
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            return None
        
        with open(filepath, 'rb') as f:
            self.fitted_model = pickle.load(f)
        
        logger.info(f"ARIMA model loaded from {filepath}")
        return self.fitted_model


class LSTMTrainer:
    """
    Trains LSTM neural network for stock price prediction.
    """
    
    def __init__(
        self,
        sequence_length: int = LSTM_SEQUENCE_LENGTH,
        units: list = LSTM_UNITS,
        dropout: float = LSTM_DROPOUT
    ):
        """
        Initialize LSTM trainer.
        
        Args:
            sequence_length: Number of time steps to look back
            units: List of units for each LSTM layer
            dropout: Dropout rate
        """
        self.sequence_length = sequence_length
        self.units = units if isinstance(units, list) else [units]
        self.dropout = dropout
        self.model = None
    
    def create_sequences(
        self, 
        data: np.ndarray,
        target_data: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Input data
            target_data: Target data (if different from input)
            
        Returns:
            Tuple of (X, y) sequences
        """
        if target_data is None:
            target_data = data
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(target_data[i])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: tuple) -> Sequential:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        if keras is None:
            logger.error("TensorFlow/Keras not available")
            return None
        
        model = Sequential()
        
        # Add LSTM layers
        for i, units in enumerate(self.units):
            return_sequences = i < len(self.units) - 1
            
            if i == 0:
                model.add(LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    input_shape=input_shape
                ))
            else:
                model.add(LSTM(
                    units=units,
                    return_sequences=return_sequences
                ))
            
            model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=LSTM_LEARNING_RATE),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        logger.info("LSTM model architecture:")
        model.summary(print_fn=logger.info)
        
        return model
    
    def train(
        self,
        train_data: np.ndarray,
        batch_size: int = LSTM_BATCH_SIZE,
        epochs: int = LSTM_EPOCHS,
        validation_split: float = LSTM_VALIDATION_SPLIT,
        early_stopping: bool = True
    ) -> tuple:
        """
        Train LSTM model.
        
        Args:
            train_data: Training data (normalized)
            batch_size: Batch size
            epochs: Number of epochs
            validation_split: Validation split ratio
            early_stopping: Whether to use early stopping
            
        Returns:
            Tuple of (trained model, training history)
        """
        if keras is None:
            logger.error("TensorFlow/Keras not available")
            return None, None
        
        logger.info(f"Training LSTM model with {len(train_data)} data points")
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_data)
        logger.info(f"Created sequences: X shape {X_train.shape}, y shape {y_train.shape}")
        
        # Reshape for LSTM [samples, time steps, features]
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        if self.model is None:
            return None, None
        
        # Callbacks
        callbacks = []
        
        if early_stopping:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            callbacks.append(early_stop)
        
        # Model checkpoint
        checkpoint_path = os.path.join(MODELS_DIR, 'lstm_best.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        callbacks.append(checkpoint)
        
        # Train model
        logger.info(f"Training for {epochs} epochs...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("LSTM model trained successfully")
        
        return self.model, history
    
    def save_model(self, filepath: Optional[str] = None) -> None:
        """
        Save the trained LSTM model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            logger.warning("No model to save")
            return
        
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, LSTM_MODEL_FILE)
        
        self.model.save(filepath)
        logger.info(f"LSTM model saved to {filepath}")
    
    def load_model(self, filepath: Optional[str] = None) -> Sequential:
        """
        Load a trained LSTM model.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded Keras model
        """
        if keras is None:
            logger.error("TensorFlow/Keras not available")
            return None
        
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, LSTM_MODEL_FILE)
        
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            return None
        
        self.model = keras.models.load_model(filepath)
        logger.info(f"LSTM model loaded from {filepath}")
        
        return self.model


def train_models(
    data: pd.DataFrame,
    target_column: str = CLOSE_COLUMN,
    train_arima: bool = True,
    train_lstm: bool = True,
    auto_tune_arima: bool = False
) -> Tuple[Optional[object], Optional[object]]:
    """
    Train both ARIMA and LSTM models.
    
    Args:
        data: Input DataFrame with stock data (normalized)
        target_column: Target column to predict
        train_arima: Whether to train ARIMA model
        train_lstm: Whether to train LSTM model
        auto_tune_arima: Whether to auto-tune ARIMA parameters
        
    Returns:
        Tuple of (ARIMA model, LSTM model)
    """
    if target_column not in data.columns:
        logger.error(f"Target column {target_column} not found in data")
        return None, None
    
    # Extract target data
    target_data = data[target_column].values
    
    # Check if we have enough data
    if len(target_data) < 60:
        logger.error(f"Insufficient data: only {len(target_data)} samples")
        logger.error("Need at least 60 samples for LSTM and 30 for ARIMA")
        return None, None
    
    # Split data
    split_idx = int(len(target_data) * TRAIN_TEST_SPLIT)
    train_data = target_data[:split_idx]
    
    # Check training data size
    if len(train_data) < 30:
        logger.error(f"Insufficient training data: only {len(train_data)} samples after split")
        logger.error("Need at least 30 training samples")
        return None, None
    
    logger.info(f"Training with {len(train_data)} samples (test: {len(target_data) - len(train_data)})")
    
    arima_model = None
    lstm_model = None
    
    # Train ARIMA
    if train_arima:
        logger.info("\n" + "="*50)
        logger.info("Training ARIMA Model")
        logger.info("="*50)
        
        arima_trainer = ARIMATrainer()
        arima_model = arima_trainer.train(train_data, auto_tune=auto_tune_arima)
        
        if arima_model is not None:
            arima_trainer.save_model()
    
    # Train LSTM
    if train_lstm:
        logger.info("\n" + "="*50)
        logger.info("Training LSTM Model")
        logger.info("="*50)
        
        lstm_trainer = LSTMTrainer()
        lstm_model, history = lstm_trainer.train(train_data)
        
        if lstm_model is not None:
            lstm_trainer.save_model()
    
    return arima_model, lstm_model


if __name__ == "__main__":
    # Test the trainers
    print("Testing Model Trainers...")
    
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='B')
    prices = np.random.uniform(100, 200, len(dates))
    
    sample_data = pd.DataFrame({
        'Date': dates,
        'Close': prices
    })
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Normalize data (simple normalization for testing)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    sample_data['Close'] = scaler.fit_transform(sample_data[['Close']])
    
    # Train models
    print("\nTraining models...")
    arima_model, lstm_model = train_models(
        sample_data,
        train_arima=True,
        train_lstm=True,
        auto_tune_arima=False
    )
    
    print("\nTraining completed!")
    print(f"ARIMA model: {'Trained' if arima_model is not None else 'Failed'}")
    print(f"LSTM model: {'Trained' if lstm_model is not None else 'Failed'}")
