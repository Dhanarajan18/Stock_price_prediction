"""
Data preprocessing module for cleaning and normalizing stock data.
Handles missing values, outliers, and data normalization.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import Tuple, Optional
import pickle
import os

from config import (
    CLOSE_COLUMN, DATE_COLUMN, OPEN_COLUMN, HIGH_COLUMN, 
    LOW_COLUMN, VOLUME_COLUMN, MODELS_DIR, SCALER_FILE
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses stock data for model training.
    Handles missing values, normalization, and feature scaling.
    """
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler_fitted = False
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the stock data by handling missing values and outliers.
        
        Args:
            df: Input DataFrame with stock data
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning data: {len(df)} rows before cleaning")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Handle missing values in date column
        if DATE_COLUMN in df_clean.columns:
            df_clean = df_clean.dropna(subset=[DATE_COLUMN])
        
        # Forward fill missing values for price columns
        price_columns = [OPEN_COLUMN, HIGH_COLUMN, LOW_COLUMN, CLOSE_COLUMN]
        existing_price_cols = [col for col in price_columns if col in df_clean.columns]
        
        if existing_price_cols:
            df_clean[existing_price_cols] = df_clean[existing_price_cols].fillna(method='ffill')
            df_clean[existing_price_cols] = df_clean[existing_price_cols].fillna(method='bfill')
        
        # Handle volume - fill with median
        if VOLUME_COLUMN in df_clean.columns:
            median_volume = df_clean[VOLUME_COLUMN].median()
            df_clean[VOLUME_COLUMN] = df_clean[VOLUME_COLUMN].fillna(median_volume)
        
        # Remove any remaining rows with NaN values
        df_clean = df_clean.dropna()
        
        # Remove duplicates based on date
        if DATE_COLUMN in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=[DATE_COLUMN], keep='last')
        
        # Sort by date
        if DATE_COLUMN in df_clean.columns:
            df_clean = df_clean.sort_values(DATE_COLUMN).reset_index(drop=True)
        
        logger.info(f"Data cleaned: {len(df_clean)} rows after cleaning")
        
        return df_clean
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Handle outliers in the data.
        
        Args:
            df: Input DataFrame
            method: Method to use ('iqr' or 'zscore')
            
        Returns:
            DataFrame with outliers handled
        """
        df_clean = df.copy()
        
        # Only check price columns for outliers
        price_columns = [CLOSE_COLUMN, OPEN_COLUMN, HIGH_COLUMN, LOW_COLUMN]
        existing_cols = [col for col in price_columns if col in df_clean.columns]
        
        if method == 'iqr':
            for col in existing_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # Using 3*IQR for less aggressive filtering
                upper_bound = Q3 + 3 * IQR
                
                # Cap outliers instead of removing them
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif method == 'zscore':
            for col in existing_cols:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                lower_bound = mean - 4 * std  # Using 4 std for less aggressive filtering
                upper_bound = mean + 4 * std
                
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_clean
    
    def normalize_data(
        self, 
        df: pd.DataFrame, 
        fit: bool = True,
        columns_to_scale: Optional[list] = None
    ) -> Tuple[pd.DataFrame, MinMaxScaler]:
        """
        Normalize the data using MinMaxScaler.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for training, False for testing)
            columns_to_scale: List of columns to scale (default: price and volume)
            
        Returns:
            Tuple of (normalized DataFrame, scaler object)
        """
        df_normalized = df.copy()
        
        # Default columns to scale
        if columns_to_scale is None:
            columns_to_scale = [CLOSE_COLUMN, OPEN_COLUMN, HIGH_COLUMN, LOW_COLUMN, VOLUME_COLUMN]
        
        # Filter to only existing columns
        columns_to_scale = [col for col in columns_to_scale if col in df.columns]
        
        if not columns_to_scale:
            logger.warning("No columns to scale found")
            return df_normalized, self.scaler
        
        # Fit and transform or just transform
        if fit:
            df_normalized[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
            self.scaler_fitted = True
            logger.info(f"Scaler fitted on columns: {columns_to_scale}")
        else:
            if not self.scaler_fitted:
                logger.warning("Scaler not fitted yet. Fitting now...")
                df_normalized[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
                self.scaler_fitted = True
            else:
                df_normalized[columns_to_scale] = self.scaler.transform(df[columns_to_scale])
        
        return df_normalized, self.scaler
    
    def create_lagged_features(
        self, 
        df: pd.DataFrame, 
        target_column: str = CLOSE_COLUMN,
        lags: list = [1, 2, 3, 5, 7]
    ) -> pd.DataFrame:
        """
        Create lagged features for time series prediction.
        
        Args:
            df: Input DataFrame
            target_column: Column to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        df_lagged = df.copy()
        
        if target_column not in df.columns:
            logger.warning(f"Column {target_column} not found in DataFrame")
            return df_lagged
        
        # Create lagged features
        for lag in lags:
            df_lagged[f'{target_column}_lag_{lag}'] = df_lagged[target_column].shift(lag)
        
        # Drop rows with NaN values created by lagging
        df_lagged = df_lagged.dropna()
        
        logger.info(f"Created {len(lags)} lagged features")
        
        return df_lagged
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        column: str = CLOSE_COLUMN,
        windows: list = [7, 14, 30]
    ) -> pd.DataFrame:
        """
        Create rolling statistical features.
        
        Args:
            df: Input DataFrame
            column: Column to calculate rolling statistics for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        df_rolling = df.copy()
        
        if column not in df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            return df_rolling
        
        for window in windows:
            df_rolling[f'{column}_rolling_mean_{window}'] = df_rolling[column].rolling(window=window).mean()
            df_rolling[f'{column}_rolling_std_{window}'] = df_rolling[column].rolling(window=window).std()
        
        # Drop NaN values
        df_rolling = df_rolling.dropna()
        
        logger.info(f"Created rolling features for {len(windows)} windows")
        
        return df_rolling
    
    def save_scaler(self, filepath: Optional[str] = None) -> None:
        """
        Save the fitted scaler to disk.
        
        Args:
            filepath: Path to save the scaler (default: models/scaler.pkl)
        """
        if not self.scaler_fitted:
            logger.warning("Scaler not fitted yet. Nothing to save.")
            return
        
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, SCALER_FILE)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: Optional[str] = None) -> None:
        """
        Load a fitted scaler from disk.
        
        Args:
            filepath: Path to load the scaler from (default: models/scaler.pkl)
        """
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, SCALER_FILE)
        
        if not os.path.exists(filepath):
            logger.error(f"Scaler file not found: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.scaler_fitted = True
        logger.info(f"Scaler loaded from {filepath}")
    
    def inverse_transform(self, data: np.ndarray, column_index: int = 0) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            data: Normalized data
            column_index: Index of the column to inverse transform
            
        Returns:
            Data in original scale
        """
        if not self.scaler_fitted:
            logger.error("Scaler not fitted. Cannot inverse transform.")
            return data
        
        # Create array with zeros for other features
        data_array = np.zeros((len(data), self.scaler.n_features_in_))
        data_array[:, column_index] = data.flatten()
        
        # Inverse transform
        inverse_data = self.scaler.inverse_transform(data_array)
        
        return inverse_data[:, column_index]


def preprocess_data(df: pd.DataFrame, fit_scaler: bool = True) -> Tuple[pd.DataFrame, DataPreprocessor]:
    """
    Convenience function to preprocess stock data.
    
    Args:
        df: Input DataFrame
        fit_scaler: Whether to fit the scaler
        
    Returns:
        Tuple of (preprocessed DataFrame, preprocessor object)
    """
    preprocessor = DataPreprocessor()
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Handle outliers
    df_clean = preprocessor.handle_outliers(df_clean)
    
    # Normalize data
    df_normalized, _ = preprocessor.normalize_data(df_clean, fit=fit_scaler)
    
    return df_normalized, preprocessor


if __name__ == "__main__":
    # Test the preprocessor
    print("Testing Data Preprocessor...")
    
    # Create sample data
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='B')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(100, 200, len(dates)),
        'High': np.random.uniform(150, 250, len(dates)),
        'Low': np.random.uniform(50, 150, len(dates)),
        'Close': np.random.uniform(100, 200, len(dates)),
        'Volume': np.random.randint(100000, 1000000, len(dates))
    })
    
    # Add some missing values
    sample_data.loc[10:15, 'Close'] = np.nan
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Missing values:\n{sample_data.isnull().sum()}")
    
    # Preprocess
    df_processed, preprocessor = preprocess_data(sample_data)
    
    print(f"\nProcessed data shape: {df_processed.shape}")
    print(f"Missing values after processing:\n{df_processed.isnull().sum()}")
    print(f"\nFirst few rows:\n{df_processed.head()}")
