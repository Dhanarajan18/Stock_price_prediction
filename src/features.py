"""
Feature engineering module for creating technical indicators.
Implements Moving Averages, RSI, MACD, and Bollinger Bands.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

from config import (
    CLOSE_COLUMN, OPEN_COLUMN, HIGH_COLUMN, LOW_COLUMN, VOLUME_COLUMN,
    MOVING_AVERAGES, RSI_PERIOD, MACD_FAST, MACD_SLOW, 
    MACD_SIGNAL, BOLLINGER_PERIOD, BOLLINGER_STD
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates technical indicators for stock price prediction.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        pass
    
    def add_moving_averages(
        self, 
        df: pd.DataFrame, 
        column: str = CLOSE_COLUMN,
        windows: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Add moving average features.
        
        Args:
            df: Input DataFrame
            column: Column to calculate moving averages for
            windows: List of window sizes (default: from config)
            
        Returns:
            DataFrame with moving average features
        """
        df_ma = df.copy()
        
        if column not in df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            return df_ma
        
        if windows is None:
            windows = MOVING_AVERAGES
        
        for window in windows:
            col_name = f'MA_{window}'
            df_ma[col_name] = df_ma[column].rolling(window=window).mean()
            logger.info(f"Added {col_name}")
        
        return df_ma
    
    def add_exponential_moving_average(
        self,
        df: pd.DataFrame,
        column: str = CLOSE_COLUMN,
        windows: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Add exponential moving average (EMA) features.
        
        Args:
            df: Input DataFrame
            column: Column to calculate EMA for
            windows: List of window sizes
            
        Returns:
            DataFrame with EMA features
        """
        df_ema = df.copy()
        
        if column not in df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            return df_ema
        
        if windows is None:
            windows = MOVING_AVERAGES
        
        for window in windows:
            col_name = f'EMA_{window}'
            df_ema[col_name] = df_ema[column].ewm(span=window, adjust=False).mean()
            logger.info(f"Added {col_name}")
        
        return df_ema
    
    def add_rsi(
        self,
        df: pd.DataFrame,
        column: str = CLOSE_COLUMN,
        period: int = RSI_PERIOD
    ) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) indicator.
        
        Args:
            df: Input DataFrame
            column: Column to calculate RSI for
            period: RSI period (default: 14)
            
        Returns:
            DataFrame with RSI feature
        """
        df_rsi = df.copy()
        
        if column not in df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            return df_rsi
        
        # Calculate price changes
        delta = df_rsi[column].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        df_rsi['RSI'] = rsi
        logger.info(f"Added RSI with period {period}")
        
        return df_rsi
    
    def add_macd(
        self,
        df: pd.DataFrame,
        column: str = CLOSE_COLUMN,
        fast_period: int = MACD_FAST,
        slow_period: int = MACD_SLOW,
        signal_period: int = MACD_SIGNAL
    ) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence) indicator.
        
        Args:
            df: Input DataFrame
            column: Column to calculate MACD for
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
            
        Returns:
            DataFrame with MACD features
        """
        df_macd = df.copy()
        
        if column not in df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            return df_macd
        
        # Calculate EMAs
        ema_fast = df_macd[column].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df_macd[column].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate MACD histogram
        macd_histogram = macd_line - signal_line
        
        df_macd['MACD'] = macd_line
        df_macd['MACD_Signal'] = signal_line
        df_macd['MACD_Histogram'] = macd_histogram
        
        logger.info(f"Added MACD with periods ({fast_period}, {slow_period}, {signal_period})")
        
        return df_macd
    
    def add_bollinger_bands(
        self,
        df: pd.DataFrame,
        column: str = CLOSE_COLUMN,
        period: int = BOLLINGER_PERIOD,
        std_dev: int = BOLLINGER_STD
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands indicator.
        
        Args:
            df: Input DataFrame
            column: Column to calculate Bollinger Bands for
            period: Moving average period (default: 20)
            std_dev: Number of standard deviations (default: 2)
            
        Returns:
            DataFrame with Bollinger Bands features
        """
        df_bb = df.copy()
        
        if column not in df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            return df_bb
        
        # Calculate middle band (SMA)
        middle_band = df_bb[column].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = df_bb[column].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        # Calculate bandwidth and %B
        bandwidth = (upper_band - lower_band) / middle_band
        percent_b = (df_bb[column] - lower_band) / (upper_band - lower_band)
        
        df_bb['BB_Middle'] = middle_band
        df_bb['BB_Upper'] = upper_band
        df_bb['BB_Lower'] = lower_band
        df_bb['BB_Width'] = bandwidth
        df_bb['BB_PercentB'] = percent_b
        
        logger.info(f"Added Bollinger Bands with period {period} and {std_dev} std dev")
        
        return df_bb
    
    def add_volume_features(
        self,
        df: pd.DataFrame,
        volume_column: str = VOLUME_COLUMN
    ) -> pd.DataFrame:
        """
        Add volume-based features.
        
        Args:
            df: Input DataFrame
            volume_column: Volume column name
            
        Returns:
            DataFrame with volume features
        """
        df_vol = df.copy()
        
        if volume_column not in df.columns:
            logger.warning(f"Column {volume_column} not found in DataFrame")
            return df_vol
        
        # Volume moving averages
        df_vol['Volume_MA_10'] = df_vol[volume_column].rolling(window=10).mean()
        df_vol['Volume_MA_20'] = df_vol[volume_column].rolling(window=20).mean()
        
        # Volume ratio
        df_vol['Volume_Ratio'] = df_vol[volume_column] / df_vol['Volume_MA_20']
        
        # On-Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(df_vol)):
            if df_vol[CLOSE_COLUMN].iloc[i] > df_vol[CLOSE_COLUMN].iloc[i-1]:
                obv.append(obv[-1] + df_vol[volume_column].iloc[i])
            elif df_vol[CLOSE_COLUMN].iloc[i] < df_vol[CLOSE_COLUMN].iloc[i-1]:
                obv.append(obv[-1] - df_vol[volume_column].iloc[i])
            else:
                obv.append(obv[-1])
        
        df_vol['OBV'] = obv
        
        logger.info("Added volume features")
        
        return df_vol
    
    def add_price_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add price-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with price features
        """
        df_price = df.copy()
        
        required_cols = [OPEN_COLUMN, HIGH_COLUMN, LOW_COLUMN, CLOSE_COLUMN]
        if not all(col in df.columns for col in required_cols):
            logger.warning("Not all price columns available for price features")
            return df_price
        
        # Daily returns
        df_price['Daily_Return'] = df_price[CLOSE_COLUMN].pct_change()
        
        # Price range
        df_price['Price_Range'] = df_price[HIGH_COLUMN] - df_price[LOW_COLUMN]
        
        # Percentage price range
        df_price['Price_Range_Pct'] = df_price['Price_Range'] / df_price[CLOSE_COLUMN]
        
        # Gap (difference between open and previous close)
        df_price['Gap'] = df_price[OPEN_COLUMN] - df_price[CLOSE_COLUMN].shift(1)
        
        # Body size (close - open)
        df_price['Body_Size'] = df_price[CLOSE_COLUMN] - df_price[OPEN_COLUMN]
        
        # Upper shadow
        df_price['Upper_Shadow'] = df_price[HIGH_COLUMN] - df_price[[OPEN_COLUMN, CLOSE_COLUMN]].max(axis=1)
        
        # Lower shadow
        df_price['Lower_Shadow'] = df_price[[OPEN_COLUMN, CLOSE_COLUMN]].min(axis=1) - df_price[LOW_COLUMN]
        
        logger.info("Added price features")
        
        return df_price
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators and features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all features
        """
        logger.info("Adding all technical indicators...")
        
        df_features = df.copy()
        
        # Add all features
        df_features = self.add_moving_averages(df_features)
        df_features = self.add_exponential_moving_average(df_features)
        df_features = self.add_rsi(df_features)
        df_features = self.add_macd(df_features)
        df_features = self.add_bollinger_bands(df_features)
        
        if VOLUME_COLUMN in df_features.columns:
            df_features = self.add_volume_features(df_features)
        
        if all(col in df_features.columns for col in [OPEN_COLUMN, HIGH_COLUMN, LOW_COLUMN, CLOSE_COLUMN]):
            df_features = self.add_price_features(df_features)
        
        # Drop rows with NaN values created by indicators
        initial_rows = len(df_features)
        df_features = df_features.dropna()
        logger.info(f"Dropped {initial_rows - len(df_features)} rows with NaN values")
        
        logger.info(f"Total features: {len(df_features.columns)}")
        
        return df_features


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to create all features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with all features
    """
    engineer = FeatureEngineer()
    return engineer.add_all_features(df)


if __name__ == "__main__":
    # Test the feature engineer
    print("Testing Feature Engineer...")
    
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
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Original columns: {sample_data.columns.tolist()}")
    
    # Create features
    df_features = create_features(sample_data)
    
    print(f"\nData with features shape: {df_features.shape}")
    print(f"Feature columns: {df_features.columns.tolist()}")
    print(f"\nFirst few rows:\n{df_features.head()}")
