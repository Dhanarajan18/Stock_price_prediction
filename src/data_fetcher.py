"""
Data fetcher module for downloading historical and live stock data from NSE.
Uses nsepython library for Indian stock market data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional
import time

try:
    from nsepython import equity_history
except ImportError:
    equity_history = None
    logging.warning("nsepython not installed. Install with: pip install nsepython")

# Alternative data source
try:
    import yfinance as yf
except ImportError:
    yf = None
    logging.warning("yfinance not installed. Install with: pip install yfinance")

from config import DEFAULT_DAYS_HISTORY, DATA_DIR, DATE_COLUMN
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockDataFetcher:
    """
    Fetches historical stock data for Indian companies from NSE.
    """
    
    def __init__(self):
        """Initialize the stock data fetcher."""
        self.data_dir = DATA_DIR
        
    def fetch_historical_data(
        self, 
        symbol: str, 
        days: int = DEFAULT_DAYS_HISTORY,
        save_to_csv: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data for a given symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
            days: Number of days of historical data to fetch
            save_to_csv: Whether to save data to CSV file
            
        Returns:
            DataFrame with historical stock data or None if failed
        """
        logger.info(f"Fetching {days} days of historical data for {symbol}")
        
        # Warn if requesting too little data
        if days < 100:
            logger.warning(f"Requesting only {days} days of data. Recommended: 365+ days for reliable predictions.")
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch data using nsepython
            if equity_history is None:
                logger.error("nsepython library not available. Using fallback method.")
                return self._fetch_fallback_data(symbol, start_date, end_date)
            
            # NSE symbol format (add .NS suffix if needed)
            nse_symbol = symbol.upper()
            
            # Try multiple data sources
            df = None
            
            # Method 1: Try NSE directly via nsepython
            logger.info(f"Requesting data from NSE for {nse_symbol}")
            df = self._fetch_nse_data(nse_symbol, start_date, end_date)
            
            # Method 2: If NSE fails, try Yahoo Finance
            if df is None or df.empty:
                logger.info(f"NSE fetch failed, trying Yahoo Finance...")
                df = self._fetch_yfinance_data(nse_symbol, start_date, end_date)
            
            # Method 3: If all methods fail, use fallback
            if df is None or df.empty:
                logger.warning(f"All real data sources failed for {symbol}. Using fallback sample data.")
                logger.warning("NOTE: This is synthetic data, not real market data!")
                df = self._fetch_fallback_data(symbol, start_date, end_date)
            else:
                # Process and clean the real data
                df = self._process_data(df)
            
            # Save to CSV if requested (works for both NSE and fallback data)
            if save_to_csv and df is not None and not df.empty:
                csv_path = os.path.join(self.data_dir, f"{symbol}_historical.csv")
                df.to_csv(csv_path, index=False)
                logger.info(f"Data saved to {csv_path}")
            
            if df is not None:
                logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def _fetch_nse_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from NSE using nsepython.
        
        Args:
            symbol: NSE stock symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with stock data or None
        """
        try:
            # Format dates for NSE API
            start_str = start_date.strftime('%d-%m-%Y')
            end_str = end_date.strftime('%d-%m-%Y')
            
            # Fetch equity history
            data = equity_history(symbol, 'EQ', start_str, end_str)
            
            if data is None:
                return None
                
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            logger.warning(f"NSE fetch failed: {str(e)}")
            return None
    
    def _fetch_yfinance_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from Yahoo Finance using yfinance.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with stock data or None
        """
        if yf is None:
            logger.warning("yfinance not available")
            return None
        
        try:
            # For NSE stocks, add .NS suffix for Yahoo Finance
            yf_symbol = f"{symbol}.NS"
            
            logger.info(f"Trying Yahoo Finance for {yf_symbol}...")
            
            # Download data
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                # Try BSE with .BO suffix
                yf_symbol = f"{symbol}.BO"
                logger.info(f"Trying BSE via Yahoo Finance for {yf_symbol}...")
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                return None
            
            # Reset index to get Date as column
            df = df.reset_index()
            
            # Standardize column names
            df = df.rename(columns={
                'Date': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            # Select required columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            logger.info(f"Successfully fetched {len(df)} records from Yahoo Finance")
            return df
            
        except Exception as e:
            logger.warning(f"Yahoo Finance fetch failed: {str(e)}")
            return None
    
    def _fetch_fallback_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fallback method to generate sample data when API is unavailable.
        This is for development/testing purposes only.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with sample stock data
        """
        logger.warning("Using fallback sample data generation. This is NOT real market data!")
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Generate sample price data with realistic movement
        np.random.seed(hash(symbol) % 2**32)  # Seed based on symbol for consistency
        base_price = np.random.uniform(100, 2000)
        
        prices = [base_price]
        for _ in range(len(dates) - 1):
            change = np.random.normal(0, base_price * 0.02)  # 2% daily volatility
            new_price = max(prices[-1] + change, base_price * 0.5)  # Don't go below 50% of base
            prices.append(new_price)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
            'High': [p * np.random.uniform(1.00, 1.05) for p in prices],
            'Low': [p * np.random.uniform(0.95, 1.00) for p in prices],
            'Close': prices,
            'Volume': [np.random.randint(100000, 10000000) for _ in prices]
        })
        
        return df
    
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean the raw data.
        
        Args:
            df: Raw DataFrame from API
            
        Returns:
            Processed DataFrame
        """
        # Standardize column names
        column_mapping = {
            'CH_TIMESTAMP': 'Date',
            'CH_OPENING_PRICE': 'Open',
            'CH_TRADE_HIGH_PRICE': 'High',
            'CH_TRADE_LOW_PRICE': 'Low',
            'CH_CLOSING_PRICE': 'Close',
            'CH_TOT_TRADED_QTY': 'Volume',
            'TIMESTAMP': 'Date',
            'OPEN': 'Open',
            'HIGH': 'High',
            'LOW': 'Low',
            'CLOSE': 'Close',
            'VOLUME': 'Volume'
        }
        
        # Rename columns if they exist
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Missing required columns. Available: {df.columns.tolist()}")
            # Try to use only close price if available
            if 'Close' in df.columns or 'CLOSE' in df.columns:
                df = df.rename(columns={'CLOSE': 'Close'})
        
        # Convert Date to datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['Date'], keep='last')
        
        # Select only required columns
        columns_to_keep = [col for col in required_columns if col in df.columns]
        df = df[columns_to_keep]
        
        return df
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest closing price or None
        """
        try:
            df = self.fetch_historical_data(symbol, days=5, save_to_csv=False)
            if df is not None and not df.empty:
                return df.iloc[-1]['Close']
            return None
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {str(e)}")
            return None


def fetch_stock_data(symbol: str, days: int = DEFAULT_DAYS_HISTORY) -> Optional[pd.DataFrame]:
    """
    Convenience function to fetch stock data.
    
    Args:
        symbol: Stock symbol
        days: Number of days of historical data
        
    Returns:
        DataFrame with stock data or None
    """
    fetcher = StockDataFetcher()
    return fetcher.fetch_historical_data(symbol, days)


if __name__ == "__main__":
    # Test the data fetcher
    print("Testing Stock Data Fetcher...")
    
    # Test symbols
    test_symbols = ['RELIANCE', 'TCS', 'INFY']
    
    for symbol in test_symbols:
        print(f"\nFetching data for {symbol}...")
        df = fetch_stock_data(symbol, days=365)
        
        if df is not None:
            print(f"Successfully fetched {len(df)} records")
            print(df.head())
            print(f"\nData range: {df['Date'].min()} to {df['Date'].max()}")
        else:
            print(f"Failed to fetch data for {symbol}")
