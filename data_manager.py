"""
Binance Historical Data Manager
Downloads 1H OHLCV data for specified symbols with skip-existing logic.
"""

import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
INTERVAL = "1h"
LIMIT_PER_REQUEST = 1000

class DataManager:
    def __init__(self, data_dir: str = "historical_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self._cache = {}
    
    def get_csv_path(self, symbol: str) -> Path:
        return self.data_dir / f"{symbol}_1h.csv"
    
    def symbol_exists(self, symbol: str) -> bool:
        csv_path = self.get_csv_path(symbol)
        return csv_path.exists() and csv_path.stat().st_size > 0
    
    def download_symbol(self, symbol: str, force: bool = False) -> pd.DataFrame:
        """Download historical data for a single symbol from Binance."""
        csv_path = self.get_csv_path(symbol)
        
        if self.symbol_exists(symbol) and not force:
            logger.info(f"Skipping {symbol} - CSV already exists at {csv_path}")
            return self.load_symbol(symbol)
        
        logger.info(f"Downloading {symbol} 1H data from Binance...")
        
        all_data = []
        start_time = 1502928000000  # Aug 17, 2017 - Binance launch approximate
        end_time = int(datetime.now().timestamp() * 1000)
        
        current_start = start_time
        request_count = 0
        
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "interval": INTERVAL,
                "startTime": current_start,
                "limit": LIMIT_PER_REQUEST
            }
            
            try:
                response = requests.get(BINANCE_API_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                current_start = data[-1][0] + 1  # Next millisecond after last candle
                request_count += 1
                
                if request_count % 10 == 0:
                    logger.info(f"  {symbol}: Downloaded {len(all_data):,} candles...")
                
                # Rate limiting - Binance allows 1200 requests/min
                time.sleep(0.05)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error downloading {symbol}: {e}")
                if all_data:
                    logger.info(f"Saving partial data ({len(all_data)} candles)...")
                    break
                raise
        
        if not all_data:
            raise ValueError(f"No data downloaded for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Keep only OHLCV columns and convert types
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {symbol}: {len(df):,} candles to {csv_path}")
        
        return df
    
    def load_symbol(self, symbol: str) -> pd.DataFrame:
        """Load historical data for a symbol from CSV."""
        if symbol in self._cache:
            return self._cache[symbol]
        
        csv_path = self.get_csv_path(symbol)
        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}. Run download first.")
        
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        self._cache[symbol] = df
        logger.info(f"Loaded {symbol}: {len(df):,} candles")
        return df
    
    def download_all(self, force: bool = False) -> dict:
        """Download data for all configured symbols."""
        results = {}
        for symbol in SYMBOLS:
            try:
                results[symbol] = self.download_symbol(symbol, force=force)
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
                raise
        return results
    
    def load_all(self) -> dict:
        """Load all available symbol data."""
        results = {}
        for symbol in SYMBOLS:
            if self.symbol_exists(symbol):
                results[symbol] = self.load_symbol(symbol)
        return results
    
    def ensure_data_available(self) -> dict:
        """Ensure all symbol data is available, downloading if necessary."""
        missing = [s for s in SYMBOLS if not self.symbol_exists(s)]
        if missing:
            logger.info(f"Missing data for: {missing}. Downloading...")
            self.download_all()
        return self.load_all()


if __name__ == "__main__":
    # Test download
    dm = DataManager()
    dm.download_all()

