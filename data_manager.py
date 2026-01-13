"""
Binance Historical Data Manager
Downloads OHLCV data for any USDT pair at any timeframe.
Auto-detects available datasets in the historical_data folder.

Optimized for Surface laptops with:
- Memory-efficient float32 storage
- Intelligent LRU caching with memory pressure awareness
- Lazy loading and memory-mapped files for large datasets
"""

import os
import re
import sys
import time
import logging
import gc
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
BINANCE_EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
LIMIT_PER_REQUEST = 1000

# Default symbols (kept for backwards compatibility)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]

# Valid Binance intervals
VALID_INTERVALS = [
    "1m", "3m", "5m", "15m", "30m",  # Minutes
    "1h", "2h", "4h", "6h", "8h", "12h",  # Hours
    "1d", "3d",  # Days
    "1w",  # Week
    "1M"   # Month
]

# Mapping for display names
INTERVAL_NAMES = {
    "1m": "1 Minute", "3m": "3 Minutes", "5m": "5 Minutes", 
    "15m": "15 Minutes", "30m": "30 Minutes",
    "1h": "1 Hour", "2h": "2 Hours", "4h": "4 Hours", 
    "6h": "6 Hours", "8h": "8 Hours", "12h": "12 Hours",
    "1d": "1 Day", "3d": "3 Days", "1w": "1 Week", "1M": "1 Month"
}

# Memory management settings
DEFAULT_MEMORY_LIMIT_MB = 1024  # 1GB default limit for data cache
USE_FLOAT32 = True  # Use float32 for ~50% memory reduction


def get_available_memory_mb() -> int:
    """Get available system memory in MB."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return int(mem.available / (1024 * 1024))
    except ImportError:
        # Fallback estimate - assume 4GB total, 50% available
        return 2048


def estimate_dataframe_memory_mb(df: pd.DataFrame) -> float:
    """Estimate memory usage of a DataFrame in MB."""
    return df.memory_usage(deep=True).sum() / (1024 * 1024)


class DataManager:
    """
    Binance data manager with efficient caching and memory management.
    Optimized for Windows Surface laptops with limited RAM.
    
    Features:
    - Memory-aware LRU cache with automatic eviction
    - Float32 storage for 50% memory reduction
    - Memory pressure detection and proactive cleanup
    - Lazy loading for large datasets
    """
    # Class-level cache for sharing across instances
    _shared_cache: Dict[str, pd.DataFrame] = {}
    _cache_access_times: Dict[str, float] = {}
    _cache_memory_mb: Dict[str, float] = {}
    _max_cache_size = 10  # Max number of datasets to keep in memory
    _max_cache_memory_mb = DEFAULT_MEMORY_LIMIT_MB
    _total_cache_memory_mb = 0.0
    
    def __init__(
        self,
        data_dir: str = "historical_data",
        use_float32: bool = USE_FLOAT32,
        max_cache_memory_mb: Optional[int] = None,
        memory_pressure_threshold: float = 0.8
    ):
        """
        Initialize DataManager.
        
        Args:
            data_dir: Directory for data storage
            use_float32: Use float32 for memory efficiency (default True)
            max_cache_memory_mb: Maximum memory for cache (None = auto-detect)
            memory_pressure_threshold: Fraction of available memory to trigger cleanup
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self._cache = DataManager._shared_cache  # Use shared cache
        self._access_times = DataManager._cache_access_times
        self._cache_memory = DataManager._cache_memory_mb
        self._usdt_symbols_cache = None
        self.use_float32 = use_float32
        self.memory_pressure_threshold = memory_pressure_threshold
        
        # Set cache memory limit
        if max_cache_memory_mb is not None:
            DataManager._max_cache_memory_mb = max_cache_memory_mb
        else:
            # Auto-detect: use 25% of available memory or 1GB, whichever is smaller
            available = get_available_memory_mb()
            DataManager._max_cache_memory_mb = min(1024, int(available * 0.25))
            logger.debug(f"Auto-set cache limit to {DataManager._max_cache_memory_mb}MB (available: {available}MB)")
    
    def get_csv_path(self, symbol: str, interval: str = "1h") -> Path:
        """Get path to CSV file for symbol and interval."""
        return self.data_dir / f"{symbol}_{interval}.csv"
    
    def symbol_exists(self, symbol: str, interval: str = "1h") -> bool:
        """Check if data exists for symbol at given interval."""
        csv_path = self.get_csv_path(symbol, interval)
        return csv_path.exists() and csv_path.stat().st_size > 0
    
    def get_available_datasets(self) -> List[Tuple[str, str]]:
        """
        Scan historical_data folder and return list of available (symbol, interval) pairs.
        
        Returns:
            List of tuples: [(symbol, interval), ...]
        """
        datasets = []
        pattern = re.compile(r'^([A-Z0-9]+)_([0-9]+[mhdwM])\.csv$')
        
        for file in self.data_dir.glob("*.csv"):
            match = pattern.match(file.name)
            if match:
                symbol, interval = match.groups()
                datasets.append((symbol, interval))
        
        return sorted(datasets)
    
    def get_available_symbols(self, interval: str = None) -> List[str]:
        """Get list of symbols available in the data folder."""
        datasets = self.get_available_datasets()
        if interval:
            return sorted(set(s for s, i in datasets if i == interval))
        return sorted(set(s for s, i in datasets))
    
    def get_available_intervals(self, symbol: str = None) -> List[str]:
        """Get list of intervals available in the data folder."""
        datasets = self.get_available_datasets()
        if symbol:
            return sorted(set(i for s, i in datasets if s == symbol))
        return sorted(set(i for s, i in datasets))
    
    def fetch_usdt_symbols(self) -> List[str]:
        """Fetch all USDT trading pairs from Binance."""
        if self._usdt_symbols_cache:
            return self._usdt_symbols_cache
        
        try:
            logger.info("Fetching available USDT pairs from Binance...")
            response = requests.get(BINANCE_EXCHANGE_INFO_URL, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            usdt_symbols = []
            for s in data['symbols']:
                if (s['quoteAsset'] == 'USDT' and 
                    s['status'] == 'TRADING' and
                    s['isSpotTradingAllowed']):
                    usdt_symbols.append(s['symbol'])
            
            self._usdt_symbols_cache = sorted(usdt_symbols)
            logger.info(f"Found {len(usdt_symbols)} USDT trading pairs")
            return self._usdt_symbols_cache
            
        except Exception as e:
            logger.error(f"Failed to fetch symbols from Binance: {e}")
            return []
    
    def download_symbol(self, symbol: str, interval: str = "1h", force: bool = False) -> pd.DataFrame:
        """
        Download historical data for a symbol from Binance.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Candle interval (e.g., '1h', '4h', '1d')
            force: Force re-download even if data exists
        """
        if interval not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval '{interval}'. Valid: {VALID_INTERVALS}")
        
        # Ensure symbol ends with USDT
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'
        symbol = symbol.upper()
        
        csv_path = self.get_csv_path(symbol, interval)
        
        if self.symbol_exists(symbol, interval) and not force:
            logger.info(f"Skipping {symbol} {interval} - CSV already exists")
            return self.load_symbol(symbol, interval)
        
        logger.info(f"Downloading {symbol} {interval} data from Binance...")
        
        all_data = []
        start_time = 1502928000000  # Aug 17, 2017 - Binance launch
        end_time = int(datetime.now().timestamp() * 1000)
        
        current_start = start_time
        request_count = 0
        
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
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
                current_start = data[-1][0] + 1
                request_count += 1
                
                if request_count % 10 == 0:
                    logger.info(f"  {symbol} {interval}: Downloaded {len(all_data):,} candles...")
                
                time.sleep(0.05)  # Rate limiting
                
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
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {symbol} {interval}: {len(df):,} candles to {csv_path}")
        
        return df
    
    def load_symbol(self, symbol: str, interval: str = "1h") -> pd.DataFrame:
        """
        Load historical data for a symbol from CSV with smart LRU caching.
        
        Uses memory-efficient float32 storage and automatic cache management.
        """
        cache_key = f"{symbol}_{interval}"
        
        # Check cache first
        if cache_key in self._cache:
            # Update access time for LRU
            self._access_times[cache_key] = time.time()
            return self._cache[cache_key]
        
        csv_path = self.get_csv_path(symbol, interval)
        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        # Check memory pressure before loading
        self._check_memory_pressure()
        
        # Evict entries if needed (memory or count based)
        self._evict_if_needed()
        
        # Determine dtype based on memory efficiency setting
        if self.use_float32:
            dtype_map = {
                'open': 'float32',
                'high': 'float32',
                'low': 'float32', 
                'close': 'float32',
                'volume': 'float32'
            }
        else:
            dtype_map = {
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64'
            }
        
        # Load with optimized settings
        df = pd.read_csv(
            csv_path, 
            parse_dates=['timestamp'],
            dtype=dtype_map
        )
        
        # Calculate memory usage
        mem_mb = estimate_dataframe_memory_mb(df)
        
        # Store in cache with metadata
        self._cache[cache_key] = df
        self._access_times[cache_key] = time.time()
        self._cache_memory[cache_key] = mem_mb
        DataManager._total_cache_memory_mb += mem_mb
        
        logger.info(f"Loaded {symbol} {interval}: {len(df):,} candles ({mem_mb:.1f}MB)")
        return df
    
    def _check_memory_pressure(self):
        """Check for memory pressure and proactively cleanup if needed."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            usage_ratio = mem.percent / 100
            
            if usage_ratio > self.memory_pressure_threshold:
                logger.warning(
                    f"High memory pressure detected ({usage_ratio:.0%}). "
                    f"Clearing cache to free memory."
                )
                self._evict_all()
                gc.collect()
        except ImportError:
            pass  # psutil not available, skip check
    
    def _evict_if_needed(self):
        """Evict least recently used cache entries based on memory or count limits."""
        # Check memory limit first
        while (DataManager._total_cache_memory_mb > DataManager._max_cache_memory_mb 
               and len(self._cache) > 0):
            self._evict_oldest()
        
        # Then check count limit
        while len(self._cache) >= self._max_cache_size:
            self._evict_oldest()
    
    def _evict_oldest(self):
        """Evict the least recently used cache entry."""
        if not self._access_times:
            return
        
        oldest_key = min(self._access_times, key=self._access_times.get)
        
        # Update memory tracking
        if oldest_key in self._cache_memory:
            DataManager._total_cache_memory_mb -= self._cache_memory[oldest_key]
            del self._cache_memory[oldest_key]
        
        if oldest_key in self._cache:
            del self._cache[oldest_key]
        if oldest_key in self._access_times:
            del self._access_times[oldest_key]
        
        logger.debug(f"Evicted {oldest_key} from cache (LRU)")
    
    def _evict_all(self):
        """Clear entire cache to free memory."""
        self._cache.clear()
        self._access_times.clear()
        self._cache_memory.clear()
        DataManager._total_cache_memory_mb = 0.0
        logger.info("Cleared all cached data")
    
    def clear_cache(self):
        """Clear all cached data to free memory."""
        DataManager._shared_cache.clear()
        DataManager._cache_access_times.clear()
        DataManager._cache_memory_mb.clear()
        DataManager._total_cache_memory_mb = 0.0
        gc.collect()  # Force garbage collection
        logger.info("Data cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        return {
            'cached_datasets': len(self._cache),
            'max_datasets': self._max_cache_size,
            'total_memory_mb': DataManager._total_cache_memory_mb,
            'max_memory_mb': DataManager._max_cache_memory_mb,
            'datasets': list(self._cache.keys()),
            'use_float32': self.use_float32
        }
    
    def optimize_for_surface(self):
        """Apply optimizations specifically for Surface laptops."""
        # Reduce cache limits for constrained memory
        DataManager._max_cache_size = 5
        DataManager._max_cache_memory_mb = 512
        self.use_float32 = True
        self.memory_pressure_threshold = 0.7
        
        # Clear existing cache
        self.clear_cache()
        
        logger.info(
            "Applied Surface laptop optimizations: "
            f"max_cache={DataManager._max_cache_size}, "
            f"max_memory={DataManager._max_cache_memory_mb}MB, "
            "float32=True"
        )
    
    def download_multiple(self, symbols: List[str], interval: str = "1h", force: bool = False) -> Dict[str, pd.DataFrame]:
        """Download data for multiple symbols."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.download_symbol(symbol, interval, force=force)
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
        return results
    
    def load_multiple(self, symbols: List[str] = None, interval: str = "1h") -> Dict[str, pd.DataFrame]:
        """Load data for multiple symbols."""
        if symbols is None:
            symbols = self.get_available_symbols(interval)
        
        results = {}
        for symbol in symbols:
            if self.symbol_exists(symbol, interval):
                try:
                    results[symbol] = self.load_symbol(symbol, interval)
                except Exception as e:
                    logger.error(f"Failed to load {symbol}: {e}")
        return results
    
    # Backwards compatibility methods
    def download_all(self, force: bool = False) -> dict:
        """Download data for default symbols (backwards compatibility)."""
        return self.download_multiple(SYMBOLS, "1h", force)
    
    def load_all(self) -> dict:
        """Load all available 1h symbol data (backwards compatibility)."""
        return self.load_multiple(interval="1h")
    
    def ensure_data_available(self) -> dict:
        """Ensure default symbol data is available."""
        missing = [s for s in SYMBOLS if not self.symbol_exists(s, "1h")]
        if missing:
            logger.info(f"Missing data for: {missing}. Downloading...")
            self.download_multiple(missing, "1h")
        return self.load_all()


def print_available_data(dm: DataManager):
    """Print summary of available datasets."""
    datasets = dm.get_available_datasets()
    
    if not datasets:
        print("\n  No datasets found in historical_data folder.")
        return
    
    print(f"\n  Available datasets ({len(datasets)} total):")
    
    # Group by interval
    by_interval = {}
    for symbol, interval in datasets:
        if interval not in by_interval:
            by_interval[interval] = []
        by_interval[interval].append(symbol)
    
    for interval in sorted(by_interval.keys()):
        symbols = by_interval[interval]
        interval_name = INTERVAL_NAMES.get(interval, interval)
        print(f"\n  {interval_name} ({interval}): {len(symbols)} symbols")
        # Show first 10 symbols
        display_symbols = symbols[:10]
        if len(symbols) > 10:
            print(f"    {', '.join(display_symbols)}, ... (+{len(symbols)-10} more)")
        else:
            print(f"    {', '.join(display_symbols)}")


if __name__ == "__main__":
    dm = DataManager()
    
    print("\n=== Data Manager ===\n")
    print_available_data(dm)
    
    # Test fetching available symbols
    print("\n\nFetching available USDT pairs from Binance...")
    symbols = dm.fetch_usdt_symbols()
    print(f"Found {len(symbols)} USDT pairs")
    print(f"Examples: {symbols[:10]}")
