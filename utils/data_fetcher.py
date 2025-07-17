"""
Data Fetcher Module
Handles real-time data ingestion from CoinMarketCap and Yahoo Finance
"""

import yfinance as yf
import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import time
from config import config

class DataFetcher:
    """Class for fetching financial data from various sources"""
    
    def __init__(self):
        self.coinmarketcap_api_key = config.COINMARKETCAP_API_KEY
        self.coinmarketcap_base_url = config.COINMARKETCAP_BASE_URL
        self.cache_duration = config.DATA_CONFIG["cache_duration"]
        
    def fetch_crypto_data(self, symbols, period='1y', interval='1d'):
        """
        Fetch cryptocurrency data from CoinMarketCap API
        
        Args:
            symbols: List of crypto symbols (e.g., ['BTC', 'ETH'])
            period: Time period for data
            interval: Data interval
            
        Returns:
            crypto_data: Dictionary with crypto data for each symbol
        """
        try:
            crypto_data = {}
            
            for symbol in symbols:
                try:
                    # Get symbol ID from CoinMarketCap
                    symbol_id = self._get_crypto_symbol_id(symbol)
                    
                    if symbol_id:
                        # Fetch historical data
                        historical_data = self._fetch_crypto_historical_data(symbol_id, period)
                        
                        if historical_data is not None:
                            crypto_data[symbol] = historical_data
                        else:
                            # Fallback to current price data
                            current_data = self._fetch_crypto_current_price(symbol)
                            if current_data:
                                crypto_data[symbol] = current_data
                    
                except Exception as e:
                    print(f"Error fetching data for {symbol}: {str(e)}")
                    continue
            
            return crypto_data
            
        except Exception as e:
            raise Exception(f"Error fetching crypto data: {str(e)}")
    
    def _get_crypto_symbol_id(self, symbol):
        """Get CoinMarketCap symbol ID"""
        try:
            url = f"{self.coinmarketcap_base_url}/cryptocurrency/map"
            headers = {
                'X-CMC_PRO_API_KEY': self.coinmarketcap_api_key
            }
            params = {
                'symbol': symbol,
                'limit': 1
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data['data']:
                    return data['data'][0]['id']
            
            return None
            
        except Exception as e:
            print(f"Error getting symbol ID for {symbol}: {str(e)}")
            return None
    
    def _fetch_crypto_historical_data(self, symbol_id, period):
        """Fetch historical crypto data"""
        try:
            # Note: Historical data requires a higher tier API plan
            # This is a placeholder for the actual implementation
            
            # For now, we'll use current price data
            return None
            
        except Exception as e:
            print(f"Error fetching historical crypto data: {str(e)}")
            return None
    
    def _fetch_crypto_current_price(self, symbol):
        """Fetch current crypto price data"""
        try:
            url = f"{self.coinmarketcap_base_url}/cryptocurrency/quotes/latest"
            headers = {
                'X-CMC_PRO_API_KEY': self.coinmarketcap_api_key
            }
            params = {
                'symbol': symbol,
                'convert': 'USD'
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if symbol in data['data']:
                    quote = data['data'][symbol]['quote']['USD']
                    
                    # Create a simple DataFrame with current price
                    current_time = datetime.now()
                    price_data = {
                        'open': [quote['price']],
                        'high': [quote['price']],
                        'low': [quote['price']],
                        'close': [quote['price']],
                        'volume': [quote['volume_24h']]
                    }
                    
                    df = pd.DataFrame(price_data, index=[current_time])
                    return df
            
            return None
            
        except Exception as e:
            print(f"Error fetching current crypto price for {symbol}: {str(e)}")
            return None
    
    def fetch_stock_data(self, symbols, period='1y', interval='1d'):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL'])
            period: Time period for data
            interval: Data interval
            
        Returns:
            stock_data: Dictionary with stock data for each symbol
        """
        try:
            stock_data = {}
            
            for symbol in symbols:
                try:
                    # Create ticker object
                    ticker = yf.Ticker(symbol)
                    
                    # Fetch historical data
                    hist_data = ticker.history(period=period, interval=interval)
                    
                    if not hist_data.empty:
                        # Standardize column names
                        hist_data.columns = [col.lower() for col in hist_data.columns]
                        
                        # Ensure we have the required columns
                        required_cols = ['open', 'high', 'low', 'close', 'volume']
                        if all(col in hist_data.columns for col in required_cols):
                            stock_data[symbol] = hist_data[required_cols]
                        else:
                            print(f"Missing required columns for {symbol}")
                    
                except Exception as e:
                    print(f"Error fetching stock data for {symbol}: {str(e)}")
                    continue
            
            return stock_data
            
        except Exception as e:
            raise Exception(f"Error fetching stock data: {str(e)}")
    
    def fetch_real_time_data(self, symbols, data_type='mixed'):
        """
        Fetch real-time data for mixed assets
        
        Args:
            symbols: List of symbols
            data_type: 'crypto', 'stock', or 'mixed'
            
        Returns:
            real_time_data: Dictionary with real-time data
        """
        try:
            real_time_data = {}
            
            if data_type in ['crypto', 'mixed']:
                # Fetch crypto data
                crypto_symbols = config.DATA_CONFIG["crypto_symbols"]
                crypto_data = self.fetch_crypto_data(crypto_symbols, period='1d', interval='1h')
                real_time_data.update(crypto_data)
            
            if data_type in ['stock', 'mixed']:
                # Fetch stock data
                stock_symbols = config.DATA_CONFIG["stock_symbols"]
                stock_data = self.fetch_stock_data(stock_symbols, period='1d', interval='1h')
                real_time_data.update(stock_data)
            
            return real_time_data
            
        except Exception as e:
            raise Exception(f"Error fetching real-time data: {str(e)}")
    
    def cache_data(self, data, data_type='mixed'):
        """
        Cache data to local files
        
        Args:
            data: Data to cache
            data_type: Type of data ('crypto', 'stock', 'mixed')
        """
        try:
            timestamp = datetime.now().isoformat()
            
            if data_type in ['crypto', 'mixed']:
                # Cache crypto data
                crypto_cache = {
                    'metadata': {
                        'last_updated': timestamp,
                        'source': 'CoinMarketCap API',
                        'symbols': list(data.keys())
                    },
                    'data': {}
                }
                
                for symbol, df in data.items():
                    if symbol in config.DATA_CONFIG["crypto_symbols"]:
                        crypto_cache['data'][symbol] = df.to_dict('records')
                
                with open(config.PATHS["crypto_data"], 'w') as f:
                    json.dump(crypto_cache, f, indent=2, default=str)
            
            if data_type in ['stock', 'mixed']:
                # Cache stock data
                stock_cache = {
                    'metadata': {
                        'last_updated': timestamp,
                        'source': 'Yahoo Finance',
                        'symbols': list(data.keys())
                    },
                    'data': {}
                }
                
                for symbol, df in data.items():
                    if symbol in config.DATA_CONFIG["stock_symbols"]:
                        stock_cache['data'][symbol] = df.to_dict('records')
                
                with open(config.PATHS["stock_data"], 'w') as f:
                    json.dump(stock_cache, f, indent=2, default=str)
            
        except Exception as e:
            print(f"Error caching data: {str(e)}")
    
    def load_cached_data(self, data_type='mixed'):
        """
        Load cached data from local files
        
        Args:
            data_type: Type of data to load
            
        Returns:
            cached_data: Dictionary with cached data
        """
        try:
            cached_data = {}
            
            if data_type in ['crypto', 'mixed']:
                # Load crypto data
                crypto_file = config.PATHS["crypto_data"]
                if os.path.exists(crypto_file):
                    with open(crypto_file, 'r') as f:
                        crypto_cache = json.load(f)
                    
                    # Check if cache is still valid
                    if self._is_cache_valid(crypto_cache['metadata']['last_updated']):
                        for symbol, records in crypto_cache['data'].items():
                            df = pd.DataFrame(records)
                            if not df.empty:
                                df.index = pd.to_datetime(df.index)
                                cached_data[symbol] = df
            
            if data_type in ['stock', 'mixed']:
                # Load stock data
                stock_file = config.PATHS["stock_data"]
                if os.path.exists(stock_file):
                    with open(stock_file, 'r') as f:
                        stock_cache = json.load(f)
                    
                    # Check if cache is still valid
                    if self._is_cache_valid(stock_cache['metadata']['last_updated']):
                        for symbol, records in stock_cache['data'].items():
                            df = pd.DataFrame(records)
                            if not df.empty:
                                df.index = pd.to_datetime(df.index)
                                cached_data[symbol] = df
            
            return cached_data
            
        except Exception as e:
            print(f"Error loading cached data: {str(e)}")
            return {}
    
    def _is_cache_valid(self, timestamp_str):
        """Check if cached data is still valid"""
        try:
            if timestamp_str is None:
                return False
                
            cached_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            current_time = datetime.now()
            
            # Remove timezone info for comparison
            if cached_time.tzinfo:
                cached_time = cached_time.replace(tzinfo=None)
            
            time_diff = current_time - cached_time
            return time_diff.total_seconds() < self.cache_duration
            
        except Exception as e:
            print(f"Error checking cache validity: {str(e)}")
            return False
    
    def get_data(self, symbols=None, data_type='mixed', use_cache=True):
        """
        Get data with caching support
        
        Args:
            symbols: List of symbols (if None, use default from config)
            data_type: Type of data ('crypto', 'stock', 'mixed')
            use_cache: Whether to use cached data
            
        Returns:
            data: Dictionary with requested data
        """
        try:
            # Load cached data if requested
            if use_cache:
                cached_data = self.load_cached_data(data_type)
                if cached_data:
                    return cached_data
            
            # Fetch fresh data
            if symbols is None:
                if data_type == 'crypto':
                    symbols = config.DATA_CONFIG["crypto_symbols"]
                elif data_type == 'stock':
                    symbols = config.DATA_CONFIG["stock_symbols"]
                else:
                    symbols = config.DATA_CONFIG["crypto_symbols"] + config.DATA_CONFIG["stock_symbols"]
            
            fresh_data = self.fetch_real_time_data(symbols, data_type)
            
            # Cache the fresh data
            if fresh_data:
                self.cache_data(fresh_data, data_type)
            
            return fresh_data
            
        except Exception as e:
            raise Exception(f"Error getting data: {str(e)}")
    
    def get_market_overview(self):
        """
        Get market overview data
        
        Returns:
            overview: Dictionary with market overview
        """
        try:
            overview = {
                'crypto_market': {},
                'stock_market': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Get crypto market data
            try:
                crypto_data = self.get_data(data_type='crypto')
                overview['crypto_market'] = {
                    'symbols': list(crypto_data.keys()),
                    'status': 'active' if crypto_data else 'inactive'
                }
            except Exception as e:
                overview['crypto_market'] = {'error': str(e)}
            
            # Get stock market data
            try:
                stock_data = self.get_data(data_type='stock')
                overview['stock_market'] = {
                    'symbols': list(stock_data.keys()),
                    'status': 'active' if stock_data else 'inactive'
                }
            except Exception as e:
                overview['stock_market'] = {'error': str(e)}
            
            return overview
            
        except Exception as e:
            return {'error': f"Error getting market overview: {str(e)}"}
