"""
Enhanced Crypto Data Fetcher
Supports dynamic top N coins with robust error handling
"""

import yfinance as yf
import requests
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional

# Handle imports from different paths
try:
    from data.top_coins_manager import TopCoinsManager
except ImportError:
    from top_coins_manager import TopCoinsManager

class EnhancedCryptoFetcher:
    """Enhanced crypto data fetcher with top N coins support"""
    
    def __init__(self):
        self.cmc_api_key = os.environ.get('COINMARKETCAP_API_KEY')
        self.cmc_base_url = "https://pro-api.coinmarketcap.com/v1"
        self.coins_manager = TopCoinsManager()
        
        # Track failed symbols
        self.failed_symbols = set()
        
        # Success/failure stats
        self.stats = {
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }
    
    def fetch_top_n_coins_data(self, n: int = 10, period: str = '1y', 
                               interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for top N coins from CoinMarketCap
        
        Args:
            n: Number of top coins to fetch (default 10)
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            Dict of {symbol: DataFrame} with OHLCV data
        """
        print(f"\n🚀 Fetching Top {n} Cryptocurrencies")
        print("=" * 70)
        
        # Reset stats
        self.stats = {'successful': 0, 'failed': 0, 'skipped': 0}
        self.failed_symbols.clear()
        
        # Get top N coins
        top_coins = self.coins_manager.fetch_top_coins(n)
        
        if not top_coins:
            print("❌ Failed to fetch top coins list")
            return {}
        
        print(f"\n📈 Fetching historical data for {len(top_coins)} coins...")
        print(f"   Period: {period} | Interval: {interval}")
        print("=" * 70)
        
        results = {}
        
        for i, coin in enumerate(top_coins, 1):
            symbol = coin['symbol']
            yahoo_symbol = coin['yahoo_symbol']
            name = coin['name']
            
            print(f"\n[{i}/{len(top_coins)}] {symbol} ({name})")
            
            try:
                # Fetch data
                df = self._fetch_yahoo_data(yahoo_symbol, period, interval)
                
                if df is not None and not df.empty:
                    results[symbol] = df
                    self.stats['successful'] += 1
                    
                    # Print summary
                    print(f"   ✅ Success: {len(df)} data points")
                    print(f"      Latest: ${df['close'].iloc[-1]:,.2f}")
                    print(f"      Range: {df.index[0].date()} to {df.index[-1].date()}")
                else:
                    print(f"   ❌ Failed: No data returned")
                    self.failed_symbols.add(symbol)
                    self.stats['failed'] += 1
                
                # Rate limiting
                time.sleep(0.3)
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                self.failed_symbols.add(symbol)
                self.stats['failed'] += 1
        
        # Print summary
        self._print_summary(results, n)
        
        return results
    
    def _fetch_yahoo_data(self, yahoo_symbol: str, period: str, 
                         interval: str) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance with error handling"""
        try:
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return None
            
            # Standardize columns
            df.columns = df.columns.str.lower()
            if 'adj close' in df.columns:
                df = df.rename(columns={'adj close': 'adj_close'})
            
            # Keep only OHLCV
            available_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            df = df[available_cols]
            
            return df
            
        except Exception as e:
            raise Exception(f"Yahoo Finance error: {str(e)}")
    
    def fetch_current_prices(self, symbols: Optional[List[str]] = None, 
                           top_n: int = 10) -> Dict[str, Dict]:
        """
        Fetch current prices for multiple symbols
        
        Args:
            symbols: List of symbols (if None, uses top N)
            top_n: Number of top coins if symbols not provided
            
        Returns:
            Dict of {symbol: price_data}
        """
        if symbols is None:
            top_coins = self.coins_manager.fetch_top_coins(top_n)
            symbols = [coin['symbol'] for coin in top_coins]
        
        print(f"\n💰 Fetching current prices for {len(symbols)} coins...")
        print("=" * 70)
        
        results = {}
        
        for symbol in symbols:
            try:
                price_data = self._fetch_current_price_cmc(symbol)
                
                if price_data:
                    results[symbol] = price_data
                    print(f"   {symbol}: ${price_data['price']:,.2f} "
                          f"({price_data.get('percent_change_24h', 0):+.2f}%)")
                else:
                    print(f"   {symbol}: Failed to fetch")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"   {symbol}: Error - {str(e)}")
        
        print("=" * 70)
        print(f"✅ Fetched prices for {len(results)}/{len(symbols)} coins\n")
        
        return results
    
    def _fetch_current_price_cmc(self, symbol: str) -> Optional[Dict]:
        """Fetch current price from CoinMarketCap"""
        try:
            if not self.cmc_api_key:
                return self._fetch_current_price_yahoo(symbol)
            
            url = f"{self.cmc_base_url}/cryptocurrency/quotes/latest"
            headers = {
                'X-CMC_PRO_API_KEY': self.cmc_api_key,
                'Accept': 'application/json'
            }
            params = {
                'symbol': symbol,
                'convert': 'USD'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and symbol in data['data']:
                    quote = data['data'][symbol]['quote']['USD']
                    
                    return {
                        'symbol': symbol,
                        'price': quote['price'],
                        'volume_24h': quote.get('volume_24h', 0),
                        'percent_change_1h': quote.get('percent_change_1h', 0),
                        'percent_change_24h': quote.get('percent_change_24h', 0),
                        'percent_change_7d': quote.get('percent_change_7d', 0),
                        'market_cap': quote.get('market_cap', 0),
                        'last_updated': quote.get('last_updated', datetime.now().isoformat())
                    }
            
            # Fallback to Yahoo
            return self._fetch_current_price_yahoo(symbol)
            
        except Exception:
            return self._fetch_current_price_yahoo(symbol)
    
    def _fetch_current_price_yahoo(self, symbol: str) -> Optional[Dict]:
        """Fallback: Fetch from Yahoo Finance"""
        try:
            # Get Yahoo symbol
            yahoo_symbol = self.coins_manager.yahoo_symbol_map.get(symbol, f'{symbol}-USD')
            
            ticker = yf.Ticker(yahoo_symbol)
            hist = ticker.history(period='1d', interval='1m')
            
            if hist.empty:
                return None
            
            latest = hist.iloc[-1]
            info = ticker.info
            previous_close = info.get('previousClose', latest['Close'])
            
            return {
                'symbol': symbol,
                'price': float(latest['Close']),
                'volume_24h': float(hist['Volume'].sum()),
                'percent_change_24h': ((latest['Close'] - previous_close) / previous_close * 100),
                'last_updated': datetime.now().isoformat(),
                'source': 'yahoo'
            }
            
        except Exception:
            return None
    
    def _print_summary(self, results: Dict, target_count: int):
        """Print fetch summary"""
        print("\n" + "=" * 70)
        print("📊 FETCH SUMMARY")
        print("=" * 70)
        print(f"   Target Coins: {target_count}")
        print(f"   ✅ Successful: {self.stats['successful']}")
        print(f"   ❌ Failed: {self.stats['failed']}")
        
        if self.failed_symbols:
            print(f"\n   Failed Symbols: {', '.join(self.failed_symbols)}")
        
        if results:
            print(f"\n   📈 Data Statistics:")
            total_points = sum(len(df) for df in results.values())
            avg_points = total_points / len(results)
            print(f"      Total Data Points: {total_points}")
            print(f"      Average per Coin: {avg_points:.0f}")
            
            # Date range
            all_dates = []
            for df in results.values():
                all_dates.extend([df.index[0], df.index[-1]])
            
            if all_dates:
                print(f"      Date Range: {min(all_dates).date()} to {max(all_dates).date()}")
        
        print("=" * 70 + "\n")
    
    def save_to_cache(self, data: Dict[str, pd.DataFrame], filename: str = 'crypto_top10_cache.json'):
        """Save data to JSON cache"""
        try:
            cache_data = {}
            
            for symbol, df in data.items():
                cache_data[symbol] = {
                    'data': df.reset_index().to_dict(orient='records'),
                    'last_updated': datetime.now().isoformat(),
                    'rows': len(df)
                }
            
            cache_path = f'data/{filename}'
            import json
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            print(f"💾 Cached {len(data)} coins to: {cache_path}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error saving cache: {str(e)}")
            return False
    
    def get_portfolio_summary(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate portfolio summary statistics"""
        summary_data = []
        
        for symbol, df in data.items():
            if df.empty:
                continue
            
            summary = {
                'symbol': symbol,
                'days': len(df),
                'latest_price': df['close'].iloc[-1],
                'highest': df['close'].max(),
                'lowest': df['close'].min(),
                'total_return_%': ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100),
                'volatility_%': (df['close'].pct_change().std() * 100),
                'avg_volume': df['volume'].mean()
            }
            summary_data.append(summary)
        
        df_summary = pd.DataFrame(summary_data)
        
        if not df_summary.empty:
            df_summary = df_summary.sort_values('total_return_%', ascending=False)
        
        return df_summary


# Example usage and testing
if __name__ == '__main__':
    print("\n🚀 Enhanced Crypto Fetcher - Top 10 Coins Test")
    print("=" * 70)
    
    fetcher = EnhancedCryptoFetcher()
    
    # Test 1: Fetch top 10 coins (6 months)
    print("\n📊 Test 1: Fetch Top 10 Historical Data (6 months)")
    data = fetcher.fetch_top_n_coins_data(n=10, period='6mo')
    
    # Test 2: Portfolio summary
    if data:
        print("\n📊 Test 2: Portfolio Summary")
        summary = fetcher.get_portfolio_summary(data)
        print("\n" + summary.to_string(index=False))
        
        # Save cache
        fetcher.save_to_cache(data)
    
    # Test 3: Current prices
    print("\n📊 Test 3: Current Prices (Top 5)")
    current = fetcher.fetch_current_prices(top_n=5)
    
    print("\n✅ All tests completed!")
