"""
CoinMarketCap Top Coins Manager
Dynamically fetches and manages the top N cryptocurrencies
"""

import os
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class TopCoinsManager:
    """Manage top cryptocurrencies from CoinMarketCap"""
    
    def __init__(self):
        self.api_key = os.environ.get('COINMARKETCAP_API_KEY')
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.cache_file = 'data/top_coins_cache.json'
        self.cache_ttl = 3600  # 1 hour cache
        
        # Comprehensive Yahoo Finance symbol mapping
        self.yahoo_symbol_map = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD',
            'USDT': 'USDT-USD',
            'BNB': 'BNB-USD',
            'SOL': 'SOL-USD',
            'USDC': 'USDC-USD',
            'XRP': 'XRP-USD',
            'DOGE': 'DOGE-USD',
            'ADA': 'ADA-USD',
            'TRX': 'TRX-USD',
            'AVAX': 'AVAX-USD',
            'SHIB': 'SHIB-USD',
            'TON': 'TON11419-USD',
            'DOT': 'DOT-USD',
            'LINK': 'LINK-USD',
            'MATIC': 'MATIC-USD',
            'BCH': 'BCH-USD',
            'LTC': 'LTC-USD',
            'UNI': 'UNI-USD',
            'XLM': 'XLM-USD',
            'ATOM': 'ATOM-USD',
            'ETC': 'ETC-USD',
            'XMR': 'XMR-USD',
            'ICP': 'ICP-USD',
            'APT': 'APT21794-USD',
            'FIL': 'FIL-USD',
            'ARB': 'ARB11841-USD',
            'VET': 'VET-USD',
            'HBAR': 'HBAR-USD',
            'NEAR': 'NEAR-USD'
        }
    
    def fetch_top_coins(self, limit: int = 10) -> Optional[List[Dict]]:
        """
        Fetch top N cryptocurrencies by market cap
        
        Args:
            limit: Number of top coins to fetch (default 10)
            
        Returns:
            List of coin info dictionaries
        """
        try:
            # Check cache first
            cached_data = self._load_from_cache()
            if cached_data and len(cached_data) >= limit:
                print(f"✅ Loaded top {limit} coins from cache")
                return cached_data[:limit]
            
            if not self.api_key:
                print("⚠️ No CoinMarketCap API key, using default top 10")
                return self._get_default_top_coins(limit)
            
            print(f"🔍 Fetching top {limit} coins from CoinMarketCap...")
            
            url = f"{self.base_url}/cryptocurrency/listings/latest"
            headers = {
                'X-CMC_PRO_API_KEY': self.api_key,
                'Accept': 'application/json'
            }
            params = {
                'start': 1,
                'limit': limit,
                'convert': 'USD',
                'sort': 'market_cap'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    coins = []
                    for coin in data['data']:
                        coin_info = {
                            'id': coin['id'],
                            'name': coin['name'],
                            'symbol': coin['symbol'],
                            'rank': coin['cmc_rank'],
                            'price': coin['quote']['USD']['price'],
                            'market_cap': coin['quote']['USD']['market_cap'],
                            'volume_24h': coin['quote']['USD']['volume_24h'],
                            'percent_change_24h': coin['quote']['USD'].get('percent_change_24h', 0),
                            'yahoo_symbol': self._get_yahoo_symbol(coin['symbol']),
                            'last_updated': datetime.now().isoformat()
                        }
                        coins.append(coin_info)
                    
                    # Save to cache
                    self._save_to_cache(coins)
                    
                    print(f"✅ Fetched {len(coins)} coins from CoinMarketCap")
                    self._print_coin_summary(coins)
                    
                    return coins
                else:
                    print("⚠️ No data in API response, using defaults")
                    return self._get_default_top_coins(limit)
            else:
                print(f"⚠️ API error ({response.status_code}), using defaults")
                return self._get_default_top_coins(limit)
                
        except Exception as e:
            print(f"⚠️ Error fetching top coins: {str(e)}")
            print("   Falling back to default top 10")
            return self._get_default_top_coins(limit)
    
    def _get_yahoo_symbol(self, symbol: str) -> Optional[str]:
        """Get Yahoo Finance symbol for a crypto"""
        yahoo_sym = self.yahoo_symbol_map.get(symbol)
        
        if not yahoo_sym:
            # Default pattern for unknown symbols
            yahoo_sym = f"{symbol}-USD"
            print(f"   ⚠️ No mapping for {symbol}, using default: {yahoo_sym}")
        
        return yahoo_sym
    
    def _get_default_top_coins(self, limit: int) -> List[Dict]:
        """
        Get default top 10 coins (hardcoded fallback)
        Based on typical CoinMarketCap top 10 as of 2025
        """
        default_coins = [
            {'id': 1, 'name': 'Bitcoin', 'symbol': 'BTC', 'rank': 1, 'yahoo_symbol': 'BTC-USD'},
            {'id': 1027, 'name': 'Ethereum', 'symbol': 'ETH', 'rank': 2, 'yahoo_symbol': 'ETH-USD'},
            {'id': 825, 'name': 'Tether', 'symbol': 'USDT', 'rank': 3, 'yahoo_symbol': 'USDT-USD'},
            {'id': 1839, 'name': 'BNB', 'symbol': 'BNB', 'rank': 4, 'yahoo_symbol': 'BNB-USD'},
            {'id': 5426, 'name': 'Solana', 'symbol': 'SOL', 'rank': 5, 'yahoo_symbol': 'SOL-USD'},
            {'id': 3408, 'name': 'USD Coin', 'symbol': 'USDC', 'rank': 6, 'yahoo_symbol': 'USDC-USD'},
            {'id': 52, 'name': 'XRP', 'symbol': 'XRP', 'rank': 7, 'yahoo_symbol': 'XRP-USD'},
            {'id': 74, 'name': 'Dogecoin', 'symbol': 'DOGE', 'rank': 8, 'yahoo_symbol': 'DOGE-USD'},
            {'id': 2010, 'name': 'Cardano', 'symbol': 'ADA', 'rank': 9, 'yahoo_symbol': 'ADA-USD'},
            {'id': 2, 'name': 'Litecoin', 'symbol': 'LTC', 'rank': 10, 'yahoo_symbol': 'LTC-USD'}
        ]
        
        return default_coins[:limit]
    
    def _load_from_cache(self) -> Optional[List[Dict]]:
        """Load coins from cache if valid"""
        try:
            if not os.path.exists(self.cache_file):
                return None
            
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check cache age
            cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01'))
            age_seconds = (datetime.now() - cache_time).total_seconds()
            
            if age_seconds < self.cache_ttl:
                return cache_data.get('coins', [])
            else:
                print(f"   Cache expired ({age_seconds:.0f}s old, TTL={self.cache_ttl}s)")
                return None
                
        except Exception as e:
            print(f"   ⚠️ Error loading cache: {str(e)}")
            return None
    
    def _save_to_cache(self, coins: List[Dict]):
        """Save coins to cache"""
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'coins': coins
            }
            
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            print(f"   💾 Cached {len(coins)} coins (TTL: {self.cache_ttl}s)")
            
        except Exception as e:
            print(f"   ⚠️ Error saving cache: {str(e)}")
    
    def _print_coin_summary(self, coins: List[Dict]):
        """Print summary of fetched coins"""
        print("\n" + "=" * 70)
        print(f"{'Rank':<6} {'Symbol':<8} {'Name':<20} {'Price':<15} {'Yahoo Symbol':<15}")
        print("=" * 70)
        
        for coin in coins:
            rank = coin.get('rank', '?')
            symbol = coin.get('symbol', '?')
            name = coin.get('name', 'Unknown')[:20]
            price = coin.get('price')
            yahoo = coin.get('yahoo_symbol', 'N/A')
            
            if price:
                price_str = f"${price:,.2f}" if price >= 1 else f"${price:.6f}"
            else:
                price_str = "N/A"
            
            print(f"{rank:<6} {symbol:<8} {name:<20} {price_str:<15} {yahoo:<15}")
        
        print("=" * 70 + "\n")
    
    def get_symbols_list(self, limit: int = 10) -> List[str]:
        """Get list of crypto symbols (BTC, ETH, etc.)"""
        coins = self.fetch_top_coins(limit)
        if coins:
            return [coin['symbol'] for coin in coins]
        return []
    
    def get_yahoo_symbols_list(self, limit: int = 10) -> List[str]:
        """Get list of Yahoo Finance symbols (BTC-USD, ETH-USD, etc.)"""
        coins = self.fetch_top_coins(limit)
        if coins:
            return [coin['yahoo_symbol'] for coin in coins if coin.get('yahoo_symbol')]
        return []
    
    def get_symbol_mapping(self, limit: int = 10) -> Dict[str, str]:
        """Get mapping of crypto symbol to Yahoo symbol"""
        coins = self.fetch_top_coins(limit)
        if coins:
            return {coin['symbol']: coin['yahoo_symbol'] for coin in coins if coin.get('yahoo_symbol')}
        return {}


# Example usage and testing
if __name__ == '__main__':
    print("🚀 Top Coins Manager Test")
    print("=" * 70)
    
    manager = TopCoinsManager()
    
    # Test 1: Fetch top 10 coins
    print("\n📊 Test 1: Fetch Top 10 Coins")
    top_10 = manager.fetch_top_coins(10)
    
    # Test 2: Get symbol lists
    print("\n📊 Test 2: Get Symbol Lists")
    symbols = manager.get_symbols_list(10)
    print(f"Crypto Symbols: {symbols}")
    
    yahoo_symbols = manager.get_yahoo_symbols_list(10)
    print(f"Yahoo Symbols: {yahoo_symbols}")
    
    # Test 3: Get mapping
    print("\n📊 Test 3: Get Symbol Mapping")
    mapping = manager.get_symbol_mapping(10)
    for crypto, yahoo in mapping.items():
        print(f"   {crypto} → {yahoo}")
    
    # Test 4: Cache test
    print("\n📊 Test 4: Cache Test (should load from cache)")
    top_10_cached = manager.fetch_top_coins(10)
    
    print("\n✅ All tests completed!")
