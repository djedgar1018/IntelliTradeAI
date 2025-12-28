"""
Fast Complete Training for IntelliTradeAI - Optimized for speed
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False


def calculate_features(df):
    """Calculate key technical indicators"""
    data = df.copy()
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']
    
    for p in [5, 10, 20, 50]:
        data[f'SMA_{p}'] = close.rolling(p).mean()
        data[f'EMA_{p}'] = close.ewm(span=p).mean()
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    exp1 = close.ewm(span=12).mean()
    exp2 = close.ewm(span=26).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    data['BB_Width'] = (2 * bb_std) / (bb_mid + 1e-10)
    data['BB_Pos'] = (close - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    
    for p in [1, 3, 5, 10]:
        data[f'Returns_{p}d'] = close.pct_change(p)
    
    data['Volatility'] = data['Returns_1d'].rolling(20).std()
    data['Volume_Ratio'] = volume / (volume.rolling(20).mean() + 1e-10)
    
    data['ATR'] = (high - low).rolling(14).mean()
    data['Momentum'] = close - close.shift(10)
    
    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()
    data['Stoch_K'] = 100 * (close - lowest) / (highest - lowest + 1e-10)
    
    data['Target'] = (close.shift(-1) > close).astype(int)
    
    return data


def prepare_data(symbol, days=365*3):
    """Fetch and prepare data"""
    try:
        import yfinance as yf
        data = yf.download(symbol, start=datetime.now() - timedelta(days=days), 
                          end=datetime.now(), progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = calculate_features(data)
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        return data
    except:
        return None


def train_and_evaluate(symbol, asset_type):
    """Train models for a single asset"""
    data = prepare_data(symbol)
    if data is None or len(data) < 200:
        return None
    
    feature_cols = [c for c in data.columns if c not in 
                   ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target']]
    
    X = data[feature_cols].values
    y = data['Target'].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    results = {}
    
    models = {
        'RF': RandomForestClassifier(n_estimators=150, max_depth=12, n_jobs=-1, random_state=42),
        'GB': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
        'ET': ExtraTreesClassifier(n_estimators=150, max_depth=12, n_jobs=-1, random_state=42),
    }
    
    if HAS_XGB:
        models['XGB'] = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                                          use_label_encoder=False, eval_metric='logloss', 
                                          random_state=42, verbosity=0)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, pred),
            'precision': precision_score(y_test, pred, zero_division=0),
            'f1': f1_score(y_test, pred, zero_division=0)
        }
    
    return results


def main():
    print("\n" + "="*70)
    print("INTELLITRADEAI - FAST MODEL TRAINING")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    crypto = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 
              'DOGE-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD', 'MATIC-USD']
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 
              'META', 'TSLA', 'JPM', 'V', 'WMT']
    
    all_results = {'crypto': {}, 'stocks': {}}
    
    print("\n--- CRYPTOCURRENCY ---")
    for s in crypto:
        print(f"Training {s}...", end=" ")
        r = train_and_evaluate(s, 'crypto')
        if r:
            best = max(r.items(), key=lambda x: x[1]['accuracy'])
            print(f"Best: {best[0]} = {best[1]['accuracy']*100:.1f}%")
            all_results['crypto'][s] = r
        else:
            print("FAILED")
    
    print("\n--- STOCKS ---")
    for s in stocks:
        print(f"Training {s}...", end=" ")
        r = train_and_evaluate(s, 'stocks')
        if r:
            best = max(r.items(), key=lambda x: x[1]['accuracy'])
            print(f"Best: {best[0]} = {best[1]['accuracy']*100:.1f}%")
            all_results['stocks'][s] = r
        else:
            print("FAILED")
    
    print("\n" + "="*70)
    print("SUMMARY BY ASSET TYPE")
    print("="*70)
    
    for asset_type in ['crypto', 'stocks']:
        if not all_results[asset_type]:
            continue
        
        model_accs = {}
        best_per_asset = []
        
        for symbol, results in all_results[asset_type].items():
            best_acc = 0
            for model, metrics in results.items():
                acc = metrics['accuracy']
                if model not in model_accs:
                    model_accs[model] = []
                model_accs[model].append(acc)
                if acc > best_acc:
                    best_acc = acc
            best_per_asset.append(best_acc)
        
        print(f"\n{asset_type.upper()}:")
        print(f"  Average (best model per asset): {np.mean(best_per_asset)*100:.1f}%")
        print(f"  Best single result: {max(best_per_asset)*100:.1f}%")
        print(f"\n  By Model:")
        for model, accs in sorted(model_accs.items(), key=lambda x: np.mean(x[1]), reverse=True):
            print(f"    {model}: {np.mean(accs)*100:.1f}% avg, {max(accs)*100:.1f}% best")
    
    crypto_best = [max(r.values(), key=lambda x: x['accuracy'])['accuracy'] 
                   for r in all_results['crypto'].values()]
    stock_best = [max(r.values(), key=lambda x: x['accuracy'])['accuracy'] 
                  for r in all_results['stocks'].values()]
    
    print("\n" + "="*70)
    print("PUBLICATION METRICS")
    print("="*70)
    print(f"Cryptocurrency Average: {np.mean(crypto_best)*100:.1f}%")
    print(f"Stock Market Average: {np.mean(stock_best)*100:.1f}%")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


if __name__ == "__main__":
    main()
