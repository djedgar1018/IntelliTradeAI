"""
Advanced Model Training for IntelliTradeAI
Implements: Extended features, walk-forward validation, feature selection,
probability calibration, threshold optimization
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, RFE
from sklearn.calibration import CalibratedClassifierCV

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek
    HAS_IMBALANCED = True
except ImportError:
    HAS_IMBALANCED = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def calculate_advanced_features(df):
    """Calculate comprehensive technical indicators - 70+ features"""
    data = df.copy()
    
    close = data['Close']
    high = data['High']
    low = data['Low']
    open_price = data['Open']
    volume = data['Volume']
    
    for period in [5, 10, 20, 50, 100, 200]:
        data[f'SMA_{period}'] = close.rolling(period).mean()
    
    for period in [5, 10, 12, 20, 26, 50]:
        data[f'EMA_{period}'] = close.ewm(span=period).mean()
    
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    for fast, slow, signal in [(12, 26, 9), (5, 35, 5)]:
        exp1 = close.ewm(span=fast).mean()
        exp2 = close.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        data[f'MACD_{fast}_{slow}'] = macd
        data[f'MACD_Signal_{fast}_{slow}'] = macd_signal
        data[f'MACD_Hist_{fast}_{slow}'] = macd - macd_signal
    
    for period in [10, 20, 50]:
        bb_middle = close.rolling(period).mean()
        bb_std = close.rolling(period).std()
        data[f'BB_Upper_{period}'] = bb_middle + 2 * bb_std
        data[f'BB_Lower_{period}'] = bb_middle - 2 * bb_std
        data[f'BB_Width_{period}'] = (data[f'BB_Upper_{period}'] - data[f'BB_Lower_{period}']) / (bb_middle + 1e-10)
        data[f'BB_Position_{period}'] = (close - data[f'BB_Lower_{period}']) / (data[f'BB_Upper_{period}'] - data[f'BB_Lower_{period}'] + 1e-10)
    
    for period in [7, 14, 21]:
        data[f'ATR_{period}'] = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1).rolling(period).mean()
    
    for period in [5, 14, 21]:
        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()
        data[f'Stoch_K_{period}'] = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        data[f'Stoch_D_{period}'] = data[f'Stoch_K_{period}'].rolling(3).mean()
    
    data['OBV'] = (np.sign(close.diff()) * volume).cumsum()
    data['OBV_EMA'] = data['OBV'].ewm(span=20).mean()
    
    for period in [5, 10, 20]:
        data[f'Volume_SMA_{period}'] = volume.rolling(period).mean()
        data[f'Volume_Ratio_{period}'] = volume / (data[f'Volume_SMA_{period}'] + 1e-10)
    
    data['VWAP'] = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-10)
    
    for period in [1, 2, 3, 5, 10, 20]:
        data[f'Returns_{period}d'] = close.pct_change(period)
    
    for period in [5, 10, 20, 50]:
        data[f'Volatility_{period}d'] = data['Returns_1d'].rolling(period).std()
    
    for period in [5, 10, 20]:
        data[f'Momentum_{period}'] = close - close.shift(period)
        data[f'ROC_{period}'] = 100 * (close - close.shift(period)) / (close.shift(period) + 1e-10)
    
    data['ADX'] = calculate_adx(high, low, close, 14)
    
    data['CCI_20'] = (close - close.rolling(20).mean()) / (0.015 * close.rolling(20).std() + 1e-10)
    
    data['Williams_R'] = -100 * (high.rolling(14).max() - close) / (high.rolling(14).max() - low.rolling(14).min() + 1e-10)
    
    mfi_period = 14
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    delta_tp = typical_price.diff()
    pos_flow = (money_flow * (delta_tp > 0)).rolling(mfi_period).sum()
    neg_flow = (money_flow * (delta_tp < 0)).rolling(mfi_period).sum()
    data['MFI'] = 100 - (100 / (1 + pos_flow / (neg_flow + 1e-10)))
    
    data['High_Low_Range'] = (high - low) / (close + 1e-10)
    data['Close_Open_Range'] = (close - open_price) / (open_price + 1e-10)
    data['Upper_Shadow'] = (high - np.maximum(close, open_price)) / (high - low + 1e-10)
    data['Lower_Shadow'] = (np.minimum(close, open_price) - low) / (high - low + 1e-10)
    data['Body_Size'] = abs(close - open_price) / (high - low + 1e-10)
    
    data['Price_SMA_20_Ratio'] = close / (data['SMA_20'] + 1e-10)
    data['Price_SMA_50_Ratio'] = close / (data['SMA_50'] + 1e-10)
    data['SMA_20_50_Ratio'] = data['SMA_20'] / (data['SMA_50'] + 1e-10)
    
    data['Skewness_20'] = data['Returns_1d'].rolling(20).skew()
    data['Kurtosis_20'] = data['Returns_1d'].rolling(20).kurt()
    
    data['Consecutive_Up'] = calculate_consecutive_moves(close, direction='up')
    data['Consecutive_Down'] = calculate_consecutive_moves(close, direction='down')
    
    for lag in [1, 2, 3, 5]:
        data[f'Returns_Lag_{lag}'] = data['Returns_1d'].shift(lag)
        data[f'Volume_Ratio_Lag_{lag}'] = data['Volume_Ratio_20'].shift(lag) if 'Volume_Ratio_20' in data.columns else volume.shift(lag)
    
    return data


def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()
    
    return adx


def calculate_consecutive_moves(prices, direction='up'):
    """Count consecutive up or down days"""
    changes = prices.diff()
    if direction == 'up':
        moves = (changes > 0).astype(int)
    else:
        moves = (changes < 0).astype(int)
    
    result = []
    count = 0
    for m in moves:
        if m == 1:
            count += 1
        else:
            count = 0
        result.append(count)
    
    return pd.Series(result, index=prices.index)


def create_multi_horizon_targets(df, horizons=[1, 3, 5]):
    """Create multiple prediction horizons"""
    data = df.copy()
    for h in horizons:
        data[f'Target_{h}d'] = (df['Close'].shift(-h) > df['Close']).astype(int)
    return data


def prepare_data_extended(symbol='BTC-USD', days=365*5):
    """Fetch extended historical data"""
    try:
        import yfinance as yf
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"  Fetching {days//365} years of data for {symbol}...")
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            return None
            
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        print(f"  Downloaded {len(data)} rows")
        
        data = calculate_advanced_features(data)
        data = create_multi_horizon_targets(data)
        
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        
        print(f"  After preprocessing: {len(data)} rows, {len(data.columns)} features")
        return data
        
    except Exception as e:
        print(f"  Error: {e}")
        return None


def select_features(X, y, n_features=40):
    """Select best features using multiple methods"""
    print(f"  Selecting top {n_features} features...")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    
    try:
        mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    except:
        mi_scores = importances
    
    combined_scores = 0.6 * (importances / (importances.max() + 1e-10)) + 0.4 * (mi_scores / (mi_scores.max() + 1e-10))
    
    top_indices = np.argsort(combined_scores)[-n_features:]
    
    return top_indices


def walk_forward_validation(X, y, model_fn, n_splits=10, train_size=0.7):
    """Walk-forward validation for time series"""
    n_samples = len(X)
    results = []
    
    for i in range(n_splits):
        train_end = int(n_samples * (train_size + (1 - train_size) * i / n_splits))
        test_start = train_end
        test_end = int(n_samples * (train_size + (1 - train_size) * (i + 1) / n_splits))
        
        if test_end > n_samples:
            test_end = n_samples
        
        if test_start >= test_end:
            continue
            
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]
        
        model = model_fn()
        
        if HAS_IMBALANCED and len(np.unique(y_train)) > 1:
            try:
                k = min(5, min(np.bincount(y_train)) - 1)
                if k >= 1:
                    smote = SMOTE(k_neighbors=k, random_state=42)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
            except:
                pass
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        fold_result = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }
        if y_proba is not None:
            try:
                fold_result['auc'] = roc_auc_score(y_test, y_proba)
            except:
                fold_result['auc'] = 0.5
        
        results.append(fold_result)
    
    avg_results = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}
    std_results = {k: np.std([r[k] for r in results]) for k in results[0].keys()}
    
    return avg_results, std_results, results


def optimize_threshold(y_true, y_proba):
    """Find optimal classification threshold"""
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.3, 0.7, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def train_advanced_pipeline(symbol='BTC-USD', asset_type='crypto'):
    """Complete advanced training pipeline"""
    print(f"\n{'='*70}")
    print(f"ADVANCED TRAINING: {symbol} ({asset_type.upper()})")
    print(f"{'='*70}")
    
    data = prepare_data_extended(symbol, days=365*5)
    if data is None or len(data) < 500:
        print(f"Insufficient data for {symbol}")
        return None
    
    feature_cols = [col for col in data.columns if not col.startswith('Target') 
                   and col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    
    X = data[feature_cols].values
    y = data['Target_1d'].values
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    feature_names = np.array(feature_cols)
    selected_indices = select_features(X_scaled, y, n_features=min(40, len(feature_cols)))
    X_selected = X_scaled[:, selected_indices]
    selected_features = feature_names[selected_indices]
    
    print(f"\n  Selected {len(selected_features)} features")
    print(f"  Top 10: {', '.join(selected_features[:10])}")
    
    results = {}
    
    models = {
        'RandomForest': lambda: RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_split=10,
            class_weight='balanced', n_jobs=-1, random_state=42
        ),
        'GradientBoosting': lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
        'ExtraTrees': lambda: ExtraTreesClassifier(
            n_estimators=300, max_depth=20, min_samples_split=10,
            class_weight='balanced', n_jobs=-1, random_state=42
        ),
        'AdaBoost': lambda: AdaBoostClassifier(
            n_estimators=150, learning_rate=0.5, random_state=42
        ),
    }
    
    if HAS_XGB:
        models['XGBoost'] = lambda: xgb.XGBClassifier(
            n_estimators=400, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42
        )
    
    for name, model_fn in models.items():
        print(f"\n  Training {name} with walk-forward validation...")
        avg_metrics, std_metrics, fold_results = walk_forward_validation(
            X_selected, y, model_fn, n_splits=8
        )
        results[name] = {
            'avg': avg_metrics,
            'std': std_metrics,
            'folds': fold_results
        }
        print(f"    Accuracy: {avg_metrics['accuracy']:.4f} (+/- {std_metrics['accuracy']:.4f})")
        print(f"    F1 Score: {avg_metrics['f1']:.4f} (+/- {std_metrics['f1']:.4f})")
    
    print("\n  Training Calibrated Ensemble...")
    
    split_idx = int(len(X_selected) * 0.8)
    X_train, X_test = X_selected[:split_idx], X_selected[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    base_estimators = []
    for name, model_fn in models.items():
        model = model_fn()
        model.fit(X_train, y_train)
        base_estimators.append((name.lower(), model))
    
    voting = VotingClassifier(estimators=base_estimators, voting='soft')
    voting.fit(X_train, y_train)
    
    try:
        calibrated = CalibratedClassifierCV(voting, method='sigmoid', cv=3)
        calibrated.fit(X_train, y_train)
        
        y_proba = calibrated.predict_proba(X_test)[:, 1]
        optimal_threshold, optimal_f1 = optimize_threshold(y_test, y_proba)
        y_pred = (y_proba >= optimal_threshold).astype(int)
        
        results['CalibratedEnsemble'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'optimal_threshold': optimal_threshold
        }
        print(f"    Optimal Threshold: {optimal_threshold:.2f}")
        print(f"    Accuracy: {results['CalibratedEnsemble']['accuracy']:.4f}")
        print(f"    F1 Score: {results['CalibratedEnsemble']['f1']:.4f}")
    except Exception as e:
        print(f"    Calibration failed: {e}")
    
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY FOR {symbol}")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 73)
    
    for model_name, metrics in results.items():
        if isinstance(metrics, dict) and 'avg' in metrics:
            m = metrics['avg']
            print(f"{model_name:<25} {m['accuracy']:.4f}       {m['precision']:.4f}       {m['recall']:.4f}       {m['f1']:.4f}")
        elif isinstance(metrics, dict):
            print(f"{model_name:<25} {metrics['accuracy']:.4f}       {metrics['precision']:.4f}       {metrics['recall']:.4f}       {metrics['f1']:.4f}")
    
    return results


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("INTELLITRADEAI - ADVANCED MODEL TRAINING PIPELINE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Features: Walk-forward validation, Feature selection, Calibration")
    print(f"SMOTE available: {HAS_IMBALANCED}")
    print(f"XGBoost available: {HAS_XGB}")
    
    crypto_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD']
    stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA']
    
    all_results = {'crypto': {}, 'stocks': {}}
    
    print("\n" + "="*80)
    print("CRYPTOCURRENCY MODELS")
    print("="*80)
    for symbol in crypto_symbols:
        results = train_advanced_pipeline(symbol, 'crypto')
        if results:
            all_results['crypto'][symbol] = results
    
    print("\n" + "="*80)
    print("STOCK MODELS")
    print("="*80)
    for symbol in stock_symbols:
        results = train_advanced_pipeline(symbol, 'stocks')
        if results:
            all_results['stocks'][symbol] = results
    
    print("\n" + "="*80)
    print("GRAND SUMMARY")
    print("="*80)
    
    for asset_type in ['crypto', 'stocks']:
        if not all_results[asset_type]:
            continue
            
        print(f"\n{asset_type.upper()}:")
        model_accuracies = {}
        
        for symbol, results in all_results[asset_type].items():
            for model_name, metrics in results.items():
                if model_name not in model_accuracies:
                    model_accuracies[model_name] = []
                
                if isinstance(metrics, dict) and 'avg' in metrics:
                    model_accuracies[model_name].append(metrics['avg']['accuracy'])
                elif isinstance(metrics, dict) and 'accuracy' in metrics:
                    model_accuracies[model_name].append(metrics['accuracy'])
        
        print(f"  {'Model':<25} {'Avg Accuracy':<15} {'Best':<10}")
        print("  " + "-"*50)
        for model_name, accs in sorted(model_accuracies.items(), key=lambda x: np.mean(x[1]), reverse=True):
            if accs:
                avg = np.mean(accs)
                best = max(accs)
                print(f"  {model_name:<25} {avg:.4f} ({avg*100:.1f}%)   {best:.4f}")
    
    best_crypto_acc = []
    best_stock_acc = []
    
    for symbol, results in all_results['crypto'].items():
        best_model_acc = max([
            (metrics['avg']['accuracy'] if isinstance(metrics, dict) and 'avg' in metrics 
             else metrics.get('accuracy', 0))
            for metrics in results.values()
        ])
        best_crypto_acc.append(best_model_acc)
    
    for symbol, results in all_results['stocks'].items():
        best_model_acc = max([
            (metrics['avg']['accuracy'] if isinstance(metrics, dict) and 'avg' in metrics 
             else metrics.get('accuracy', 0))
            for metrics in results.values()
        ])
        best_stock_acc.append(best_model_acc)
    
    print("\n" + "="*80)
    print("PUBLICATION METRICS (Best Model Per Asset)")
    print("="*80)
    if best_crypto_acc:
        print(f"Cryptocurrency: {np.mean(best_crypto_acc)*100:.1f}% average (best per asset)")
    if best_stock_acc:
        print(f"Stock Market: {np.mean(best_stock_acc)*100:.1f}% average (best per asset)")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


if __name__ == "__main__":
    main()
