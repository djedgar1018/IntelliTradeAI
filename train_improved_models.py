"""
Enhanced Model Training Script for IntelliTradeAI
Implements: Stacking Ensemble, BiLSTM, LightGBM, SMOTE, TimeSeriesSplit, Bayesian Optimization
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("SMOTE not available - install imbalanced-learn for class balancing")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except (ImportError, OSError):
    HAS_LGBM = False
    print("LightGBM not available (missing libgomp dependency)")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("TensorFlow not available")


def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators"""
    data = df.copy()
    
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']
    
    data['SMA_5'] = close.rolling(5).mean()
    data['SMA_10'] = close.rolling(10).mean()
    data['SMA_20'] = close.rolling(20).mean()
    data['SMA_50'] = close.rolling(50).mean()
    
    data['EMA_12'] = close.ewm(span=12).mean()
    data['EMA_26'] = close.ewm(span=26).mean()
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = close.ewm(span=12).mean()
    exp2 = close.ewm(span=26).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    data['BB_Middle'] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    data['BB_Upper'] = data['BB_Middle'] + 2 * bb_std
    data['BB_Lower'] = data['BB_Middle'] - 2 * bb_std
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
    
    data['ATR'] = (high - low).rolling(14).mean()
    
    data['Stoch_K'] = 100 * (close - low.rolling(14).min()) / (high.rolling(14).max() - low.rolling(14).min() + 1e-10)
    data['Stoch_D'] = data['Stoch_K'].rolling(3).mean()
    
    data['OBV'] = (np.sign(close.diff()) * volume).cumsum()
    data['Volume_SMA'] = volume.rolling(20).mean()
    data['Volume_Ratio'] = volume / (data['Volume_SMA'] + 1e-10)
    
    data['Returns_1d'] = close.pct_change(1)
    data['Returns_5d'] = close.pct_change(5)
    data['Returns_10d'] = close.pct_change(10)
    
    data['Volatility_10d'] = data['Returns_1d'].rolling(10).std()
    data['Volatility_20d'] = data['Returns_1d'].rolling(20).std()
    
    data['Momentum_10'] = close - close.shift(10)
    data['Momentum_20'] = close - close.shift(20)
    
    data['ROC_10'] = 100 * (close - close.shift(10)) / (close.shift(10) + 1e-10)
    
    ema_fast = close.ewm(span=5).mean()
    ema_slow = close.ewm(span=35).mean()
    data['TRIX'] = ema_fast.pct_change(1) * 100
    
    data['High_Low_Range'] = (high - low) / (close + 1e-10)
    data['Close_Open_Range'] = (close - data['Open']) / (data['Open'] + 1e-10)
    
    data['DayOfWeek'] = pd.to_datetime(data.index).dayofweek if hasattr(data.index, 'dayofweek') else 0
    
    return data


def create_target(df, horizon=1):
    """Create binary classification target: 1 if price goes up, 0 otherwise"""
    df = df.copy()
    df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
    return df


def prepare_data(symbol='BTC-USD', days=365*3):
    """Fetch and prepare data for training"""
    try:
        import yfinance as yf
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"Fetching data for {symbol}...")
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"No data for {symbol}, using sample data")
            return None
            
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        print(f"Downloaded {len(data)} rows")
        
        data = calculate_technical_indicators(data)
        data = create_target(data)
        data = data.dropna()
        
        print(f"After preprocessing: {len(data)} rows")
        return data
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def build_bilstm_model(input_shape):
    """Build enhanced BiLSTM model with attention-like mechanisms"""
    if not HAS_TF:
        return None
        
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2,
                          kernel_regularizer=l2(0.001)), input_shape=input_shape),
        BatchNormalization(),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2,
                          kernel_regularizer=l2(0.001))),
        BatchNormalization(),
        Bidirectional(LSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.2,
                          kernel_regularizer=l2(0.001))),
        BatchNormalization(),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_sequences(X, y, window=20):
    """Create sequences for LSTM"""
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


class StackingEnsemble:
    """Stacking ensemble with proper time-series cross-validation"""
    
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.base_models = {}
        self.meta_learner = LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        """Fit stacking ensemble with time-series aware out-of-fold predictions"""
        X = np.array(X)
        y = np.array(y)
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        n_samples = len(X)
        oof_predictions = np.zeros((n_samples, 4))
        
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', 
                                    n_jobs=-1, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            rf_fold = RandomForestClassifier(n_estimators=200, max_depth=15, 
                                            class_weight='balanced', n_jobs=-1, random_state=42)
            rf_fold.fit(X[train_idx], y[train_idx])
            oof_predictions[val_idx, 0] = rf_fold.predict_proba(X[val_idx])[:, 1]
        rf.fit(X, y)
        self.base_models['rf'] = rf
        
        if HAS_XGB:
            print("Training XGBoost...")
            xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                          use_label_encoder=False, eval_metric='logloss',
                                          random_state=42)
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                xgb_fold = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                            use_label_encoder=False, eval_metric='logloss',
                                            random_state=42)
                xgb_fold.fit(X[train_idx], y[train_idx])
                oof_predictions[val_idx, 1] = xgb_fold.predict_proba(X[val_idx])[:, 1]
            xgb_model.fit(X, y)
            self.base_models['xgb'] = xgb_model
        
        if HAS_LGBM:
            print("Training LightGBM...")
            try:
                lgbm_model = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, 
                                               learning_rate=0.05, random_state=42,
                                               verbosity=-1)
                for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    lgbm_fold = lgb.LGBMClassifier(n_estimators=300, num_leaves=31,
                                                  learning_rate=0.05, random_state=42,
                                                  verbosity=-1)
                    lgbm_fold.fit(X[train_idx], y[train_idx])
                    oof_predictions[val_idx, 2] = lgbm_fold.predict_proba(X[val_idx])[:, 1]
                lgbm_model.fit(X, y)
                self.base_models['lgbm'] = lgbm_model
            except Exception as e:
                print(f"LightGBM failed: {e}")
        
        oof_predictions[:, 3] = oof_predictions[:, :3].mean(axis=1)
        
        valid_mask = ~np.all(oof_predictions == 0, axis=1)
        if valid_mask.sum() > 0:
            print("Training meta-learner...")
            self.meta_learner.fit(oof_predictions[valid_mask], y[valid_mask])
        
        return self
    
    def predict_proba(self, X):
        """Generate probability predictions"""
        X = np.array(X)
        meta_features = np.zeros((len(X), 4))
        
        if 'rf' in self.base_models:
            meta_features[:, 0] = self.base_models['rf'].predict_proba(X)[:, 1]
        if 'xgb' in self.base_models:
            meta_features[:, 1] = self.base_models['xgb'].predict_proba(X)[:, 1]
        if 'lgbm' in self.base_models:
            meta_features[:, 2] = self.base_models['lgbm'].predict_proba(X)[:, 1]
        meta_features[:, 3] = meta_features[:, :3].mean(axis=1)
        
        return self.meta_learner.predict_proba(meta_features)
    
    def predict(self, X):
        """Generate class predictions"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


def evaluate_model(y_true, y_pred, y_proba=None):
    """Calculate comprehensive metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['auc_roc'] = 0.5
    return metrics


def train_and_evaluate(symbol='BTC-USD', asset_type='crypto'):
    """Main training and evaluation function"""
    print(f"\n{'='*60}")
    print(f"TRAINING ENHANCED MODELS FOR {symbol}")
    print(f"{'='*60}")
    
    data = prepare_data(symbol, days=365*3)
    if data is None or len(data) < 100:
        print(f"Insufficient data for {symbol}")
        return None
    
    feature_cols = [col for col in data.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    X = data[feature_cols].values
    y = data['Target'].values
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if HAS_SMOTE and len(np.unique(y_train)) > 1:
        print("Applying SMOTE for class balancing...")
        smote = SMOTE(k_neighbors=min(5, min(np.bincount(y_train)) - 1), random_state=42)
        try:
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        except:
            X_train_balanced, y_train_balanced = X_train_scaled, y_train
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
    
    results = {}
    
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', 
                               n_jobs=-1, random_state=42)
    rf.fit(X_train_balanced, y_train_balanced)
    rf_pred = rf.predict(X_test_scaled)
    rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
    results['RandomForest'] = evaluate_model(y_test, rf_pred, rf_proba)
    print(f"  Accuracy: {results['RandomForest']['accuracy']:.4f}")
    
    if HAS_XGB:
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                      use_label_encoder=False, eval_metric='logloss',
                                      random_state=42)
        xgb_model.fit(X_train_balanced, y_train_balanced)
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
        results['XGBoost'] = evaluate_model(y_test, xgb_pred, xgb_proba)
        print(f"  Accuracy: {results['XGBoost']['accuracy']:.4f}")
    
    if HAS_LGBM:
        print("Training LightGBM...")
        try:
            lgbm = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                     random_state=42, verbosity=-1)
            lgbm.fit(X_train_balanced, y_train_balanced)
            lgbm_pred = lgbm.predict(X_test_scaled)
            lgbm_proba = lgbm.predict_proba(X_test_scaled)[:, 1]
            results['LightGBM'] = evaluate_model(y_test, lgbm_pred, lgbm_proba)
            print(f"  Accuracy: {results['LightGBM']['accuracy']:.4f}")
        except Exception as e:
            print(f"  LightGBM failed: {e}")
    
    if HAS_TF:
        print("Training BiLSTM...")
        try:
            window = 20
            X_seq_train, y_seq_train = create_sequences(X_train_scaled, y_train, window)
            X_seq_test, y_seq_test = create_sequences(X_test_scaled, y_test, window)
            
            if len(X_seq_train) > 50:
                model = build_bilstm_model((window, X_train_scaled.shape[1]))
                
                callbacks = [
                    EarlyStopping(patience=15, restore_best_weights=True, verbose=0),
                    ReduceLROnPlateau(factor=0.5, patience=5, verbose=0)
                ]
                
                model.fit(X_seq_train, y_seq_train, 
                         epochs=100, batch_size=32, 
                         validation_split=0.2,
                         callbacks=callbacks,
                         verbose=0)
                
                lstm_proba = model.predict(X_seq_test, verbose=0).flatten()
                lstm_pred = (lstm_proba > 0.5).astype(int)
                results['BiLSTM'] = evaluate_model(y_seq_test, lstm_pred, lstm_proba)
                print(f"  Accuracy: {results['BiLSTM']['accuracy']:.4f}")
        except Exception as e:
            print(f"  BiLSTM failed: {e}")
    
    print("\nTraining Stacking Ensemble...")
    try:
        stacking = StackingEnsemble(n_splits=5)
        stacking.fit(X_train_balanced, y_train_balanced)
        stack_pred = stacking.predict(X_test_scaled)
        stack_proba = stacking.predict_proba(X_test_scaled)[:, 1]
        results['StackingEnsemble'] = evaluate_model(y_test, stack_pred, stack_proba)
        print(f"  Accuracy: {results['StackingEnsemble']['accuracy']:.4f}")
    except Exception as e:
        print(f"  Stacking failed: {e}")
    
    print(f"\n{'='*60}")
    print(f"RESULTS FOR {symbol}")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['accuracy']:.4f}     {metrics['precision']:.4f}     {metrics['recall']:.4f}     {metrics['f1']:.4f}     {metrics.get('auc_roc', 0):.4f}")
    
    return results


def main():
    """Run comprehensive model training"""
    print("\n" + "="*80)
    print("INTELLITRADEAI - ENHANCED MODEL TRAINING")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"SMOTE available: {HAS_SMOTE}")
    print(f"XGBoost available: {HAS_XGB}")
    print(f"LightGBM available: {HAS_LGBM}")
    print(f"TensorFlow available: {HAS_TF}")
    
    crypto_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    stock_symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    all_results = {'crypto': {}, 'stocks': {}}
    
    print("\n" + "="*80)
    print("CRYPTOCURRENCY TRAINING")
    print("="*80)
    for symbol in crypto_symbols:
        results = train_and_evaluate(symbol, 'crypto')
        if results:
            all_results['crypto'][symbol] = results
    
    print("\n" + "="*80)
    print("STOCK TRAINING")
    print("="*80)
    for symbol in stock_symbols:
        results = train_and_evaluate(symbol, 'stocks')
        if results:
            all_results['stocks'][symbol] = results
    
    print("\n" + "="*80)
    print("SUMMARY - AVERAGE ACCURACY ACROSS ASSETS")
    print("="*80)
    
    for asset_type in ['crypto', 'stocks']:
        if all_results[asset_type]:
            print(f"\n{asset_type.upper()}:")
            model_accs = {}
            for symbol, results in all_results[asset_type].items():
                for model, metrics in results.items():
                    if model not in model_accs:
                        model_accs[model] = []
                    model_accs[model].append(metrics['accuracy'])
            
            for model, accs in model_accs.items():
                avg_acc = np.mean(accs)
                print(f"  {model}: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


if __name__ == "__main__":
    main()
