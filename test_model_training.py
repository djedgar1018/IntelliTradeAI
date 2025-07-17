#!/usr/bin/env python3
"""
Comprehensive Test Script for Robust Model Training
Tests the end-to-end model training pipeline with real and synthetic data
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('.')

from models.model_trainer import RobustModelTrainer
from data.data_ingestion import DataIngestion
from config import config

def create_synthetic_ohlcv_data(symbol='TEST', days=500):
    """
    Create synthetic OHLCV data for testing
    
    Args:
        symbol: Trading symbol
        days: Number of days of data
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Creating synthetic OHLCV data for {symbol} ({days} days)...")
    
    # Create date range
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate synthetic price data with trends and volatility
    np.random.seed(42)  # For reproducibility
    
    # Base price and trend
    base_price = 100
    trend = np.linspace(0, 20, days)  # Upward trend
    
    # Random walk with volatility
    returns = np.random.normal(0, 0.02, days)  # 2% daily volatility
    price_changes = np.cumsum(returns)
    
    # Create realistic OHLCV data
    close_prices = base_price + trend + price_changes * 10
    
    # Generate OHLC from close prices
    data = []
    for i in range(days):
        close = close_prices[i]
        
        # Add some intraday volatility
        daily_vol = np.random.uniform(0.005, 0.03)  # 0.5% to 3% daily range
        
        high = close * (1 + daily_vol)
        low = close * (1 - daily_vol)
        
        # Open is close to previous close with some gap
        if i == 0:
            open_price = close * (1 + np.random.normal(0, 0.005))
        else:
            open_price = close_prices[i-1] * (1 + np.random.normal(0, 0.01))
        
        # Volume varies with price movements
        volume = np.random.uniform(10000, 100000)
        if abs(returns[i]) > 0.02:  # High volatility = higher volume
            volume *= np.random.uniform(1.5, 3.0)
        
        data.append({
            'open': max(0, open_price),
            'high': max(0, high),
            'low': max(0, low),
            'close': max(0, close),
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    print(f"  → Generated {len(df)} rows of synthetic data")
    print(f"  → Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"  → Average volume: {df['volume'].mean():,.0f}")
    
    return df

def test_feature_engineering():
    """Test the feature engineering pipeline"""
    print("\n" + "="*60)
    print("TESTING FEATURE ENGINEERING")
    print("="*60)
    
    try:
        # Create synthetic data
        data = create_synthetic_ohlcv_data('TEST', 200)
        
        # Initialize trainer
        trainer = RobustModelTrainer()
        
        # Test feature engineering
        print("\nRunning feature engineering...")
        engineered_data = trainer.engineer_features(data, 'TEST')
        
        print(f"✓ Feature engineering completed successfully")
        print(f"  → Original features: {len(data.columns)}")
        print(f"  → Engineered features: {len(engineered_data.columns)}")
        print(f"  → Data shape: {engineered_data.shape}")
        
        # Show some feature examples
        feature_examples = [col for col in engineered_data.columns if col not in ['open', 'high', 'low', 'close', 'volume']][:10]
        print(f"  → Example features: {feature_examples}")
        
        # Check target variable
        if 'target' in engineered_data.columns:
            target_dist = engineered_data['target'].value_counts()
            print(f"  → Target distribution: {target_dist.to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {str(e)}")
        return False

def test_model_training_synthetic():
    """Test model training with synthetic data"""
    print("\n" + "="*60)
    print("TESTING MODEL TRAINING (SYNTHETIC DATA)")
    print("="*60)
    
    try:
        # Create synthetic data
        data = create_synthetic_ohlcv_data('BTC', 300)
        
        # Initialize trainer
        trainer = RobustModelTrainer()
        
        # Test comprehensive training
        print("\nRunning comprehensive model training...")
        results = trainer.run_comprehensive_training(
            data, 
            'BTC', 
            algorithms=['random_forest', 'xgboost'],
            optimize_hyperparams=False  # Faster for testing
        )
        
        print(f"✓ Model training completed successfully")
        print(f"  → Symbol: {results['symbol']}")
        print(f"  → Training data shape: {results['data_shape']['train']}")
        print(f"  → Test data shape: {results['data_shape']['test']}")
        print(f"  → Number of features: {len(results['data_shape']['features'])}")
        
        # Show model results
        for algorithm, result in results['models'].items():
            if result['status'] == 'success':
                metrics = result['metrics']
                print(f"  → {algorithm.upper()}: Accuracy={metrics['accuracy']:.4f}, F1={metrics.get('f1_score', 0):.4f}")
            else:
                print(f"  → {algorithm.upper()}: FAILED - {result.get('error', 'Unknown error')}")
        
        # Show best model
        best_model = results['best_model']
        print(f"  → Best model: {best_model['algorithm']} (Score: {best_model['score']:.4f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training test failed: {str(e)}")
        return False

def test_model_training_real_data():
    """Test model training with real data (requires API keys)"""
    print("\n" + "="*60)
    print("TESTING MODEL TRAINING (REAL DATA)")
    print("="*60)
    
    try:
        # Initialize data ingestion
        data_ingestion = DataIngestion()
        
        # Try to fetch real data (Yahoo Finance - no API key required)
        print("Attempting to fetch real stock data...")
        stock_data = data_ingestion.fetch_stock_data(['AAPL'], period='1y')
        
        if stock_data and 'AAPL' in stock_data:
            data = stock_data['AAPL']
            print(f"✓ Real data fetched successfully: {len(data)} rows")
            
            # Initialize trainer
            trainer = RobustModelTrainer()
            
            # Test comprehensive training
            print("\nRunning comprehensive training on real data...")
            results = trainer.run_comprehensive_training(
                data, 
                'AAPL', 
                algorithms=['random_forest'],  # Just one algorithm for speed
                optimize_hyperparams=False
            )
            
            print(f"✓ Real data training completed successfully")
            print(f"  → Symbol: {results['symbol']}")
            print(f"  → Training data shape: {results['data_shape']['train']}")
            
            # Show model results
            for algorithm, result in results['models'].items():
                if result['status'] == 'success':
                    metrics = result['metrics']
                    print(f"  → {algorithm.upper()}: Accuracy={metrics['accuracy']:.4f}")
                    
                    # Show feature importance
                    if 'feature_importance' in metrics and metrics['feature_importance']:
                        print(f"  → Top 5 features:")
                        for i, feat in enumerate(metrics['feature_importance'][:5]):
                            print(f"    {i+1}. {feat['feature']}: {feat['importance']:.4f}")
                else:
                    print(f"  → {algorithm.upper()}: FAILED - {result.get('error', 'Unknown error')}")
            
            return True
            
        else:
            print("⚠ Real data not available, skipping real data test")
            return True
            
    except Exception as e:
        print(f"❌ Real data training test failed: {str(e)}")
        return False

def test_model_persistence():
    """Test model saving and loading"""
    print("\n" + "="*60)
    print("TESTING MODEL PERSISTENCE")
    print("="*60)
    
    try:
        # Create and train a model
        data = create_synthetic_ohlcv_data('PERSIST_TEST', 150)
        trainer = RobustModelTrainer()
        
        print("Training model for persistence test...")
        results = trainer.run_comprehensive_training(
            data, 
            'PERSIST_TEST', 
            algorithms=['random_forest'],
            optimize_hyperparams=False
        )
        
        if results['models']['random_forest']['status'] == 'success':
            model_key = results['models']['random_forest']['model_key']
            print(f"✓ Model trained and saved: {model_key}")
            
            # Test loading
            print("Testing model loading...")
            loaded_model = trainer.load_model(model_key)
            
            if loaded_model:
                print(f"✓ Model loaded successfully")
                print(f"  → Algorithm: {type(loaded_model['model']).__name__}")
                print(f"  → Features: {len(loaded_model['feature_names'])}")
                print(f"  → Has scaler: {loaded_model.get('scaler') is not None}")
                
                # Test prediction
                print("Testing prediction with loaded model...")
                sample_data = data.tail(10)
                engineered_sample = trainer.engineer_features(sample_data, 'PERSIST_TEST')
                
                try:
                    predictions, probabilities = trainer.predict_with_model(model_key, engineered_sample)
                    print(f"✓ Prediction successful: {len(predictions)} predictions")
                    print(f"  → Predictions: {predictions[:5]}")
                    
                    return True
                    
                except Exception as e:
                    print(f"❌ Prediction failed: {str(e)}")
                    return False
                    
            else:
                print("❌ Model loading failed")
                return False
        else:
            print("❌ Model training failed")
            return False
            
    except Exception as e:
        print(f"❌ Model persistence test failed: {str(e)}")
        return False

def test_model_summary():
    """Test model summary functionality"""
    print("\n" + "="*60)
    print("TESTING MODEL SUMMARY")
    print("="*60)
    
    try:
        # Create trainer and train a few models
        trainer = RobustModelTrainer()
        
        # Train models with different symbols
        for symbol in ['SUMMARY_TEST1', 'SUMMARY_TEST2']:
            data = create_synthetic_ohlcv_data(symbol, 100)
            results = trainer.run_comprehensive_training(
                data, 
                symbol, 
                algorithms=['random_forest'],
                optimize_hyperparams=False
            )
        
        # Get model summary
        print("Getting model summary...")
        summary = trainer.get_model_summary()
        
        print(f"✓ Model summary generated successfully")
        print(f"  → Total models: {summary['total_models']}")
        
        for model_key, info in summary['models'].items():
            print(f"  → {model_key}:")
            print(f"    Features: {info['features']}")
            print(f"    Algorithm: {info['algorithm']}")
            print(f"    Has scaler: {info['has_scaler']}")
            print(f"    Has selector: {info['has_selector']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model summary test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("ROBUST MODEL TRAINER TEST SUITE")
    print("="*60)
    print("Testing comprehensive machine learning pipeline")
    print("Including feature engineering, model training, and persistence")
    print("="*60)
    
    # Run all tests
    tests = [
        ("Feature Engineering", test_feature_engineering),
        ("Model Training (Synthetic)", test_model_training_synthetic),
        ("Model Training (Real Data)", test_model_training_real_data),
        ("Model Persistence", test_model_persistence),
        ("Model Summary", test_model_summary),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"RUNNING: {test_name}")
            print(f"{'='*60}")
            
            result = test_func()
            results[test_name] = result
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: CRASHED - {str(e)}")
            results[test_name] = False
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The robust model trainer is working correctly.")
        return 0
    else:
        print("⚠ Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())