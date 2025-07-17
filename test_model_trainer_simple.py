#!/usr/bin/env python3
"""
Simple Test Script for Robust Model Trainer
Tests core functionality with synthetic data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from models.model_trainer import RobustModelTrainer

def create_test_data(days=200):
    """Create synthetic OHLCV data for testing"""
    print(f"Creating test data with {days} days...")
    
    # Generate dates
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 100
    
    # Random walk with trend
    returns = np.random.normal(0.001, 0.02, days)  # Small upward trend with 2% volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Add intraday volatility
        daily_range = price * np.random.uniform(0.01, 0.05)  # 1-5% daily range
        
        high = price + daily_range * np.random.uniform(0.3, 0.7)
        low = price - daily_range * np.random.uniform(0.3, 0.7)
        
        # Open close to previous close
        if i == 0:
            open_price = price
        else:
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.005))
        
        close = price
        volume = np.random.uniform(50000, 200000)
        
        data.append({
            'open': max(0.01, open_price),
            'high': max(0.01, high),
            'low': max(0.01, low),
            'close': max(0.01, close),
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    print(f"✓ Test data created: {len(df)} rows")
    return df

def test_feature_engineering():
    """Test feature engineering"""
    print("\n" + "="*50)
    print("TESTING FEATURE ENGINEERING")
    print("="*50)
    
    try:
        # Create test data
        data = create_test_data(100)
        
        # Initialize trainer
        trainer = RobustModelTrainer()
        
        # Test feature engineering
        engineered_data = trainer.engineer_features(data, 'TEST')
        
        print(f"✓ Feature engineering successful")
        print(f"  Original columns: {len(data.columns)}")
        print(f"  Engineered columns: {len(engineered_data.columns)}")
        print(f"  Final rows: {len(engineered_data)}")
        
        # Check target variable
        if 'target' in engineered_data.columns:
            target_counts = engineered_data['target'].value_counts()
            print(f"  Target distribution: {target_counts.to_dict()}")
        
        return True, engineered_data
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return False, None

def test_model_training():
    """Test model training"""
    print("\n" + "="*50)
    print("TESTING MODEL TRAINING")
    print("="*50)
    
    try:
        # Get feature engineered data
        success, engineered_data = test_feature_engineering()
        
        if not success:
            print("❌ Skipping model training due to feature engineering failure")
            return False
        
        # Initialize trainer
        trainer = RobustModelTrainer()
        
        # Prepare training data
        print("Preparing training data...")
        X_train, X_test, y_train, y_test = trainer.prepare_training_data(
            engineered_data, 
            target_column='target',
            scale_features=True,
            select_features=True,
            feature_selection_k=20
        )
        
        print(f"✓ Data preparation successful")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {len(X_train.columns)}")
        
        # Train Random Forest model
        print("Training Random Forest model...")
        model, metrics = trainer.train_model_with_optimization(
            X_train, X_test, y_train, y_test,
            algorithm='random_forest',
            optimize_hyperparams=False
        )
        
        print(f"✓ Random Forest training successful")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall: {metrics.get('recall', 0):.4f}")
        
        # Show top features if available
        if metrics.get('feature_importance'):
            print("  Top 5 features:")
            for i, feat in enumerate(metrics['feature_importance'][:5]):
                print(f"    {i+1}. {feat['feature']}: {feat['importance']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def test_comprehensive_training():
    """Test comprehensive training pipeline"""
    print("\n" + "="*50)
    print("TESTING COMPREHENSIVE TRAINING")
    print("="*50)
    
    try:
        # Create test data
        data = create_test_data(200)
        
        # Initialize trainer
        trainer = RobustModelTrainer()
        
        # Run comprehensive training
        print("Running comprehensive training pipeline...")
        results = trainer.run_comprehensive_training(
            data,
            'BTC',
            algorithms=['random_forest'],
            optimize_hyperparams=False
        )
        
        print(f"✓ Comprehensive training successful")
        print(f"  Symbol: {results['symbol']}")
        print(f"  Training shape: {results['data_shape']['train']}")
        print(f"  Test shape: {results['data_shape']['test']}")
        
        # Show model results
        for algorithm, result in results['models'].items():
            if result['status'] == 'success':
                metrics = result['metrics']
                print(f"  {algorithm.upper()}:")
                print(f"    Accuracy: {metrics['accuracy']:.4f}")
                print(f"    AUC: {metrics.get('test_auc', 0):.4f}")
                print(f"    Cross-val mean: {metrics.get('cross_val_score', {}).get('mean', 0):.4f}")
            else:
                print(f"  {algorithm.upper()}: FAILED")
        
        # Show best model
        best_model = results['best_model']
        print(f"  Best model: {best_model['algorithm']} (Score: {best_model['score']:.4f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive training failed: {e}")
        return False

def main():
    """Main test function"""
    print("ROBUST MODEL TRAINER - SIMPLE TEST")
    print("="*50)
    
    tests = [
        ("Feature Engineering", test_feature_engineering),
        ("Model Training", test_model_training),
        ("Comprehensive Training", test_comprehensive_training)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"RUNNING: {test_name}")
        print(f"{'='*50}")
        
        try:
            if test_name == "Feature Engineering":
                result, _ = test_func()
            else:
                result = test_func()
            results.append(result)
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: CRASHED - {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Model trainer is working correctly.")
        return 0
    else:
        print("⚠ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())