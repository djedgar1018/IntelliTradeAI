"""
Random Forest Model for Trading Signal Generation
Implements Random Forest classifier for trading decisions
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import joblib
import os
from config import config

class RandomForestPredictor:
    """Random Forest Model for trading prediction"""
    
    def __init__(self, model_type='classifier'):
        """
        Initialize Random Forest model
        
        Args:
            model_type: 'classifier' or 'regressor'
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.is_trained = False
        
        if model_type == 'classifier':
            self.model = RandomForestClassifier(**config.RANDOM_FOREST_CONFIG)
        else:
            self.model = RandomForestRegressor(**config.RANDOM_FOREST_CONFIG)
    
    def create_features(self, data):
        """
        Create features for Random Forest model
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            features: DataFrame with engineered features
        """
        features = pd.DataFrame()
        
        # Price features
        features['price_change'] = data['close'].pct_change()
        features['high_low_ratio'] = data['high'] / data['low']
        features['volume_change'] = data['volume'].pct_change()
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'ma_{period}'] = data['close'].rolling(window=period).mean()
            features[f'price_ma_{period}_ratio'] = data['close'] / features[f'ma_{period}']
        
        # Volatility features
        features['volatility_5'] = data['close'].rolling(window=5).std()
        features['volatility_20'] = data['close'].rolling(window=20).std()
        
        # Price position features
        features['price_position_5'] = (data['close'] - data['close'].rolling(5).min()) / \
                                     (data['close'].rolling(5).max() - data['close'].rolling(5).min())
        features['price_position_20'] = (data['close'] - data['close'].rolling(20).min()) / \
                                      (data['close'].rolling(20).max() - data['close'].rolling(20).min())
        
        # Momentum features
        features['momentum_5'] = data['close'] / data['close'].shift(5)
        features['momentum_10'] = data['close'] / data['close'].shift(10)
        
        # Volume features
        features['volume_ma_5'] = data['volume'].rolling(window=5).mean()
        features['volume_ma_20'] = data['volume'].rolling(window=20).mean()
        features['volume_ratio'] = data['volume'] / features['volume_ma_20']
        
        # Technical indicators (if available)
        if 'rsi' in data.columns:
            features['rsi'] = data['rsi']
        if 'macd' in data.columns:
            features['macd'] = data['macd']
        if 'macd_signal' in data.columns:
            features['macd_signal'] = data['macd_signal']
        if 'bb_upper' in data.columns:
            features['bb_position'] = (data['close'] - data['bb_lower']) / \
                                     (data['bb_upper'] - data['bb_lower'])
        
        return features
    
    def create_target(self, data, target_type='classification', lookahead=1):
        """
        Create target variable
        
        Args:
            data: DataFrame with price data
            target_type: 'classification' or 'regression'
            lookahead: Number of periods to look ahead
            
        Returns:
            target: Target variable
        """
        if target_type == 'classification':
            # Create classification target (0: sell, 1: hold, 2: buy)
            future_return = data['close'].shift(-lookahead) / data['close'] - 1
            
            target = np.where(future_return > 0.02, 2,  # Buy
                            np.where(future_return < -0.02, 0, 1))  # Sell, Hold
            
        else:
            # Create regression target (future price)
            target = data['close'].shift(-lookahead)
        
        return target
    
    def train(self, data, target_type='classification', lookahead=1, test_size=0.2):
        """
        Train Random Forest model
        
        Args:
            data: DataFrame with OHLCV data
            target_type: 'classification' or 'regression'
            lookahead: Number of periods to look ahead
            test_size: Fraction of data for testing
            
        Returns:
            metrics: Training metrics
        """
        try:
            # Create features and target
            features = self.create_features(data)
            target = self.create_target(data, target_type, lookahead)
            
            # Remove NaN values
            valid_mask = ~(features.isna().any(axis=1) | pd.isna(target))
            features = features[valid_mask]
            target = target[valid_mask]
            
            if len(features) < 100:
                raise ValueError("Not enough valid data for training")
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, target, test_size=test_size, 
                random_state=config.RANDOM_FOREST_CONFIG["random_state"]
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            if target_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                metrics = {
                    'accuracy': accuracy,
                    'classification_report': report,
                    'feature_importance': dict(zip(features.columns, self.model.feature_importances_))
                }
            else:
                mse = mean_squared_error(y_test, y_pred)
                metrics = {
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'feature_importance': dict(zip(features.columns, self.model.feature_importances_))
                }
            
            self.is_trained = True
            self.feature_names = features.columns.tolist()
            
            return metrics
            
        except Exception as e:
            raise Exception(f"Error training Random Forest model: {str(e)}")
    
    def predict(self, data):
        """
        Make predictions using trained model
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            predictions: Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Create features
            features = self.create_features(data)
            
            # Handle missing features
            for col in self.feature_names:
                if col not in features.columns:
                    features[col] = 0
            
            # Reorder columns to match training
            features = features[self.feature_names]
            
            # Remove NaN values
            valid_mask = ~features.isna().any(axis=1)
            features_clean = features[valid_mask]
            
            if len(features_clean) == 0:
                raise ValueError("No valid data for prediction")
            
            # Scale features
            features_scaled = self.scaler.transform(features_clean)
            
            # Make predictions
            predictions = self.model.predict(features_scaled)
            
            # Create full predictions array with NaN for invalid rows
            full_predictions = np.full(len(features), np.nan)
            full_predictions[valid_mask] = predictions
            
            return full_predictions
            
        except Exception as e:
            raise Exception(f"Error making predictions: {str(e)}")
    
    def predict_proba(self, data):
        """
        Get prediction probabilities (for classification only)
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            probabilities: Array of prediction probabilities
        """
        if not self.is_trained or self.model_type != 'classifier':
            raise ValueError("Model must be trained classifier to get probabilities")
        
        try:
            # Create features
            features = self.create_features(data)
            
            # Handle missing features
            for col in self.feature_names:
                if col not in features.columns:
                    features[col] = 0
            
            # Reorder columns to match training
            features = features[self.feature_names]
            
            # Remove NaN values
            valid_mask = ~features.isna().any(axis=1)
            features_clean = features[valid_mask]
            
            if len(features_clean) == 0:
                raise ValueError("No valid data for prediction")
            
            # Scale features
            features_scaled = self.scaler.transform(features_clean)
            
            # Get probabilities
            probabilities = self.model.predict_proba(features_scaled)
            
            return probabilities
            
        except Exception as e:
            raise Exception(f"Error getting probabilities: {str(e)}")
    
    def get_feature_importance(self):
        """
        Get feature importance scores
        
        Returns:
            importance: Dictionary of feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def save_model(self, filepath):
        """Save trained model and scaler"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            joblib.dump(self.model, f"{filepath}_model.pkl")
            
            # Save scaler
            joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
            
            # Save feature names
            joblib.dump(self.feature_names, f"{filepath}_features.pkl")
            
            return True
            
        except Exception as e:
            raise Exception(f"Error saving model: {str(e)}")
    
    def load_model(self, filepath):
        """Load trained model and scaler"""
        try:
            # Load model
            self.model = joblib.load(f"{filepath}_model.pkl")
            
            # Load scaler
            self.scaler = joblib.load(f"{filepath}_scaler.pkl")
            
            # Load feature names
            self.feature_names = joblib.load(f"{filepath}_features.pkl")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def get_signal(self, data):
        """
        Get trading signal based on predictions
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            signal: Trading signal (buy/sell/hold)
            confidence: Confidence score
        """
        if not self.is_trained:
            return 'hold', 0.0
        
        try:
            if self.model_type == 'classifier':
                # Get probabilities
                probabilities = self.predict_proba(data)
                
                if len(probabilities) == 0:
                    return 'hold', 0.0
                
                # Get last prediction
                last_proba = probabilities[-1]
                prediction = np.argmax(last_proba)
                confidence = np.max(last_proba)
                
                # Convert to signal
                if prediction == 2:  # Buy
                    signal = 'buy'
                elif prediction == 0:  # Sell
                    signal = 'sell'
                else:  # Hold
                    signal = 'hold'
                
            else:
                # For regression, compare predicted vs current price
                predictions = self.predict(data)
                
                if len(predictions) == 0 or np.isnan(predictions[-1]):
                    return 'hold', 0.0
                
                predicted_price = predictions[-1]
                current_price = data['close'].iloc[-1]
                
                price_change = (predicted_price - current_price) / current_price
                
                if price_change > 0.02:
                    signal = 'buy'
                    confidence = min(abs(price_change) * 10, 1.0)
                elif price_change < -0.02:
                    signal = 'sell'
                    confidence = min(abs(price_change) * 10, 1.0)
                else:
                    signal = 'hold'
                    confidence = 0.5
            
            return signal, confidence
            
        except Exception as e:
            return 'hold', 0.0
