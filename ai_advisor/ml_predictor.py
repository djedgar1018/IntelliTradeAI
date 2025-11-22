"""
Real ML Model Predictor for Trading Signals
Uses the trained Random Forest models to generate BUY/SELL/HOLD signals
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Import feature engineering from model trainer for compatibility
from models.model_trainer import RobustModelTrainer

class MLPredictor:
    """Generate trading signals using trained ML models"""
    
    def __init__(self):
        self.model_cache_dir = 'models/cache'
        self.loaded_models = {}
    
    def _load_model(self, symbol):
        """Load a trained model for a symbol"""
        if symbol in self.loaded_models:
            return self.loaded_models[symbol]
        
        model_path = os.path.join(self.model_cache_dir, f'{symbol}_random_forest.joblib')
        
        if not os.path.exists(model_path):
            return None
        
        try:
            model_data = joblib.load(model_path)
            
            # Compatibility fix: convert feature_names to feature_columns if needed
            if isinstance(model_data, dict) and 'feature_names' in model_data and 'feature_columns' not in model_data:
                model_data['feature_columns'] = model_data['feature_names']
            
            self.loaded_models[symbol] = model_data
            return model_data
        except Exception as e:
            print(f"Error loading model for {symbol}: {e}")
            return None
    
    def _calculate_features(self, df, symbol):
        """
        Calculate technical features from OHLCV data
        Uses RobustModelTrainer's feature engineering for compatibility with trained models
        """
        # Use the same feature engineering as training for compatibility
        trainer = RobustModelTrainer()
        return trainer.engineer_features(df, symbol)
    
    def predict(self, symbol, data):
        """
        Generate trading signal for a symbol
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'XRP', 'BTC')
            data: DataFrame with OHLCV data
        
        Returns:
            Dictionary with prediction and analysis
        """
        # Load model
        model_data = self._load_model(symbol)
        
        if model_data is None:
            # No trained model - return neutral
            return {
                'symbol': symbol,
                'signal': 'HOLD',
                'confidence': 0.0,
                'probability_up': 0.5,
                'probability_down': 0.5,
                'current_price': float(data['close'].iloc[-1]),
                'price_change_24h': float(data['close'].pct_change().iloc[-1] * 100),
                'recommendation': {
                    'decision': 'HOLD',
                    'confidence_level': 'Low',
                    'risk_level': 'Unknown',
                    'action_explanation': f'No trained model available for {symbol}. Cannot generate prediction.'
                },
                'technical_indicators': {},
                'error': 'No model found'
            }
        
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        
        # Calculate features
        df_features = self._calculate_features(data, symbol)
        df_features = df_features.dropna()
        
        if len(df_features) == 0:
            return {
                'symbol': symbol,
                'signal': 'HOLD',
                'confidence': 0.0,
                'error': 'Insufficient data for prediction'
            }
        
        # Get latest features
        latest_features = df_features[feature_columns].iloc[-1:]
        
        # Make prediction
        prediction = model.predict(latest_features)[0]
        probabilities = model.predict_proba(latest_features)[0]
        
        # Interpret prediction
        probability_up = probabilities[1]  # Probability of price going UP
        probability_down = probabilities[0]  # Probability of price going DOWN
        
        # Generate signal based on confidence
        if prediction == 1 and probability_up >= 0.6:
            signal = 'BUY'
            confidence = probability_up
        elif prediction == 0 and probability_down >= 0.6:
            signal = 'SELL'
            confidence = probability_down
        else:
            signal = 'HOLD'
            confidence = max(probability_up, probability_down)
        
        # Get current price and change
        current_price = float(data['close'].iloc[-1])
        price_change_24h = float(data['close'].pct_change().iloc[-1] * 100)
        
        # Get technical indicators (using features that exist in engineered data)
        latest_row = df_features.iloc[-1]
        technical_indicators = {}
        
        # Extract available indicators safely
        for indicator in ['rsi', 'macd', 'macd_signal', 'sma_20', 'ema_12', 'ema_26', 'volume_ratio', 'volatility_20']:
            if indicator in latest_row:
                technical_indicators[indicator] = float(latest_row[indicator])
        
        # Add volatility (use volatility_20 if available, otherwise default)
        if 'volatility_20' not in technical_indicators:
            technical_indicators['volatility_20'] = 0.03
        
        # Determine confidence level and risk
        if confidence >= 0.75:
            confidence_level = 'High'
        elif confidence >= 0.60:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'
        
        # Risk level based on volatility
        volatility = technical_indicators.get('volatility_20', 0.03)
        if volatility > 0.05:
            risk_level = 'High'
        elif volatility > 0.03:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        # Generate explanation
        explanation = self._generate_explanation(
            signal, technical_indicators, confidence, symbol
        )
        
        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': float(confidence),
            'probability_up': float(probability_up),
            'probability_down': float(probability_down),
            'current_price': current_price,
            'price_change_24h': price_change_24h,
            'recommendation': {
                'decision': signal,
                'confidence_level': confidence_level,
                'risk_level': risk_level,
                'action_explanation': explanation
            },
            'technical_indicators': technical_indicators,
            'model_metrics': model_data.get('metrics', {})
        }
    
    def _generate_explanation(self, signal, indicators, confidence, symbol):
        """Generate human-readable explanation for the signal"""
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        sma_20 = indicators.get('sma_20', 0)
        ema_12 = indicators.get('ema_12', 0)
        ema_26 = indicators.get('ema_26', 0)
        
        explanations = []
        
        if signal == 'BUY':
            explanations.append(f"AI model predicts upward price movement for {symbol}")
            
            if rsi < 30:
                explanations.append("RSI indicates oversold conditions (potential bounce)")
            elif rsi < 50:
                explanations.append("RSI shows room for upward movement")
            
            if macd > 0:
                explanations.append("MACD shows bullish momentum")
            
            if ema_12 > ema_26:
                explanations.append("Short-term trend is above long-term (bullish)")
            
            explanations.append(f"Model confidence: {confidence:.1%}")
            
        elif signal == 'SELL':
            explanations.append(f"AI model predicts downward price movement for {symbol}")
            
            if rsi > 70:
                explanations.append("RSI indicates overbought conditions (potential drop)")
            elif rsi > 50:
                explanations.append("RSI suggests limited upside potential")
            
            if macd < 0:
                explanations.append("MACD shows bearish momentum")
            
            if ema_12 < ema_26:
                explanations.append("Short-term trend is below long-term (bearish)")
            
            explanations.append(f"Model confidence: {confidence:.1%}")
            
        else:  # HOLD
            explanations.append(f"AI model suggests waiting for {symbol}")
            explanations.append("Unclear signals or low confidence in direction")
            explanations.append(f"Probability UP: {confidence:.1%}, Probability DOWN: {1-confidence:.1%}")
        
        return ". ".join(explanations) + "."
    
    def analyze_asset(self, symbol, data):
        """
        Compatibility method for the dashboard
        Mimics the TradingIntelligence interface
        """
        return self.predict(symbol, data)
