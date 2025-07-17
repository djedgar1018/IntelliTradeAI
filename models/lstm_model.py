"""
LSTM Model for Time Series Prediction
Implements LSTM neural network for trading signal generation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from config import config

# Try to import TensorFlow, use fallback if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class LSTMPredictor:
    """LSTM Model for time series prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = config.LSTM_CONFIG["sequence_length"]
        self.is_trained = False
        
    def prepare_data(self, data, target_column='close'):
        """
        Prepare data for LSTM training
        
        Args:
            data: DataFrame with time series data
            target_column: Column to predict
            
        Returns:
            X_train, y_train: Training data
        """
        if len(data) < self.sequence_length + 1:
            raise ValueError(f"Not enough data. Need at least {self.sequence_length + 1} samples")
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data[[target_column]])
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input data
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model but not available")
            
        model = Sequential([
            LSTM(config.LSTM_CONFIG["units"], 
                 return_sequences=True, 
                 input_shape=input_shape),
            Dropout(config.LSTM_CONFIG["dropout_rate"]),
            
            LSTM(config.LSTM_CONFIG["units"], 
                 return_sequences=True),
            Dropout(config.LSTM_CONFIG["dropout_rate"]),
            
            LSTM(config.LSTM_CONFIG["units"]),
            Dropout(config.LSTM_CONFIG["dropout_rate"]),
            
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=config.LSTM_CONFIG["learning_rate"]),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, data, target_column='close', validation_split=0.2):
        """
        Train the LSTM model
        
        Args:
            data: DataFrame with time series data
            target_column: Column to predict
            validation_split: Fraction of data for validation
            
        Returns:
            training_history: Training history
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model but not available")
            
        try:
            # Prepare data
            X, y = self.prepare_data(data, target_column)
            
            # Reshape for LSTM
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Build model
            self.build_model((X.shape[1], 1))
            
            # Train model
            history = self.model.fit(
                X, y,
                batch_size=config.LSTM_CONFIG["batch_size"],
                epochs=config.LSTM_CONFIG["epochs"],
                validation_split=validation_split,
                verbose=1
            )
            
            self.is_trained = True
            return history
            
        except Exception as e:
            raise Exception(f"Error training LSTM model: {str(e)}")
    
    def predict(self, data, target_column='close'):
        """
        Make predictions using trained model
        
        Args:
            data: DataFrame with time series data
            target_column: Column to predict
            
        Returns:
            predictions: Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prepare data
            scaled_data = self.scaler.transform(data[[target_column]])
            
            # Create sequences
            X = []
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i, 0])
            
            if len(X) == 0:
                raise ValueError("Not enough data for prediction")
            
            X = np.array(X)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Inverse transform predictions
            predictions = self.scaler.inverse_transform(predictions)
            
            return predictions.flatten()
            
        except Exception as e:
            raise Exception(f"Error making predictions: {str(e)}")
    
    def predict_next(self, data, target_column='close', steps=1):
        """
        Predict next N steps
        
        Args:
            data: DataFrame with time series data
            target_column: Column to predict
            steps: Number of steps to predict
            
        Returns:
            predictions: Array of future predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Get last sequence
            scaled_data = self.scaler.transform(data[[target_column]])
            last_sequence = scaled_data[-self.sequence_length:]
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(steps):
                # Reshape for prediction
                X = current_sequence.reshape((1, self.sequence_length, 1))
                
                # Predict next value
                next_pred = self.model.predict(X, verbose=0)
                predictions.append(next_pred[0, 0])
                
                # Update sequence
                current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
                current_sequence = current_sequence.reshape(-1, 1)
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            
            return predictions.flatten()
            
        except Exception as e:
            raise Exception(f"Error predicting next steps: {str(e)}")
    
    def evaluate(self, data, target_column='close'):
        """
        Evaluate model performance
        
        Args:
            data: DataFrame with time series data
            target_column: Column to predict
            
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            # Make predictions
            predictions = self.predict(data, target_column)
            
            # Get actual values
            actual = data[target_column].values[self.sequence_length:]
            
            # Calculate metrics
            mse = mean_squared_error(actual, predictions)
            mae = mean_absolute_error(actual, predictions)
            rmse = np.sqrt(mse)
            
            # Calculate accuracy (directional)
            actual_direction = np.diff(actual) > 0
            pred_direction = np.diff(predictions) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction)
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'directional_accuracy': directional_accuracy,
                'sample_size': len(actual)
            }
            
        except Exception as e:
            raise Exception(f"Error evaluating model: {str(e)}")
    
    def save_model(self, filepath):
        """Save trained model and scaler"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            self.model.save(f"{filepath}_model.h5")
            
            # Save scaler
            joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
            
            return True
            
        except Exception as e:
            raise Exception(f"Error saving model: {str(e)}")
    
    def load_model(self, filepath):
        """Load trained model and scaler"""
        try:
            # Load model
            self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
            
            # Load scaler
            self.scaler = joblib.load(f"{filepath}_scaler.pkl")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def get_signal(self, data, target_column='close'):
        """
        Get trading signal based on predictions
        
        Args:
            data: DataFrame with time series data
            target_column: Column to predict
            
        Returns:
            signal: Trading signal (buy/sell/hold)
            confidence: Confidence score
        """
        if not self.is_trained:
            return 'hold', 0.0
        
        try:
            # Predict next price
            next_price = self.predict_next(data, target_column, steps=1)[0]
            current_price = data[target_column].iloc[-1]
            
            # Calculate price change
            price_change = (next_price - current_price) / current_price
            
            # Generate signal
            if price_change > config.TRADING_CONFIG["buy_threshold"] * 0.01:
                signal = 'buy'
                confidence = min(abs(price_change) * 100, 1.0)
            elif price_change < -config.TRADING_CONFIG["sell_threshold"] * 0.01:
                signal = 'sell'
                confidence = min(abs(price_change) * 100, 1.0)
            else:
                signal = 'hold'
                confidence = 0.5
            
            return signal, confidence
            
        except Exception as e:
            return 'hold', 0.0
