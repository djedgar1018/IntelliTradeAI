"""
Technical Indicators Module
Implements RSI, MACD, Bollinger Bands, EMA calculations
"""

import pandas as pd
import numpy as np
from config import config

class TechnicalIndicators:
    """Class for calculating technical indicators"""
    
    @staticmethod
    def rsi(data, period=None):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            data: Series of closing prices
            period: RSI period (default from config)
            
        Returns:
            rsi: RSI values
        """
        if period is None:
            period = config.INDICATOR_CONFIG["rsi_period"]
        
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data, fast=None, slow=None, signal=None):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Series of closing prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            macd, signal_line, histogram: MACD components
        """
        if fast is None:
            fast = config.INDICATOR_CONFIG["macd_fast"]
        if slow is None:
            slow = config.INDICATOR_CONFIG["macd_slow"]
        if signal is None:
            signal = config.INDICATOR_CONFIG["macd_signal"]
        
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data, period=None, std_dev=None):
        """
        Calculate Bollinger Bands
        
        Args:
            data: Series of closing prices
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            upper_band, middle_band, lower_band: Bollinger bands
        """
        if period is None:
            period = config.INDICATOR_CONFIG["bollinger_period"]
        if std_dev is None:
            std_dev = config.INDICATOR_CONFIG["bollinger_std"]
        
        middle_band = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def ema(data, period):
        """
        Calculate Exponential Moving Average
        
        Args:
            data: Series of closing prices
            period: EMA period
            
        Returns:
            ema: EMA values
        """
        return data.ewm(span=period).mean()
    
    @staticmethod
    def sma(data, period):
        """
        Calculate Simple Moving Average
        
        Args:
            data: Series of closing prices
            period: SMA period
            
        Returns:
            sma: SMA values
        """
        return data.rolling(window=period).mean()
    
    @staticmethod
    def stochastic_oscillator(high, low, close, k_period=14, d_period=3):
        """
        Calculate Stochastic Oscillator
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            k_period: %K period
            d_period: %D period
            
        Returns:
            k_percent, d_percent: Stochastic oscillator values
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high, low, close, period=14):
        """
        Calculate Williams %R
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: Lookback period
            
        Returns:
            williams_r: Williams %R values
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    @staticmethod
    def atr(high, low, close, period=14):
        """
        Calculate Average True Range (ATR)
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: ATR period
            
        Returns:
            atr: ATR values
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_all_indicators(data):
        """
        Calculate all technical indicators for a dataset
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            data: DataFrame with added indicator columns
        """
        try:
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # RSI
            data['rsi'] = TechnicalIndicators.rsi(data['close'])
            
            # MACD
            macd, signal, histogram = TechnicalIndicators.macd(data['close'])
            data['macd'] = macd
            data['macd_signal'] = signal
            data['macd_histogram'] = histogram
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(data['close'])
            data['bb_upper'] = bb_upper
            data['bb_middle'] = bb_middle
            data['bb_lower'] = bb_lower
            
            # EMAs
            for period in config.INDICATOR_CONFIG["ema_periods"]:
                data[f'ema_{period}'] = TechnicalIndicators.ema(data['close'], period)
            
            # SMAs
            for period in [5, 10, 20, 50, 200]:
                data[f'sma_{period}'] = TechnicalIndicators.sma(data['close'], period)
            
            # Stochastic Oscillator
            stoch_k, stoch_d = TechnicalIndicators.stochastic_oscillator(
                data['high'], data['low'], data['close']
            )
            data['stoch_k'] = stoch_k
            data['stoch_d'] = stoch_d
            
            # Williams %R
            data['williams_r'] = TechnicalIndicators.williams_r(
                data['high'], data['low'], data['close']
            )
            
            # ATR
            data['atr'] = TechnicalIndicators.atr(
                data['high'], data['low'], data['close']
            )
            
            # Volume indicators
            data['volume_sma_20'] = TechnicalIndicators.sma(data['volume'], 20)
            data['volume_ratio'] = data['volume'] / data['volume_sma_20']
            
            # Price momentum
            data['price_change'] = data['close'].pct_change()
            data['price_change_5'] = data['close'].pct_change(periods=5)
            data['price_change_10'] = data['close'].pct_change(periods=10)
            
            # Volatility
            data['volatility_10'] = data['close'].rolling(window=10).std()
            data['volatility_20'] = data['close'].rolling(window=20).std()
            
            return data
            
        except Exception as e:
            raise Exception(f"Error calculating indicators: {str(e)}")
    
    @staticmethod
    def get_signal_from_indicators(data):
        """
        Generate trading signals based on technical indicators
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            signal: Trading signal (buy/sell/hold)
            confidence: Confidence score
        """
        try:
            if len(data) < 2:
                return 'hold', 0.0
            
            latest = data.iloc[-1]
            previous = data.iloc[-2]
            
            signals = []
            weights = []
            
            # RSI signals
            if not pd.isna(latest['rsi']):
                if latest['rsi'] < 30:
                    signals.append('buy')
                    weights.append(0.8)
                elif latest['rsi'] > 70:
                    signals.append('sell')
                    weights.append(0.8)
                else:
                    signals.append('hold')
                    weights.append(0.2)
            
            # MACD signals
            if not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']):
                if latest['macd'] > latest['macd_signal'] and previous['macd'] <= previous['macd_signal']:
                    signals.append('buy')
                    weights.append(0.7)
                elif latest['macd'] < latest['macd_signal'] and previous['macd'] >= previous['macd_signal']:
                    signals.append('sell')
                    weights.append(0.7)
                else:
                    signals.append('hold')
                    weights.append(0.3)
            
            # Bollinger Bands signals
            if not pd.isna(latest['bb_upper']) and not pd.isna(latest['bb_lower']):
                if latest['close'] <= latest['bb_lower']:
                    signals.append('buy')
                    weights.append(0.6)
                elif latest['close'] >= latest['bb_upper']:
                    signals.append('sell')
                    weights.append(0.6)
                else:
                    signals.append('hold')
                    weights.append(0.2)
            
            # EMA crossover signals
            if not pd.isna(latest['ema_12']) and not pd.isna(latest['ema_26']):
                if latest['ema_12'] > latest['ema_26'] and previous['ema_12'] <= previous['ema_26']:
                    signals.append('buy')
                    weights.append(0.5)
                elif latest['ema_12'] < latest['ema_26'] and previous['ema_12'] >= previous['ema_26']:
                    signals.append('sell')
                    weights.append(0.5)
                else:
                    signals.append('hold')
                    weights.append(0.1)
            
            # Calculate weighted signal
            if not signals:
                return 'hold', 0.0
            
            signal_scores = {'buy': 0, 'sell': 0, 'hold': 0}
            total_weight = 0
            
            for signal, weight in zip(signals, weights):
                signal_scores[signal] += weight
                total_weight += weight
            
            # Normalize scores
            for signal in signal_scores:
                signal_scores[signal] /= total_weight
            
            # Get final signal
            final_signal = max(signal_scores, key=signal_scores.get)
            confidence = signal_scores[final_signal]
            
            return final_signal, confidence
            
        except Exception as e:
            return 'hold', 0.0
    
    @staticmethod
    def get_indicator_summary(data):
        """
        Get summary of all indicators for the latest data point
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            summary: Dictionary with indicator summary
        """
        if len(data) == 0:
            return {}
        
        latest = data.iloc[-1]
        summary = {}
        
        # RSI
        if 'rsi' in latest and not pd.isna(latest['rsi']):
            summary['RSI'] = {
                'value': latest['rsi'],
                'signal': 'Oversold' if latest['rsi'] < 30 else 'Overbought' if latest['rsi'] > 70 else 'Neutral'
            }
        
        # MACD
        if 'macd' in latest and 'macd_signal' in latest and not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']):
            summary['MACD'] = {
                'value': latest['macd'],
                'signal': 'Bullish' if latest['macd'] > latest['macd_signal'] else 'Bearish'
            }
        
        # Bollinger Bands
        if 'bb_upper' in latest and 'bb_lower' in latest and not pd.isna(latest['bb_upper']) and not pd.isna(latest['bb_lower']):
            bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
            summary['Bollinger Bands'] = {
                'position': bb_position,
                'signal': 'Oversold' if bb_position < 0.2 else 'Overbought' if bb_position > 0.8 else 'Neutral'
            }
        
        # Moving Averages
        if 'ema_12' in latest and 'ema_26' in latest and not pd.isna(latest['ema_12']) and not pd.isna(latest['ema_26']):
            summary['EMA Crossover'] = {
                'signal': 'Bullish' if latest['ema_12'] > latest['ema_26'] else 'Bearish'
            }
        
        # Volume
        if 'volume_ratio' in latest and not pd.isna(latest['volume_ratio']):
            summary['Volume'] = {
                'ratio': latest['volume_ratio'],
                'signal': 'High' if latest['volume_ratio'] > 1.5 else 'Low' if latest['volume_ratio'] < 0.5 else 'Normal'
            }
        
        return summary
