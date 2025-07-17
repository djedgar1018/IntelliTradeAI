"""
Data Cleaner Module
Handles data formatting, cleaning, and validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """Class for cleaning and formatting financial data"""
    
    @staticmethod
    def clean_ohlcv_data(data):
        """
        Clean OHLCV data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            cleaned_data: Cleaned DataFrame
        """
        try:
            # Make a copy to avoid modifying original data
            cleaned_data = data.copy()
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in cleaned_data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Convert to numeric
            for col in required_columns:
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
            
            # Remove rows with all NaN values
            cleaned_data = cleaned_data.dropna(subset=required_columns, how='all')
            
            # Forward fill missing values
            cleaned_data[required_columns] = cleaned_data[required_columns].fillna(method='ffill')
            
            # Remove rows where high < low (invalid data)
            invalid_mask = cleaned_data['high'] < cleaned_data['low']
            if invalid_mask.any():
                cleaned_data = cleaned_data[~invalid_mask]
            
            # Remove rows where close is outside high-low range
            invalid_close_mask = (cleaned_data['close'] > cleaned_data['high']) | \
                                (cleaned_data['close'] < cleaned_data['low'])
            if invalid_close_mask.any():
                cleaned_data = cleaned_data[~invalid_close_mask]
            
            # Remove rows with zero or negative prices
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                cleaned_data = cleaned_data[cleaned_data[col] > 0]
            
            # Remove rows with negative volume
            cleaned_data = cleaned_data[cleaned_data['volume'] >= 0]
            
            # Remove extreme outliers (more than 10 standard deviations from mean)
            for col in price_cols:
                mean_val = cleaned_data[col].mean()
                std_val = cleaned_data[col].std()
                if std_val > 0:
                    outlier_mask = np.abs(cleaned_data[col] - mean_val) > (10 * std_val)
                    cleaned_data = cleaned_data[~outlier_mask]
            
            # Sort by index (date)
            cleaned_data = cleaned_data.sort_index()
            
            # Remove duplicates
            cleaned_data = cleaned_data[~cleaned_data.index.duplicated(keep='first')]
            
            return cleaned_data
            
        except Exception as e:
            raise Exception(f"Error cleaning OHLCV data: {str(e)}")
    
    @staticmethod
    def validate_data_quality(data):
        """
        Validate data quality and return quality metrics
        
        Args:
            data: DataFrame with financial data
            
        Returns:
            quality_metrics: Dictionary with quality metrics
        """
        try:
            metrics = {
                'total_rows': len(data),
                'missing_values': {},
                'outliers': {},
                'data_gaps': 0,
                'quality_score': 0.0,
                'issues': []
            }
            
            if len(data) == 0:
                metrics['quality_score'] = 0.0
                metrics['issues'].append("No data available")
                return metrics
            
            # Check for missing values
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col in data.columns:
                    missing_count = data[col].isna().sum()
                    metrics['missing_values'][col] = missing_count
                    if missing_count > 0:
                        metrics['issues'].append(f"Missing values in {col}: {missing_count}")
            
            # Check for outliers
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in data.columns:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                    metrics['outliers'][col] = outliers
                    if outliers > len(data) * 0.05:  # More than 5% outliers
                        metrics['issues'].append(f"High number of outliers in {col}: {outliers}")
            
            # Check for data gaps (if data has datetime index)
            if isinstance(data.index, pd.DatetimeIndex):
                expected_freq = pd.infer_freq(data.index)
                if expected_freq:
                    expected_range = pd.date_range(start=data.index.min(), 
                                                 end=data.index.max(), 
                                                 freq=expected_freq)
                    missing_dates = expected_range.difference(data.index)
                    metrics['data_gaps'] = len(missing_dates)
                    if metrics['data_gaps'] > 0:
                        metrics['issues'].append(f"Data gaps found: {metrics['data_gaps']} missing periods")
            
            # Calculate quality score
            total_missing = sum(metrics['missing_values'].values())
            total_outliers = sum(metrics['outliers'].values())
            
            missing_penalty = (total_missing / (len(data) * len(required_columns))) * 0.4
            outlier_penalty = min((total_outliers / len(data)) * 0.3, 0.3)
            gap_penalty = min((metrics['data_gaps'] / len(data)) * 0.3, 0.3)
            
            metrics['quality_score'] = max(0.0, 1.0 - missing_penalty - outlier_penalty - gap_penalty)
            
            return metrics
            
        except Exception as e:
            return {
                'total_rows': 0,
                'missing_values': {},
                'outliers': {},
                'data_gaps': 0,
                'quality_score': 0.0,
                'issues': [f"Error validating data: {str(e)}"]
            }
    
    @staticmethod
    def resample_data(data, frequency='1D'):
        """
        Resample data to different frequency
        
        Args:
            data: DataFrame with OHLCV data
            frequency: Target frequency (e.g., '1D', '1H', '5min')
            
        Returns:
            resampled_data: Resampled DataFrame
        """
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("Data must have datetime index for resampling")
            
            # Define aggregation rules
            agg_rules = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            # Only include columns that exist in the data
            available_agg_rules = {k: v for k, v in agg_rules.items() if k in data.columns}
            
            # Resample data
            resampled_data = data.resample(frequency).agg(available_agg_rules)
            
            # Remove rows with NaN values (periods with no data)
            resampled_data = resampled_data.dropna()
            
            return resampled_data
            
        except Exception as e:
            raise Exception(f"Error resampling data: {str(e)}")
    
    @staticmethod
    def normalize_data(data, method='standard'):
        """
        Normalize data for ML model training
        
        Args:
            data: DataFrame with numeric data
            method: Normalization method ('standard', 'minmax', 'robust')
            
        Returns:
            normalized_data: Normalized DataFrame
            scaler: Fitted scaler object
        """
        try:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
            
            # Select scaler based on method
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            # Fit and transform data
            normalized_data = data.copy()
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            normalized_data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
            
            return normalized_data, scaler
            
        except Exception as e:
            raise Exception(f"Error normalizing data: {str(e)}")
    
    @staticmethod
    def handle_missing_values(data, method='forward_fill'):
        """
        Handle missing values in data
        
        Args:
            data: DataFrame with missing values
            method: Method to handle missing values ('forward_fill', 'backward_fill', 'interpolate', 'drop')
            
        Returns:
            cleaned_data: DataFrame with handled missing values
        """
        try:
            cleaned_data = data.copy()
            
            if method == 'forward_fill':
                cleaned_data = cleaned_data.fillna(method='ffill')
            elif method == 'backward_fill':
                cleaned_data = cleaned_data.fillna(method='bfill')
            elif method == 'interpolate':
                cleaned_data = cleaned_data.interpolate()
            elif method == 'drop':
                cleaned_data = cleaned_data.dropna()
            else:
                raise ValueError(f"Unknown missing value handling method: {method}")
            
            return cleaned_data
            
        except Exception as e:
            raise Exception(f"Error handling missing values: {str(e)}")
    
    @staticmethod
    def detect_anomalies(data, method='iqr', threshold=3):
        """
        Detect anomalies in data
        
        Args:
            data: DataFrame with numeric data
            method: Detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for anomaly detection
            
        Returns:
            anomaly_mask: Boolean mask indicating anomalies
        """
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if method == 'iqr':
                Q1 = numeric_data.quantile(0.25)
                Q3 = numeric_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                anomaly_mask = ((numeric_data < lower_bound) | (numeric_data > upper_bound)).any(axis=1)
                
            elif method == 'zscore':
                z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
                anomaly_mask = (z_scores > threshold).any(axis=1)
                
            elif method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomaly_labels = iso_forest.fit_predict(numeric_data.fillna(numeric_data.mean()))
                anomaly_mask = anomaly_labels == -1
                
            else:
                raise ValueError(f"Unknown anomaly detection method: {method}")
            
            return anomaly_mask
            
        except Exception as e:
            raise Exception(f"Error detecting anomalies: {str(e)}")
    
    @staticmethod
    def get_data_summary(data):
        """
        Get comprehensive summary of data
        
        Args:
            data: DataFrame
            
        Returns:
            summary: Dictionary with data summary
        """
        try:
            summary = {
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'missing_values': data.isnull().sum().to_dict(),
                'numeric_summary': {},
                'date_range': None
            }
            
            # Numeric summary
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                summary['numeric_summary'] = data[numeric_columns].describe().to_dict()
            
            # Date range
            if isinstance(data.index, pd.DatetimeIndex):
                summary['date_range'] = {
                    'start': data.index.min(),
                    'end': data.index.max(),
                    'periods': len(data)
                }
            
            return summary
            
        except Exception as e:
            return {'error': f"Error creating data summary: {str(e)}"}
