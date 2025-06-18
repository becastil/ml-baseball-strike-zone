"""
Data Preprocessing Pipeline

This module handles data cleaning, normalization, and preparation
for machine learning models in financial analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Comprehensive data preprocessing pipeline for financial data"""
    
    def __init__(self,
                 scaling_method: str = 'standard',
                 imputation_method: str = 'forward_fill',
                 outlier_method: str = 'iqr',
                 handle_weekends: bool = True):
        """
        Initialize the data preprocessor
        
        Args:
            scaling_method: Method for scaling ('standard', 'minmax', 'robust')
            imputation_method: Method for handling missing values
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
            handle_weekends: Whether to handle weekend gaps in data
        """
        self.scaling_method = scaling_method
        self.imputation_method = imputation_method
        self.outlier_method = outlier_method
        self.handle_weekends = handle_weekends
        
        self.scaler = self._get_scaler()
        self.imputer = self._get_imputer()
        self.feature_ranges = {}
        
    def _get_scaler(self):
        """Get the appropriate scaler based on method"""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        return scalers.get(self.scaling_method, StandardScaler())
    
    def _get_imputer(self):
        """Get the appropriate imputer based on method"""
        if self.imputation_method == 'mean':
            return SimpleImputer(strategy='mean')
        elif self.imputation_method == 'median':
            return SimpleImputer(strategy='median')
        elif self.imputation_method == 'knn':
            return KNNImputer(n_neighbors=5)
        else:
            return None  # Use pandas methods
    
    def fetch_data(self,
                   symbol: str,
                   start_date: Union[str, datetime],
                   end_date: Union[str, datetime] = None,
                   interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data (default: today)
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Download data
        data = yf.download(symbol, start=start_date, end=end_date, 
                          interval=interval, progress=False)
        
        # Clean column names
        data.columns = [col.lower() for col in data.columns]
        
        return data
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean financial data
        
        Args:
            data: Raw financial data
            
        Returns:
            Cleaned DataFrame
        """
        df = data.copy()
        
        # Remove duplicate indices
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by index
        df = df.sort_index()
        
        # Handle missing values
        if self.imputation_method == 'forward_fill':
            df = df.fillna(method='ffill')
        elif self.imputation_method == 'backward_fill':
            df = df.fillna(method='bfill')
        elif self.imputation_method == 'interpolate':
            df = df.interpolate(method='time')
        elif self.imputer is not None:
            # Use sklearn imputer
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        # Handle weekends and holidays
        if self.handle_weekends:
            df = self._handle_missing_dates(df)
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Ensure all numeric columns are float
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        return df
    
    def _handle_missing_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing dates (weekends, holidays) in financial data"""
        # Create a complete date range
        date_range = pd.date_range(start=data.index.min(), 
                                  end=data.index.max(), 
                                  freq='D')
        
        # Reindex with the complete date range
        df = data.reindex(date_range)
        
        # Forward fill for weekends and holidays
        df = df.fillna(method='ffill')
        
        # Only keep business days if needed
        if self.handle_weekends:
            df = df[df.index.weekday < 5]  # Remove weekends
        
        return df
    
    def detect_outliers(self, 
                       data: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in the data
        
        Args:
            data: Input data
            columns: Columns to check for outliers
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier indicators
        """
        df = data.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outliers = pd.DataFrame(index=df.index)
        
        if self.outlier_method == 'zscore':
            for col in columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers[f'{col}_outlier'] = z_scores > threshold
                
        elif self.outlier_method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers[f'{col}_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        elif self.outlier_method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(df[columns])
            outliers['outlier'] = outlier_labels == -1
        
        return outliers
    
    def remove_outliers(self,
                       data: pd.DataFrame,
                       outlier_mask: pd.DataFrame,
                       method: str = 'remove') -> pd.DataFrame:
        """
        Handle outliers in the data
        
        Args:
            data: Input data
            outlier_mask: DataFrame with outlier indicators
            method: How to handle outliers ('remove', 'cap', 'transform')
            
        Returns:
            DataFrame with outliers handled
        """
        df = data.copy()
        
        if method == 'remove':
            # Remove rows with any outliers
            mask = ~outlier_mask.any(axis=1)
            df = df[mask]
            
        elif method == 'cap':
            # Cap outliers at percentiles
            for col in df.select_dtypes(include=[np.number]).columns:
                if f'{col}_outlier' in outlier_mask.columns:
                    lower = df[col].quantile(0.01)
                    upper = df[col].quantile(0.99)
                    df[col] = df[col].clip(lower=lower, upper=upper)
        
        elif method == 'transform':
            # Log transform to reduce impact of outliers
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].min() > 0:  # Only for positive values
                    df[col] = np.log1p(df[col])
        
        return df
    
    def normalize_data(self,
                      data: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      fit: bool = True) -> pd.DataFrame:
        """
        Normalize/scale the data
        
        Args:
            data: Input data
            columns: Columns to normalize
            fit: Whether to fit the scaler
            
        Returns:
            Normalized DataFrame
        """
        df = data.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        if fit:
            df[columns] = self.scaler.fit_transform(df[columns])
            # Store feature ranges for later use
            if self.scaling_method == 'minmax':
                self.feature_ranges = {
                    col: (self.scaler.data_min_[i], self.scaler.data_max_[i])
                    for i, col in enumerate(columns)
                }
        else:
            df[columns] = self.scaler.transform(df[columns])
        
        return df
    
    def create_sequences(self,
                        data: pd.DataFrame,
                        sequence_length: int,
                        target_column: str,
                        stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling
        
        Args:
            data: Input data
            sequence_length: Length of input sequences
            target_column: Name of target column
            stride: Stride for creating sequences
            
        Returns:
            X and y arrays for modeling
        """
        # Separate features and target
        features = data.drop(columns=[target_column]).values
        target = data[target_column].values
        
        X, y = [], []
        
        for i in range(0, len(data) - sequence_length, stride):
            X.append(features[i:i + sequence_length])
            y.append(target[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features to the data
        
        Args:
            data: Input data with datetime index
            
        Returns:
            DataFrame with additional time features
        """
        df = data.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Add time features
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['week_of_year'] = df.index.isocalendar().week
        
        # Cyclical encoding for periodic features
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Is it a month/quarter end?
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        return df
    
    def split_data(self,
                  data: pd.DataFrame,
                  train_size: float = 0.7,
                  val_size: float = 0.15,
                  shuffle: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            data: Input data
            train_size: Proportion for training
            val_size: Proportion for validation
            shuffle: Whether to shuffle (not recommended for time series)
            
        Returns:
            Train, validation, and test DataFrames
        """
        n = len(data)
        
        if shuffle:
            # Shuffle data (not recommended for time series)
            data = data.sample(frac=1, random_state=42)
        
        # Calculate split indices
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        
        # Split data
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        return train_data, val_data, test_data
    
    def prepare_ml_data(self,
                       data: pd.DataFrame,
                       target_column: str,
                       feature_columns: Optional[List[str]] = None,
                       sequence_length: Optional[int] = None) -> Dict:
        """
        Complete preprocessing pipeline for ML
        
        Args:
            data: Raw data
            target_column: Target column name
            feature_columns: Feature columns to use
            sequence_length: Length for sequences (if needed)
            
        Returns:
            Dictionary with processed data
        """
        # Clean data
        df = self.clean_data(data)
        
        # Add time features
        df = self.add_time_features(df)
        
        # Select features
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        # Detect and handle outliers
        outliers = self.detect_outliers(df[feature_columns])
        df = self.remove_outliers(df, outliers, method='cap')
        
        # Normalize features
        df_normalized = self.normalize_data(df, columns=feature_columns)
        
        # Split data
        train_data, val_data, test_data = self.split_data(df_normalized)
        
        # Prepare output
        result = {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'feature_columns': feature_columns,
            'target_column': target_column
        }
        
        # Create sequences if needed
        if sequence_length is not None:
            X_train, y_train = self.create_sequences(
                train_data[feature_columns + [target_column]], 
                sequence_length, 
                target_column
            )
            X_val, y_val = self.create_sequences(
                val_data[feature_columns + [target_column]], 
                sequence_length, 
                target_column
            )
            X_test, y_test = self.create_sequences(
                test_data[feature_columns + [target_column]], 
                sequence_length, 
                target_column
            )
            
            result.update({
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test
            })
        
        return result
    
    def inverse_transform(self, 
                         data: np.ndarray,
                         feature_name: Optional[str] = None) -> np.ndarray:
        """
        Inverse transform normalized data
        
        Args:
            data: Normalized data
            feature_name: Name of the feature (for single feature)
            
        Returns:
            Original scale data
        """
        if hasattr(self.scaler, 'inverse_transform'):
            return self.scaler.inverse_transform(data)
        else:
            # Manual inverse transform for specific feature
            if feature_name and feature_name in self.feature_ranges:
                min_val, max_val = self.feature_ranges[feature_name]
                return data * (max_val - min_val) + min_val
            else:
                raise ValueError("Cannot inverse transform without feature ranges")