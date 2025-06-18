"""
LSTM-based Price Prediction Model

This module implements a Long Short-Term Memory (LSTM) neural network
for predicting stock prices based on historical data and technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


class PricePredictor:
    """LSTM model for stock price prediction"""
    
    def __init__(self, 
                 sequence_length: int = 60,
                 n_features: int = 10,
                 lstm_units: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize the LSTM price predictor
        
        Args:
            sequence_length: Number of time steps to look back
            n_features: Number of input features
            lstm_units: List of LSTM units for each layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
    def build_model(self) -> Sequential:
        """Build the LSTM model architecture"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=True,
            input_shape=(self.sequence_length, self.n_features)
        ))
        model.add(Dropout(self.dropout_rate))
        model.add(BatchNormalization())
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:]):
            return_sequences = i < len(self.lstm_units) - 2
            model.add(LSTM(units=units, return_sequences=return_sequences))
            model.add(Dropout(self.dropout_rate))
            model.add(BatchNormalization())
        
        # Output layers
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def prepare_data(self, 
                     data: pd.DataFrame,
                     target_col: str = 'close',
                     feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training
        
        Args:
            data: DataFrame with time series data
            target_col: Name of target column
            feature_cols: List of feature columns to use
            
        Returns:
            X, y arrays for training
        """
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col != target_col]
        
        # Scale features
        features = data[feature_cols].values
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(data[target_col].iloc[i])
        
        return np.array(X), np.array(y)
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 32,
              verbose: int = 1) -> Dict:
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001),
            ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        # Train model
        validation_data = (X_val, y_val) if X_val is not None else None
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted = True
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_next_n_days(self, 
                           recent_data: pd.DataFrame,
                           n_days: int = 5,
                           feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """
        Predict prices for the next n days
        
        Args:
            recent_data: Recent historical data
            n_days: Number of days to predict
            feature_cols: Feature columns to use
            
        Returns:
            Array of predicted prices
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        if feature_cols is None:
            feature_cols = [col for col in recent_data.columns if col != 'close']
        
        # Scale the recent data
        scaled_data = self.scaler.transform(recent_data[feature_cols].values)
        
        # Prepare the input sequence
        if len(scaled_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} days of data")
        
        predictions = []
        current_sequence = scaled_data[-self.sequence_length:].copy()
        
        for _ in range(n_days):
            # Reshape for prediction
            X = current_sequence.reshape(1, self.sequence_length, self.n_features)
            
            # Make prediction
            pred = self.model.predict(X, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Update sequence (simplified - in practice, you'd update all features)
            # Here we're just shifting and adding the prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = pred  # Assuming first feature is price
        
        return np.array(predictions)
    
    def save_model(self, path: str):
        """Save the trained model and scaler"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save model
        self.model.save(os.path.join(path, 'lstm_model.h5'))
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        
        # Save config
        config = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate
        }
        joblib.dump(config, os.path.join(path, 'config.pkl'))
    
    def load_model(self, path: str):
        """Load a trained model and scaler"""
        # Load model
        self.model = load_model(os.path.join(path, 'lstm_model.h5'))
        
        # Load scaler
        self.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        
        # Load config
        config = joblib.load(os.path.join(path, 'config.pkl'))
        self.sequence_length = config['sequence_length']
        self.n_features = config['n_features']
        self.lstm_units = config['lstm_units']
        self.dropout_rate = config['dropout_rate']
        self.learning_rate = config['learning_rate']
        
        self.is_fitted = True
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        if len(y_test) > 1:
            actual_direction = np.diff(y_test) > 0
            pred_direction = np.diff(predictions.flatten()) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction)
        else:
            directional_accuracy = 0.0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }