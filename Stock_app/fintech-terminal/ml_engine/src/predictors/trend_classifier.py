"""
Trend Classification Model

This module implements various machine learning models for classifying
market trends (bullish, bearish, neutral) based on technical indicators
and market data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')


class TrendClassifier:
    """Multi-model trend classification system"""
    
    def __init__(self, 
                 model_type: str = 'ensemble',
                 trend_periods: int = 5,
                 trend_threshold: float = 0.02):
        """
        Initialize the trend classifier
        
        Args:
            model_type: Type of model ('rf', 'gb', 'xgb', 'ensemble')
            trend_periods: Number of periods to calculate trend
            trend_threshold: Threshold for trend classification
        """
        self.model_type = model_type
        self.trend_periods = trend_periods
        self.trend_threshold = trend_threshold
        
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the classification models"""
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        }
    
    def calculate_trend_labels(self, 
                              prices: pd.Series, 
                              method: str = 'return') -> pd.Series:
        """
        Calculate trend labels from price data
        
        Args:
            prices: Series of prices
            method: Method for trend calculation ('return', 'ma_cross')
            
        Returns:
            Series of trend labels
        """
        if method == 'return':
            # Calculate returns over trend_periods
            returns = prices.pct_change(self.trend_periods)
            
            # Classify trends
            conditions = [
                returns > self.trend_threshold,
                returns < -self.trend_threshold
            ]
            choices = ['bullish', 'bearish']
            trends = pd.Series(
                np.select(conditions, choices, default='neutral'),
                index=prices.index
            )
            
        elif method == 'ma_cross':
            # Moving average crossover method
            ma_short = prices.rolling(window=self.trend_periods).mean()
            ma_long = prices.rolling(window=self.trend_periods * 3).mean()
            
            diff = (ma_short - ma_long) / ma_long
            
            conditions = [
                diff > self.trend_threshold,
                diff < -self.trend_threshold
            ]
            choices = ['bullish', 'bearish']
            trends = pd.Series(
                np.select(conditions, choices, default='neutral'),
                index=prices.index
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return trends
    
    def prepare_features(self, 
                        data: pd.DataFrame,
                        feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Prepare features for trend classification
        
        Args:
            data: DataFrame with market data
            feature_cols: List of feature columns
            
        Returns:
            DataFrame with prepared features
        """
        features = data.copy()
        
        if feature_cols is not None:
            features = features[feature_cols]
        
        # Add technical indicators if not present
        if 'rsi' not in features.columns and 'close' in data.columns:
            features['rsi'] = self._calculate_rsi(data['close'])
        
        if 'macd' not in features.columns and 'close' in data.columns:
            features['macd'], features['macd_signal'], features['macd_diff'] = \
                self._calculate_macd(data['close'])
        
        if 'bb_upper' not in features.columns and 'close' in data.columns:
            features['bb_upper'], features['bb_middle'], features['bb_lower'] = \
                self._calculate_bollinger_bands(data['close'])
        
        # Volume indicators
        if 'volume' in data.columns:
            features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            features['volume_trend'] = data['volume'].pct_change(5)
        
        # Price-based features
        if 'close' in data.columns and 'open' in data.columns:
            features['price_change'] = (data['close'] - data['open']) / data['open']
            features['high_low_ratio'] = (data['high'] - data['low']) / data['close']
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicators"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        diff = macd - signal
        
        return macd, signal, diff
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (2 * std)
        lower = middle - (2 * std)
        
        return upper, middle, lower
    
    def train(self,
              X_train: pd.DataFrame,
              y_train: pd.Series,
              optimize_hyperparameters: bool = False) -> Dict:
        """
        Train the trend classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            optimize_hyperparameters: Whether to perform hyperparameter optimization
            
        Returns:
            Training results
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        results = {}
        
        if self.model_type == 'ensemble':
            # Train all models
            for name, model in self.models.items():
                if optimize_hyperparameters:
                    model = self._optimize_hyperparameters(model, X_scaled, y_encoded)
                    self.models[name] = model
                
                model.fit(X_scaled, y_encoded)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5)
                results[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
        else:
            # Train single model
            model = self.models[self.model_type]
            
            if optimize_hyperparameters:
                model = self._optimize_hyperparameters(model, X_scaled, y_encoded)
                self.models[self.model_type] = model
            
            model.fit(X_scaled, y_encoded)
            
            cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5)
            results[self.model_type] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        
        self.is_fitted = True
        return results
    
    def _optimize_hyperparameters(self, model, X: np.ndarray, y: np.ndarray):
        """Optimize model hyperparameters using GridSearchCV"""
        param_grids = {
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'GradientBoostingClassifier': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'XGBClassifier': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        }
        
        model_name = type(model).__name__
        if model_name in param_grids:
            grid_search = GridSearchCV(
                model,
                param_grids[model_name],
                cv=3,
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X, y)
            return grid_search.best_estimator_
        
        return model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make trend predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'ensemble':
            # Ensemble prediction (voting)
            predictions = []
            for model in self.models.values():
                pred = model.predict(X_scaled)
                predictions.append(pred)
            
            # Majority voting
            predictions = np.array(predictions)
            ensemble_pred = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x)), 
                axis=0, 
                arr=predictions
            )
            
            return self.label_encoder.inverse_transform(ensemble_pred)
        else:
            # Single model prediction
            model = self.models[self.model_type]
            predictions = model.predict(X_scaled)
            return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'ensemble':
            # Average probabilities from all models
            probas = []
            for model in self.models.values():
                proba = model.predict_proba(X_scaled)
                probas.append(proba)
            
            return np.mean(probas, axis=0)
        else:
            model = self.models[self.model_type]
            return model.predict_proba(X_scaled)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y_test, predictions)
        
        # Feature importance (for tree-based models)
        feature_importance = {}
        if self.model_type != 'ensemble':
            model = self.models[self.model_type]
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_importance = dict(zip(X_test.columns, importance))
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_importance': feature_importance
        }
    
    def get_trend_signals(self, 
                         data: pd.DataFrame,
                         confidence_threshold: float = 0.7) -> pd.DataFrame:
        """
        Generate trading signals based on trend predictions
        
        Args:
            data: Market data
            confidence_threshold: Minimum confidence for signal generation
            
        Returns:
            DataFrame with trend signals
        """
        # Prepare features
        features = self.prepare_features(data)
        
        # Get predictions and probabilities
        predictions = self.predict(features)
        probabilities = self.predict_proba(features)
        
        # Create signals DataFrame
        signals = pd.DataFrame(index=features.index)
        signals['trend'] = predictions
        signals['confidence'] = np.max(probabilities, axis=1)
        
        # Generate trading signals
        signals['signal'] = 0
        signals.loc[
            (signals['trend'] == 'bullish') & 
            (signals['confidence'] >= confidence_threshold), 
            'signal'
        ] = 1
        signals.loc[
            (signals['trend'] == 'bearish') & 
            (signals['confidence'] >= confidence_threshold), 
            'signal'
        ] = -1
        
        return signals
    
    def save_model(self, path: str):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(path, f'{name}_model.pkl'))
        
        # Save scaler and encoder
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(path, 'label_encoder.pkl'))
        
        # Save config
        config = {
            'model_type': self.model_type,
            'trend_periods': self.trend_periods,
            'trend_threshold': self.trend_threshold
        }
        joblib.dump(config, os.path.join(path, 'config.pkl'))
    
    def load_model(self, path: str):
        """Load a trained model"""
        import os
        
        # Load models
        for name in ['rf', 'gb', 'xgb']:
            model_path = os.path.join(path, f'{name}_model.pkl')
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
        
        # Load scaler and encoder
        self.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        self.label_encoder = joblib.load(os.path.join(path, 'label_encoder.pkl'))
        
        # Load config
        config = joblib.load(os.path.join(path, 'config.pkl'))
        self.model_type = config['model_type']
        self.trend_periods = config['trend_periods']
        self.trend_threshold = config['trend_threshold']
        
        self.is_fitted = True