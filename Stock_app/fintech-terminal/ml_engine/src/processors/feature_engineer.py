"""
Feature Engineering Module

This module creates advanced features for financial machine learning models,
including technical indicators, market microstructure features, and more.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import ta
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Advanced feature engineering for financial data"""
    
    def __init__(self,
                 include_technical: bool = True,
                 include_microstructure: bool = True,
                 include_statistical: bool = True,
                 include_rolling: bool = True,
                 feature_selection: bool = False):
        """
        Initialize the feature engineer
        
        Args:
            include_technical: Include technical indicators
            include_microstructure: Include market microstructure features
            include_statistical: Include statistical features
            include_rolling: Include rolling window features
            feature_selection: Whether to perform feature selection
        """
        self.include_technical = include_technical
        self.include_microstructure = include_microstructure
        self.include_statistical = include_statistical
        self.include_rolling = include_rolling
        self.feature_selection = feature_selection
        
        self.feature_selector = None
        self.selected_features = None
        self.pca = None
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to engineer all features
        
        Args:
            data: OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        df = data.copy()
        
        # Basic price features
        df = self._add_price_features(df)
        
        # Technical indicators
        if self.include_technical:
            df = self._add_technical_indicators(df)
        
        # Market microstructure features
        if self.include_microstructure:
            df = self._add_microstructure_features(df)
        
        # Statistical features
        if self.include_statistical:
            df = self._add_statistical_features(df)
        
        # Rolling window features
        if self.include_rolling:
            df = self._add_rolling_features(df)
        
        # Drop NaN values created by indicators
        df = df.dropna()
        
        return df
    
    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-derived features"""
        df = data.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Gaps
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_percentage'] = df['gap'] / df['close'].shift(1)
        
        # Price position within the day
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Money flow
        if 'volume' in df.columns:
            df['money_flow'] = df['typical_price'] * df['volume']
        
        return df
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        df = data.copy()
        
        # Trend Indicators
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = bollinger.bollinger_wband()
        df['bb_percent'] = bollinger.bollinger_pband()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ADX
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Volume indicators
        if 'volume' in df.columns:
            # OBV
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            
            # Volume SMA
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # VWAP
            df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
                df['high'], df['low'], df['close'], df['volume']
            ).volume_weighted_average_price()
            
            # MFI
            df['mfi'] = ta.volume.MFIIndicator(
                df['high'], df['low'], df['close'], df['volume']
            ).money_flow_index()
        
        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
        
        return df
    
    def _add_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        df = data.copy()
        
        # Spread
        df['spread'] = df['high'] - df['low']
        df['spread_percentage'] = df['spread'] / df['close']
        
        # Realized volatility
        df['realized_volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Parkinson volatility
        df['parkinson_volatility'] = np.sqrt(
            252 * df['log_returns'].rolling(window=20).var() / (4 * np.log(2))
        )
        
        # Garman-Klass volatility
        df['gk_volatility'] = np.sqrt(
            252 * (
                0.5 * np.log(df['high'] / df['low']).rolling(window=20).var() -
                (2 * np.log(2) - 1) * np.log(df['close'] / df['open']).rolling(window=20).var()
            )
        )
        
        # Kyle's lambda (price impact)
        if 'volume' in df.columns:
            df['kyle_lambda'] = df['returns'].abs() / df['volume']
            df['kyle_lambda_ma'] = df['kyle_lambda'].rolling(window=20).mean()
        
        # Amihud illiquidity
        if 'volume' in df.columns:
            df['amihud_illiquidity'] = df['returns'].abs() / (df['volume'] * df['close'])
            df['amihud_illiquidity_ma'] = df['amihud_illiquidity'].rolling(window=20).mean()
        
        # Roll's implied spread
        df['roll_spread'] = 2 * np.sqrt(-df['returns'].rolling(window=20).cov(df['returns'].shift(1)))
        
        return df
    
    def _add_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        df = data.copy()
        
        # Rolling statistics
        windows = [5, 10, 20, 50]
        
        for window in windows:
            # Returns statistics
            df[f'returns_mean_{window}'] = df['returns'].rolling(window=window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window=window).std()
            df[f'returns_skew_{window}'] = df['returns'].rolling(window=window).skew()
            df[f'returns_kurt_{window}'] = df['returns'].rolling(window=window).kurt()
            
            # Price statistics
            df[f'price_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'price_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'price_min_{window}'] = df['close'].rolling(window=window).min()
            df[f'price_max_{window}'] = df['close'].rolling(window=window).max()
            
            # Z-score
            df[f'price_zscore_{window}'] = (
                (df['close'] - df[f'price_mean_{window}']) / df[f'price_std_{window}']
            )
        
        # Autocorrelation
        for lag in [1, 5, 10]:
            df[f'returns_autocorr_{lag}'] = df['returns'].rolling(window=20).apply(
                lambda x: x.autocorr(lag=lag), raw=False
            )
        
        # Entropy
        df['returns_entropy'] = df['returns'].rolling(window=20).apply(
            lambda x: stats.entropy(np.histogram(x, bins=10)[0] + 1e-10), raw=False
        )
        
        return df
    
    def _add_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features"""
        df = data.copy()
        
        # Rolling returns
        for period in [5, 10, 20, 60]:
            df[f'returns_{period}d'] = df['close'].pct_change(periods=period)
            df[f'log_returns_{period}d'] = np.log(df['close'] / df['close'].shift(period))
        
        # Rolling highs and lows
        for period in [10, 20, 50]:
            df[f'high_{period}d'] = df['high'].rolling(window=period).max()
            df[f'low_{period}d'] = df['low'].rolling(window=period).min()
            df[f'high_ratio_{period}d'] = df['close'] / df[f'high_{period}d']
            df[f'low_ratio_{period}d'] = df['close'] / df[f'low_{period}d']
        
        # Support and resistance levels
        df['resistance'] = df['high'].rolling(window=20).max()
        df['support'] = df['low'].rolling(window=20).min()
        df['distance_to_resistance'] = (df['resistance'] - df['close']) / df['close']
        df['distance_to_support'] = (df['close'] - df['support']) / df['close']
        
        # Trend strength
        for period in [10, 20, 50]:
            df[f'trend_strength_{period}'] = (
                df['close'] - df['close'].shift(period)
            ) / df['close'].shift(period)
        
        # Volume patterns
        if 'volume' in df.columns:
            for period in [5, 10, 20]:
                df[f'volume_mean_{period}'] = df['volume'].rolling(window=period).mean()
                df[f'volume_std_{period}'] = df['volume'].rolling(window=period).std()
                df[f'volume_trend_{period}'] = (
                    df['volume'] - df[f'volume_mean_{period}']
                ) / df[f'volume_mean_{period}']
        
        return df
    
    def select_features(self,
                       X: pd.DataFrame,
                       y: Union[pd.Series, np.ndarray],
                       method: str = 'mutual_info',
                       k: int = 50) -> pd.DataFrame:
        """
        Select best features using various methods
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method ('mutual_info', 'f_classif', 'pca')
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        if method == 'mutual_info':
            if len(np.unique(y)) > 10:  # Regression
                from sklearn.feature_selection import mutual_info_regression
                selector = SelectKBest(score_func=mutual_info_regression, k=k)
            else:  # Classification
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
        
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        
        elif method == 'pca':
            self.pca = PCA(n_components=k)
            X_transformed = self.pca.fit_transform(X)
            return pd.DataFrame(
                X_transformed,
                index=X.index,
                columns=[f'pca_{i}' for i in range(k)]
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Fit selector
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        self.selected_features = selected_features
        self.feature_selector = selector
        
        return pd.DataFrame(X_selected, index=X.index, columns=selected_features)
    
    def create_interaction_features(self,
                                   data: pd.DataFrame,
                                   feature_pairs: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
        """
        Create interaction features between columns
        
        Args:
            data: Input data
            feature_pairs: List of feature pairs to create interactions
            
        Returns:
            DataFrame with interaction features
        """
        df = data.copy()
        
        if feature_pairs is None:
            # Default important interactions
            feature_pairs = [
                ('rsi', 'volume_ratio'),
                ('macd', 'adx'),
                ('bb_percent', 'rsi'),
                ('returns', 'volume_ratio'),
                ('atr', 'volume_ratio')
            ]
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplication
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                
                # Division (with small epsilon to avoid division by zero)
                df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-10)
                
                # Difference
                df[f'{feat1}_minus_{feat2}'] = df[feat1] - df[feat2]
        
        return df
    
    def create_polynomial_features(self,
                                  data: pd.DataFrame,
                                  columns: List[str],
                                  degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features
        
        Args:
            data: Input data
            columns: Columns to create polynomial features
            degree: Polynomial degree
            
        Returns:
            DataFrame with polynomial features
        """
        df = data.copy()
        
        for col in columns:
            if col in df.columns:
                for d in range(2, degree + 1):
                    df[f'{col}_pow{d}'] = df[col] ** d
        
        return df
    
    def create_lagged_features(self,
                              data: pd.DataFrame,
                              columns: List[str],
                              lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Create lagged features
        
        Args:
            data: Input data
            columns: Columns to create lags
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        df = data.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_target_encoding(self,
                              data: pd.DataFrame,
                              categorical_col: str,
                              target_col: str,
                              smoothing: float = 1.0) -> pd.DataFrame:
        """
        Create target encoding for categorical features
        
        Args:
            data: Input data
            categorical_col: Categorical column name
            target_col: Target column name
            smoothing: Smoothing parameter
            
        Returns:
            DataFrame with target encoding
        """
        df = data.copy()
        
        # Calculate global mean
        global_mean = df[target_col].mean()
        
        # Calculate category statistics
        agg = df.groupby(categorical_col)[target_col].agg(['count', 'mean'])
        
        # Apply smoothing
        smoothed_mean = (
            (agg['count'] * agg['mean'] + smoothing * global_mean) /
            (agg['count'] + smoothing)
        )
        
        # Map to original data
        df[f'{categorical_col}_target_encoded'] = df[categorical_col].map(smoothed_mean)
        
        return df
    
    def get_feature_importance(self, 
                              X: pd.DataFrame,
                              y: Union[pd.Series, np.ndarray],
                              model_type: str = 'random_forest') -> pd.DataFrame:
        """
        Calculate feature importance
        
        Args:
            X: Feature matrix
            y: Target variable
            model_type: Type of model to use
            
        Returns:
            DataFrame with feature importance scores
        """
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            
            if len(np.unique(y)) > 10:  # Regression
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:  # Classification
                model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        elif model_type == 'xgboost':
            import xgboost as xgb
            
            if len(np.unique(y)) > 10:  # Regression
                model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            else:  # Classification
                model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Fit model
        model.fit(X, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance