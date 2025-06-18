"""
Technical Analysis Module

This module provides comprehensive technical analysis tools and indicators
for financial data analysis and trading signal generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import ta
import warnings
warnings.filterwarnings('ignore')


class TechnicalAnalysis:
    """Comprehensive technical analysis toolkit"""
    
    def __init__(self):
        """Initialize the technical analysis module"""
        self.indicators = {}
        self.signals = {}
        
    def analyze(self, 
                data: pd.DataFrame,
                indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Perform comprehensive technical analysis
        
        Args:
            data: OHLCV data
            indicators: List of indicators to calculate (None = all)
            
        Returns:
            DataFrame with technical indicators
        """
        df = data.copy()
        
        if indicators is None:
            indicators = ['all']
        
        if 'all' in indicators:
            df = self.add_all_indicators(df)
        else:
            for indicator in indicators:
                if indicator == 'trend':
                    df = self.add_trend_indicators(df)
                elif indicator == 'momentum':
                    df = self.add_momentum_indicators(df)
                elif indicator == 'volatility':
                    df = self.add_volatility_indicators(df)
                elif indicator == 'volume':
                    df = self.add_volume_indicators(df)
                elif indicator == 'pattern':
                    df = self.add_pattern_recognition(df)
        
        # Store indicators
        self.indicators = df
        
        return df
    
    def add_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators"""
        df = data.copy()
        df = self.add_trend_indicators(df)
        df = self.add_momentum_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_volume_indicators(df)
        df = self.add_pattern_recognition(df)
        return df
    
    def add_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add trend following indicators"""
        df = data.copy()
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
        
        # Exponential Moving Averages
        for period in [12, 26, 50, 100, 200]:
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # ADX
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # Aroon
        aroon = ta.trend.AroonIndicator(df['close'])
        df['aroon_up'] = aroon.aroon_up()
        df['aroon_down'] = aroon.aroon_down()
        df['aroon_indicator'] = aroon.aroon_indicator()
        
        # CCI
        df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        
        # DPO
        df['dpo'] = ta.trend.DPOIndicator(df['close']).dpo()
        
        # Mass Index
        df['mass_index'] = ta.trend.MassIndex(df['high'], df['low']).mass_index()
        
        # Trix
        df['trix'] = ta.trend.TRIXIndicator(df['close']).trix()
        
        # Vortex
        vortex = ta.trend.VortexIndicator(df['high'], df['low'], df['close'])
        df['vortex_pos'] = vortex.vortex_indicator_pos()
        df['vortex_neg'] = vortex.vortex_indicator_neg()
        
        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
        
        # PSAR
        psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'])
        df['psar'] = psar.psar()
        df['psar_up'] = psar.psar_up()
        df['psar_down'] = psar.psar_down()
        df['psar_up_indicator'] = psar.psar_up_indicator()
        df['psar_down_indicator'] = psar.psar_down_indicator()
        
        return df
    
    def add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        df = data.copy()
        
        # RSI
        for period in [14, 21, 28]:
            df[f'rsi_{period}'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # TSI
        df['tsi'] = ta.momentum.TSIIndicator(df['close']).tsi()
        
        # Ultimate Oscillator
        df['ultimate_oscillator'] = ta.momentum.UltimateOscillator(
            df['high'], df['low'], df['close']
        ).ultimate_oscillator()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(
            df['high'], df['low'], df['close']
        ).williams_r()
        
        # Awesome Oscillator
        df['awesome_oscillator'] = ta.momentum.AwesomeOscillatorIndicator(
            df['high'], df['low']
        ).awesome_oscillator()
        
        # KAMA
        df['kama'] = ta.momentum.KAMAIndicator(df['close']).kama()
        
        # ROC
        for period in [10, 20, 30]:
            df[f'roc_{period}'] = ta.momentum.ROCIndicator(df['close'], window=period).roc()
        
        # PPO
        ppo = ta.momentum.PercentagePriceOscillator(df['close'])
        df['ppo'] = ppo.ppo()
        df['ppo_signal'] = ppo.ppo_signal()
        df['ppo_hist'] = ppo.ppo_hist()
        
        # PVO
        if 'volume' in df.columns:
            pvo = ta.momentum.PercentageVolumeOscillator(df['volume'])
            df['pvo'] = pvo.pvo()
            df['pvo_signal'] = pvo.pvo_signal()
            df['pvo_hist'] = pvo.pvo_hist()
        
        return df
    
    def add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        df = data.copy()
        
        # Bollinger Bands
        for period in [20, 30]:
            bollinger = ta.volatility.BollingerBands(df['close'], window=period)
            df[f'bb_upper_{period}'] = bollinger.bollinger_hband()
            df[f'bb_middle_{period}'] = bollinger.bollinger_mavg()
            df[f'bb_lower_{period}'] = bollinger.bollinger_lband()
            df[f'bb_width_{period}'] = bollinger.bollinger_wband()
            df[f'bb_percent_{period}'] = bollinger.bollinger_pband()
        
        # Average True Range
        for period in [14, 20]:
            df[f'atr_{period}'] = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close'], window=period
            ).average_true_range()
        
        # Keltner Channel
        keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        df['keltner_upper'] = keltner.keltner_channel_hband()
        df['keltner_middle'] = keltner.keltner_channel_mband()
        df['keltner_lower'] = keltner.keltner_channel_lband()
        df['keltner_width'] = keltner.keltner_channel_wband()
        df['keltner_percent'] = keltner.keltner_channel_pband()
        
        # Donchian Channel
        donchian = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
        df['donchian_upper'] = donchian.donchian_channel_hband()
        df['donchian_middle'] = donchian.donchian_channel_mband()
        df['donchian_lower'] = donchian.donchian_channel_lband()
        df['donchian_width'] = donchian.donchian_channel_wband()
        df['donchian_percent'] = donchian.donchian_channel_pband()
        
        # Ulcer Index
        df['ulcer_index'] = ta.volatility.UlcerIndex(df['close']).ulcer_index()
        
        return df
    
    def add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators"""
        df = data.copy()
        
        if 'volume' not in df.columns:
            return df
        
        # On Balance Volume
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Chaikin Money Flow
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).chaikin_money_flow()
        
        # Force Index
        df['force_index'] = ta.volume.ForceIndexIndicator(
            df['close'], df['volume']
        ).force_index()
        
        # Ease of Movement
        eom = ta.volume.EaseOfMovementIndicator(df['high'], df['low'], df['volume'])
        df['eom'] = eom.ease_of_movement()
        df['eom_signal'] = eom.sma_ease_of_movement()
        
        # Volume Price Trend
        df['vpt'] = ta.volume.VolumePriceTrendIndicator(
            df['close'], df['volume']
        ).volume_price_trend()
        
        # Negative Volume Index
        df['nvi'] = ta.volume.NegativeVolumeIndexIndicator(
            df['close'], df['volume']
        ).negative_volume_index()
        
        # VWAP
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            df['high'], df['low'], df['close'], df['volume']
        ).volume_weighted_average_price()
        
        # MFI
        df['mfi'] = ta.volume.MFIIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).money_flow_index()
        
        # Accumulation/Distribution
        df['ad'] = ta.volume.AccDistIndexIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).acc_dist_index()
        
        return df
    
    def add_pattern_recognition(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition"""
        df = data.copy()
        
        # Price patterns
        df['doji'] = self._detect_doji(df)
        df['hammer'] = self._detect_hammer(df)
        df['inverted_hammer'] = self._detect_inverted_hammer(df)
        df['bullish_engulfing'] = self._detect_bullish_engulfing(df)
        df['bearish_engulfing'] = self._detect_bearish_engulfing(df)
        df['morning_star'] = self._detect_morning_star(df)
        df['evening_star'] = self._detect_evening_star(df)
        df['three_white_soldiers'] = self._detect_three_white_soldiers(df)
        df['three_black_crows'] = self._detect_three_black_crows(df)
        
        # Support and Resistance
        df['support'], df['resistance'] = self._detect_support_resistance(df)
        
        # Trend patterns
        df['higher_highs'] = self._detect_higher_highs(df)
        df['lower_lows'] = self._detect_lower_lows(df)
        df['double_top'] = self._detect_double_top(df)
        df['double_bottom'] = self._detect_double_bottom(df)
        
        return df
    
    def _detect_doji(self, data: pd.DataFrame, threshold: float = 0.001) -> pd.Series:
        """Detect Doji candlestick pattern"""
        body = abs(data['close'] - data['open'])
        range_hl = data['high'] - data['low']
        return (body / range_hl < threshold).astype(int)
    
    def _detect_hammer(self, data: pd.DataFrame) -> pd.Series:
        """Detect Hammer candlestick pattern"""
        body = abs(data['close'] - data['open'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        
        return ((lower_shadow > 2 * body) & 
                (upper_shadow < body * 0.3) & 
                (body > 0)).astype(int)
    
    def _detect_inverted_hammer(self, data: pd.DataFrame) -> pd.Series:
        """Detect Inverted Hammer pattern"""
        body = abs(data['close'] - data['open'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        
        return ((upper_shadow > 2 * body) & 
                (lower_shadow < body * 0.3) & 
                (body > 0)).astype(int)
    
    def _detect_bullish_engulfing(self, data: pd.DataFrame) -> pd.Series:
        """Detect Bullish Engulfing pattern"""
        prev_body = data['close'].shift(1) - data['open'].shift(1)
        curr_body = data['close'] - data['open']
        
        return ((prev_body < 0) &  # Previous candle is bearish
                (curr_body > 0) &  # Current candle is bullish
                (data['open'] < data['close'].shift(1)) &  # Opens below previous close
                (data['close'] > data['open'].shift(1))).astype(int)  # Closes above previous open
    
    def _detect_bearish_engulfing(self, data: pd.DataFrame) -> pd.Series:
        """Detect Bearish Engulfing pattern"""
        prev_body = data['close'].shift(1) - data['open'].shift(1)
        curr_body = data['close'] - data['open']
        
        return ((prev_body > 0) &  # Previous candle is bullish
                (curr_body < 0) &  # Current candle is bearish
                (data['open'] > data['close'].shift(1)) &  # Opens above previous close
                (data['close'] < data['open'].shift(1))).astype(int)  # Closes below previous open
    
    def _detect_morning_star(self, data: pd.DataFrame) -> pd.Series:
        """Detect Morning Star pattern (simplified)"""
        first_body = data['close'].shift(2) - data['open'].shift(2)
        second_body = abs(data['close'].shift(1) - data['open'].shift(1))
        third_body = data['close'] - data['open']
        
        return ((first_body < 0) &  # First candle is bearish
                (second_body < abs(first_body) * 0.3) &  # Small middle candle
                (third_body > 0) &  # Third candle is bullish
                (data['close'] > data['open'].shift(2) - first_body/2)).astype(int)
    
    def _detect_evening_star(self, data: pd.DataFrame) -> pd.Series:
        """Detect Evening Star pattern (simplified)"""
        first_body = data['close'].shift(2) - data['open'].shift(2)
        second_body = abs(data['close'].shift(1) - data['open'].shift(1))
        third_body = data['close'] - data['open']
        
        return ((first_body > 0) &  # First candle is bullish
                (second_body < abs(first_body) * 0.3) &  # Small middle candle
                (third_body < 0) &  # Third candle is bearish
                (data['close'] < data['open'].shift(2) + first_body/2)).astype(int)
    
    def _detect_three_white_soldiers(self, data: pd.DataFrame) -> pd.Series:
        """Detect Three White Soldiers pattern"""
        cond1 = data['close'] > data['open']
        cond2 = data['close'].shift(1) > data['open'].shift(1)
        cond3 = data['close'].shift(2) > data['open'].shift(2)
        cond4 = data['close'] > data['close'].shift(1)
        cond5 = data['close'].shift(1) > data['close'].shift(2)
        
        return (cond1 & cond2 & cond3 & cond4 & cond5).astype(int)
    
    def _detect_three_black_crows(self, data: pd.DataFrame) -> pd.Series:
        """Detect Three Black Crows pattern"""
        cond1 = data['close'] < data['open']
        cond2 = data['close'].shift(1) < data['open'].shift(1)
        cond3 = data['close'].shift(2) < data['open'].shift(2)
        cond4 = data['close'] < data['close'].shift(1)
        cond5 = data['close'].shift(1) < data['close'].shift(2)
        
        return (cond1 & cond2 & cond3 & cond4 & cond5).astype(int)
    
    def _detect_support_resistance(self, 
                                  data: pd.DataFrame,
                                  window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Detect support and resistance levels"""
        support = data['low'].rolling(window=window).min()
        resistance = data['high'].rolling(window=window).max()
        return support, resistance
    
    def _detect_higher_highs(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        """Detect higher highs pattern"""
        highs = data['high'].rolling(window=period).max()
        return (highs > highs.shift(period)).astype(int)
    
    def _detect_lower_lows(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        """Detect lower lows pattern"""
        lows = data['low'].rolling(window=period).min()
        return (lows < lows.shift(period)).astype(int)
    
    def _detect_double_top(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Detect double top pattern (simplified)"""
        highs = data['high'].rolling(window=window).max()
        prev_highs = highs.shift(window)
        
        # Look for two similar highs
        similar_highs = abs(highs - prev_highs) / highs < 0.02
        
        # With a valley in between
        valley = data['low'].rolling(window=window//2).min()
        valley_depth = (highs - valley) / highs > 0.05
        
        return (similar_highs & valley_depth).astype(int)
    
    def _detect_double_bottom(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Detect double bottom pattern (simplified)"""
        lows = data['low'].rolling(window=window).min()
        prev_lows = lows.shift(window)
        
        # Look for two similar lows
        similar_lows = abs(lows - prev_lows) / lows < 0.02
        
        # With a peak in between
        peak = data['high'].rolling(window=window//2).max()
        peak_height = (peak - lows) / lows > 0.05
        
        return (similar_lows & peak_height).astype(int)
    
    def generate_signals(self, 
                        data: pd.DataFrame,
                        strategy: str = 'combined') -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators
        
        Args:
            data: DataFrame with technical indicators
            strategy: Signal generation strategy
            
        Returns:
            DataFrame with trading signals
        """
        df = data.copy()
        
        if strategy == 'ma_crossover':
            df['signal'] = self._ma_crossover_signals(df)
        elif strategy == 'rsi_oversold':
            df['signal'] = self._rsi_signals(df)
        elif strategy == 'macd':
            df['signal'] = self._macd_signals(df)
        elif strategy == 'bollinger':
            df['signal'] = self._bollinger_signals(df)
        elif strategy == 'combined':
            df['signal'] = self._combined_signals(df)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Store signals
        self.signals = df[['signal']]
        
        return df
    
    def _ma_crossover_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate MA crossover signals"""
        signals = pd.Series(0, index=data.index)
        
        if 'sma_50' in data.columns and 'sma_200' in data.columns:
            # Golden cross (bullish)
            signals[(data['sma_50'] > data['sma_200']) & 
                   (data['sma_50'].shift(1) <= data['sma_200'].shift(1))] = 1
            
            # Death cross (bearish)
            signals[(data['sma_50'] < data['sma_200']) & 
                   (data['sma_50'].shift(1) >= data['sma_200'].shift(1))] = -1
        
        return signals
    
    def _rsi_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate RSI-based signals"""
        signals = pd.Series(0, index=data.index)
        
        if 'rsi_14' in data.columns:
            # Oversold (bullish)
            signals[data['rsi_14'] < 30] = 1
            
            # Overbought (bearish)
            signals[data['rsi_14'] > 70] = -1
        
        return signals
    
    def _macd_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate MACD signals"""
        signals = pd.Series(0, index=data.index)
        
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            # MACD crosses above signal (bullish)
            signals[(data['macd'] > data['macd_signal']) & 
                   (data['macd'].shift(1) <= data['macd_signal'].shift(1))] = 1
            
            # MACD crosses below signal (bearish)
            signals[(data['macd'] < data['macd_signal']) & 
                   (data['macd'].shift(1) >= data['macd_signal'].shift(1))] = -1
        
        return signals
    
    def _bollinger_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate Bollinger Band signals"""
        signals = pd.Series(0, index=data.index)
        
        if all(col in data.columns for col in ['close', 'bb_lower_20', 'bb_upper_20']):
            # Price touches lower band (bullish)
            signals[data['close'] <= data['bb_lower_20']] = 1
            
            # Price touches upper band (bearish)
            signals[data['close'] >= data['bb_upper_20']] = -1
        
        return signals
    
    def _combined_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate combined signals from multiple indicators"""
        signal_sum = pd.Series(0, index=data.index)
        
        # Add all individual signals
        signal_sum += self._ma_crossover_signals(data)
        signal_sum += self._rsi_signals(data)
        signal_sum += self._macd_signals(data)
        signal_sum += self._bollinger_signals(data)
        
        # Convert to final signal
        signals = pd.Series(0, index=data.index)
        signals[signal_sum >= 2] = 1  # Strong buy
        signals[signal_sum <= -2] = -1  # Strong sell
        
        return signals
    
    def calculate_signal_strength(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate signal strength based on multiple indicators
        
        Returns:
            Series with signal strength scores (-100 to 100)
        """
        strength = pd.Series(0.0, index=data.index)
        count = 0
        
        # RSI contribution
        if 'rsi_14' in data.columns:
            rsi_score = (50 - data['rsi_14']) * 2  # -100 to 100
            strength += rsi_score
            count += 1
        
        # MACD contribution
        if 'macd_diff' in data.columns:
            macd_normalized = data['macd_diff'] / data['close'] * 1000
            macd_score = np.clip(macd_normalized, -100, 100)
            strength += macd_score
            count += 1
        
        # Bollinger Band contribution
        if 'bb_percent_20' in data.columns:
            bb_score = (0.5 - data['bb_percent_20']) * 200  # -100 to 100
            strength += bb_score
            count += 1
        
        # ADX contribution (trend strength)
        if 'adx' in data.columns:
            adx_weight = data['adx'] / 100
            if 'adx_pos' in data.columns and 'adx_neg' in data.columns:
                adx_direction = np.where(data['adx_pos'] > data['adx_neg'], 1, -1)
                adx_score = adx_direction * adx_weight * 50
                strength += adx_score
                count += 1
        
        # Average the scores
        if count > 0:
            strength /= count
        
        return strength