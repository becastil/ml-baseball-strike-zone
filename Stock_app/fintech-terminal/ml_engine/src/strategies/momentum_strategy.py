"""
Momentum Trading Strategy

This module implements various momentum-based trading strategies
using technical indicators and machine learning predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    entry_price: float
    entry_date: pd.Timestamp
    size: float
    side: str  # 'long' or 'short'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def current_value(self, current_price: float) -> float:
        """Calculate current position value"""
        if self.side == 'long':
            return self.size * (current_price - self.entry_price)
        else:  # short
            return self.size * (self.entry_price - current_price)
    
    def current_return(self, current_price: float) -> float:
        """Calculate current return percentage"""
        if self.side == 'long':
            return (current_price - self.entry_price) / self.entry_price
        else:  # short
            return (self.entry_price - current_price) / self.entry_price


class MomentumStrategy:
    """Advanced momentum trading strategy"""
    
    def __init__(self,
                 lookback_period: int = 20,
                 momentum_threshold: float = 0.05,
                 use_volume_filter: bool = True,
                 use_volatility_filter: bool = True,
                 risk_per_trade: float = 0.02,
                 max_positions: int = 5):
        """
        Initialize the momentum strategy
        
        Args:
            lookback_period: Period for momentum calculation
            momentum_threshold: Minimum momentum for signal
            use_volume_filter: Filter signals by volume
            use_volatility_filter: Filter signals by volatility
            risk_per_trade: Risk per trade as fraction of capital
            max_positions: Maximum number of concurrent positions
        """
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
        self.use_volume_filter = use_volume_filter
        self.use_volatility_filter = use_volatility_filter
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        
        self.positions = []
        self.closed_positions = []
        self.signals = pd.DataFrame()
    
    def calculate_momentum(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various momentum indicators
        
        Args:
            data: OHLCV data
            
        Returns:
            DataFrame with momentum indicators
        """
        df = data.copy()
        
        # Price momentum
        df['price_momentum'] = df['close'].pct_change(self.lookback_period)
        
        # Rate of change
        df['roc'] = (df['close'] - df['close'].shift(self.lookback_period)) / df['close'].shift(self.lookback_period)
        
        # Relative strength
        gains = df['close'].diff()
        losses = -df['close'].diff()
        gains[gains < 0] = 0
        losses[losses < 0] = 0
        
        avg_gains = gains.rolling(window=self.lookback_period).mean()
        avg_losses = losses.rolling(window=self.lookback_period).mean()
        
        rs = avg_gains / (avg_losses + 1e-10)
        df['rsi_momentum'] = 100 - (100 / (1 + rs))
        
        # Volume-weighted momentum
        if 'volume' in df.columns:
            df['volume_momentum'] = (df['close'] * df['volume']).diff(self.lookback_period) / (df['volume'].rolling(self.lookback_period).sum() + 1e-10)
        
        # Momentum oscillator
        df['momentum_oscillator'] = df['close'] - df['close'].shift(self.lookback_period)
        
        # Acceleration (momentum of momentum)
        df['momentum_acceleration'] = df['price_momentum'].diff()
        
        # Trend strength using ADX
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean()
        
        # Calculate directional indicators
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']
        
        pos_dm = pd.Series(0.0, index=df.index)
        neg_dm = pd.Series(0.0, index=df.index)
        
        pos_dm[(up_move > down_move) & (up_move > 0)] = up_move
        neg_dm[(down_move > up_move) & (down_move > 0)] = down_move
        
        pos_di = 100 * (pos_dm.rolling(window=14).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=14).mean() / atr)
        
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10)
        df['adx'] = dx.rolling(window=14).mean()
        
        # Momentum quality (consistency)
        returns = df['close'].pct_change()
        positive_days = (returns > 0).rolling(window=self.lookback_period).sum()
        df['momentum_quality'] = positive_days / self.lookback_period
        
        return df
    
    def identify_regime(self, data: pd.DataFrame) -> pd.Series:
        """
        Identify market regime (trending, ranging, volatile)
        
        Args:
            data: DataFrame with price and indicators
            
        Returns:
            Series with regime labels
        """
        df = data.copy()
        
        # Calculate regime indicators
        # Trend strength
        sma_short = df['close'].rolling(window=20).mean()
        sma_long = df['close'].rolling(window=50).mean()
        trend_strength = abs(sma_short - sma_long) / sma_long
        
        # Volatility
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=20).std() * np.sqrt(252)
        
        # Range indicator
        range_indicator = (df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()) / df['close']
        
        # Define regimes
        regime = pd.Series('neutral', index=df.index)
        
        # Strong trend
        regime[(trend_strength > 0.05) & (df['adx'] > 25)] = 'trending'
        
        # Range-bound
        regime[(trend_strength < 0.02) & (range_indicator < 0.1)] = 'ranging'
        
        # High volatility
        regime[volatility > volatility.rolling(window=100).mean() + 2 * volatility.rolling(window=100).std()] = 'volatile'
        
        return regime
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum trading signals
        
        Args:
            data: OHLCV data with indicators
            
        Returns:
            DataFrame with trading signals
        """
        # Calculate momentum indicators
        df = self.calculate_momentum(data)
        
        # Identify market regime
        df['regime'] = self.identify_regime(df)
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Basic momentum signals
        strong_momentum = df['price_momentum'] > self.momentum_threshold
        weak_momentum = df['price_momentum'] < -self.momentum_threshold
        
        # Quality filter
        quality_filter = df['momentum_quality'] > 0.6
        
        # Trend filter
        trend_filter = df['adx'] > 20
        
        # Volume filter
        if self.use_volume_filter and 'volume' in df.columns:
            volume_sma = df['volume'].rolling(window=20).mean()
            volume_filter = df['volume'] > volume_sma * 1.2
        else:
            volume_filter = True
        
        # Volatility filter
        if self.use_volatility_filter:
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=20).std()
            volatility_filter = volatility < volatility.rolling(window=100).mean() * 1.5
        else:
            volatility_filter = True
        
        # Generate long signals
        long_condition = (
            strong_momentum & 
            quality_filter & 
            trend_filter & 
            volume_filter & 
            volatility_filter &
            (df['regime'] == 'trending')
        )
        
        # Generate short signals
        short_condition = (
            weak_momentum & 
            trend_filter & 
            volume_filter & 
            volatility_filter &
            (df['regime'] == 'trending')
        )
        
        # Set signals
        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1
        
        # Calculate signal strength
        df['signal_strength'] = self._calculate_signal_strength(df)
        
        # Filter by signal strength
        df.loc[df['signal_strength'] < 0.5, 'signal'] = 0
        
        # Store signals
        self.signals = df[['signal', 'signal_strength', 'regime']].copy()
        
        return df
    
    def _calculate_signal_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate signal strength based on multiple factors"""
        strength = pd.Series(0.0, index=data.index)
        
        # Momentum strength
        momentum_score = abs(data['price_momentum']) / 0.1  # Normalize to 0-1
        momentum_score = momentum_score.clip(0, 1)
        
        # Trend strength (ADX)
        trend_score = data['adx'] / 50  # Normalize to 0-1
        trend_score = trend_score.clip(0, 1)
        
        # Quality score
        quality_score = data['momentum_quality']
        
        # Volume score
        if 'volume' in data.columns:
            volume_ratio = data['volume'] / data['volume'].rolling(window=20).mean()
            volume_score = (volume_ratio - 1).clip(0, 1)
        else:
            volume_score = 0.5
        
        # Combine scores
        strength = (momentum_score + trend_score + quality_score + volume_score) / 4
        
        return strength
    
    def calculate_position_size(self,
                              capital: float,
                              price: float,
                              volatility: float,
                              signal_strength: float = 1.0) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            capital: Available capital
            price: Entry price
            volatility: Current volatility
            signal_strength: Strength of signal (0-1)
            
        Returns:
            Position size (number of shares)
        """
        # Risk amount
        risk_amount = capital * self.risk_per_trade * signal_strength
        
        # Stop loss distance based on volatility
        stop_distance = 2 * volatility * price
        
        # Position size
        position_size = risk_amount / stop_distance
        
        # Apply maximum position limit
        max_position_value = capital / self.max_positions
        max_shares = max_position_value / price
        
        return min(position_size, max_shares)
    
    def execute_signals(self,
                       data: pd.DataFrame,
                       capital: float,
                       current_positions: Optional[List[Position]] = None) -> List[Dict]:
        """
        Execute trading signals with position management
        
        Args:
            data: DataFrame with signals
            capital: Available capital
            current_positions: List of current positions
            
        Returns:
            List of trade orders
        """
        if current_positions:
            self.positions = current_positions
        
        orders = []
        latest = data.iloc[-1]
        
        # Check if we have a signal
        if latest['signal'] == 0:
            return orders
        
        # Check position limits
        if len(self.positions) >= self.max_positions:
            return orders
        
        # Calculate position size
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std().iloc[-1]
        
        size = self.calculate_position_size(
            capital=capital,
            price=latest['close'],
            volatility=volatility,
            signal_strength=latest['signal_strength']
        )
        
        # Create order
        if latest['signal'] == 1:  # Long signal
            order = {
                'symbol': 'SYMBOL',  # Would be passed as parameter
                'side': 'buy',
                'size': size,
                'order_type': 'market',
                'stop_loss': latest['close'] * (1 - 2 * volatility),
                'take_profit': latest['close'] * (1 + 3 * volatility),
                'signal_strength': latest['signal_strength']
            }
            orders.append(order)
            
        elif latest['signal'] == -1:  # Short signal
            order = {
                'symbol': 'SYMBOL',
                'side': 'sell',
                'size': size,
                'order_type': 'market',
                'stop_loss': latest['close'] * (1 + 2 * volatility),
                'take_profit': latest['close'] * (1 - 3 * volatility),
                'signal_strength': latest['signal_strength']
            }
            orders.append(order)
        
        return orders
    
    def manage_positions(self,
                        data: pd.DataFrame,
                        current_positions: List[Position]) -> List[Dict]:
        """
        Manage existing positions (stop loss, take profit, trailing stop)
        
        Args:
            data: Current market data
            current_positions: List of open positions
            
        Returns:
            List of orders to close/modify positions
        """
        orders = []
        latest = data.iloc[-1]
        
        for position in current_positions:
            # Check stop loss
            if position.stop_loss:
                if position.side == 'long' and latest['close'] <= position.stop_loss:
                    orders.append({
                        'symbol': position.symbol,
                        'side': 'sell',
                        'size': position.size,
                        'order_type': 'market',
                        'reason': 'stop_loss'
                    })
                elif position.side == 'short' and latest['close'] >= position.stop_loss:
                    orders.append({
                        'symbol': position.symbol,
                        'side': 'buy',
                        'size': position.size,
                        'order_type': 'market',
                        'reason': 'stop_loss'
                    })
            
            # Check take profit
            if position.take_profit:
                if position.side == 'long' and latest['close'] >= position.take_profit:
                    orders.append({
                        'symbol': position.symbol,
                        'side': 'sell',
                        'size': position.size,
                        'order_type': 'market',
                        'reason': 'take_profit'
                    })
                elif position.side == 'short' and latest['close'] <= position.take_profit:
                    orders.append({
                        'symbol': position.symbol,
                        'side': 'buy',
                        'size': position.size,
                        'order_type': 'market',
                        'reason': 'take_profit'
                    })
            
            # Trailing stop logic
            position_return = position.current_return(latest['close'])
            if position_return > 0.05:  # 5% profit
                # Calculate new trailing stop
                returns = data['close'].pct_change()
                volatility = returns.rolling(window=20).std().iloc[-1]
                
                if position.side == 'long':
                    new_stop = latest['close'] * (1 - volatility)
                    if position.stop_loss is None or new_stop > position.stop_loss:
                        position.stop_loss = new_stop
                else:  # short
                    new_stop = latest['close'] * (1 + volatility)
                    if position.stop_loss is None or new_stop < position.stop_loss:
                        position.stop_loss = new_stop
            
            # Exit on momentum reversal
            if latest['signal'] != 0 and latest['signal'] * (1 if position.side == 'long' else -1) < 0:
                orders.append({
                    'symbol': position.symbol,
                    'side': 'sell' if position.side == 'long' else 'buy',
                    'size': position.size,
                    'order_type': 'market',
                    'reason': 'momentum_reversal'
                })
        
        return orders
    
    def backtest(self,
                data: pd.DataFrame,
                initial_capital: float = 100000,
                commission: float = 0.001) -> Dict:
        """
        Backtest the momentum strategy
        
        Args:
            data: Historical data with signals
            initial_capital: Starting capital
            commission: Commission per trade
            
        Returns:
            Backtest results
        """
        # Generate signals if not already done
        if self.signals.empty:
            data = self.generate_signals(data)
        
        # Initialize backtest variables
        capital = initial_capital
        positions = []
        trades = []
        portfolio_value = []
        
        # Iterate through data
        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]
            current_date = data.index[i]
            
            # Update portfolio value
            position_value = sum(pos.current_value(current_price) for pos in positions)
            total_value = capital + position_value
            portfolio_value.append({
                'date': current_date,
                'capital': capital,
                'position_value': position_value,
                'total_value': total_value
            })
            
            # Check for new signals
            if data['signal'].iloc[i] != 0 and len(positions) < self.max_positions:
                signal = data['signal'].iloc[i]
                
                # Calculate position size
                returns = data['close'].pct_change()
                volatility = returns.rolling(window=20).std().iloc[i]
                size = self.calculate_position_size(capital, current_price, volatility)
                
                # Execute trade
                if signal == 1:  # Long
                    cost = size * current_price * (1 + commission)
                    if cost <= capital:
                        position = Position(
                            symbol='BACKTEST',
                            entry_price=current_price,
                            entry_date=current_date,
                            size=size,
                            side='long',
                            stop_loss=current_price * (1 - 2 * volatility),
                            take_profit=current_price * (1 + 3 * volatility)
                        )
                        positions.append(position)
                        capital -= cost
                        trades.append({
                            'date': current_date,
                            'side': 'buy',
                            'price': current_price,
                            'size': size,
                            'cost': cost
                        })
                
                elif signal == -1:  # Short
                    # For simplicity, assume we can short
                    cost = size * current_price * commission
                    if cost <= capital:
                        position = Position(
                            symbol='BACKTEST',
                            entry_price=current_price,
                            entry_date=current_date,
                            size=size,
                            side='short',
                            stop_loss=current_price * (1 + 2 * volatility),
                            take_profit=current_price * (1 - 3 * volatility)
                        )
                        positions.append(position)
                        capital -= cost
                        trades.append({
                            'date': current_date,
                            'side': 'sell',
                            'price': current_price,
                            'size': size,
                            'cost': cost
                        })
            
            # Manage existing positions
            positions_to_close = []
            for j, position in enumerate(positions):
                close_position = False
                reason = ''
                
                # Check stop loss
                if position.stop_loss:
                    if position.side == 'long' and current_price <= position.stop_loss:
                        close_position = True
                        reason = 'stop_loss'
                    elif position.side == 'short' and current_price >= position.stop_loss:
                        close_position = True
                        reason = 'stop_loss'
                
                # Check take profit
                if not close_position and position.take_profit:
                    if position.side == 'long' and current_price >= position.take_profit:
                        close_position = True
                        reason = 'take_profit'
                    elif position.side == 'short' and current_price <= position.take_profit:
                        close_position = True
                        reason = 'take_profit'
                
                # Close position
                if close_position:
                    profit = position.current_value(current_price)
                    proceeds = position.size * current_price * (1 - commission)
                    capital += proceeds if position.side == 'long' else (2 * position.size * position.entry_price - proceeds)
                    
                    trades.append({
                        'date': current_date,
                        'side': 'sell' if position.side == 'long' else 'buy',
                        'price': current_price,
                        'size': position.size,
                        'profit': profit,
                        'reason': reason
                    })
                    
                    positions_to_close.append(j)
            
            # Remove closed positions
            for j in sorted(positions_to_close, reverse=True):
                positions.pop(j)
        
        # Calculate performance metrics
        portfolio_df = pd.DataFrame(portfolio_value)
        returns = portfolio_df['total_value'].pct_change().dropna()
        
        results = {
            'total_return': (portfolio_df['total_value'].iloc[-1] - initial_capital) / initial_capital,
            'annualized_return': (1 + (portfolio_df['total_value'].iloc[-1] - initial_capital) / initial_capital) ** (252 / len(data)) - 1,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_df['total_value']),
            'win_rate': self._calculate_win_rate(trades),
            'total_trades': len([t for t in trades if 'profit' in t]),
            'portfolio_history': portfolio_df,
            'trades': trades
        }
        
        return results
    
    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + values.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades"""
        completed_trades = [t for t in trades if 'profit' in t]
        if not completed_trades:
            return 0.0
        
        wins = sum(1 for t in completed_trades if t['profit'] > 0)
        return wins / len(completed_trades)