"""
Backtesting Module

This module provides comprehensive backtesting capabilities for trading strategies
with realistic market simulation and performance analytics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import quantstats as qs
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    symbol: str = ""
    side: str = "long"  # 'long' or 'short'
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    quantity: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    exit_reason: Optional[str] = None
    
    def close(self, exit_time: pd.Timestamp, exit_price: float, 
              commission: float = 0.0, slippage: float = 0.0, reason: str = "signal"):
        """Close the trade"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.commission += commission
        self.slippage += slippage
        self.exit_reason = reason
        
        # Calculate P&L
        if self.side == "long":
            gross_pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # short
            gross_pnl = (self.entry_price - self.exit_price) * self.quantity
        
        self.pnl = gross_pnl - self.commission - self.slippage
        self.return_pct = self.pnl / (self.entry_price * self.quantity)


@dataclass
class BacktestResult:
    """Container for backtest results"""
    equity_curve: pd.Series
    trades: List[Trade]
    returns: pd.Series
    positions: pd.DataFrame
    metrics: Dict[str, float]
    signals: pd.DataFrame
    benchmark_returns: Optional[pd.Series] = None


class Backtester:
    """Advanced backtesting engine for trading strategies"""
    
    def __init__(self,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0001,
                 min_commission: float = 1.0,
                 position_size: Union[float, str] = 0.1,
                 max_positions: int = 10,
                 margin_requirement: float = 1.0,
                 short_selling: bool = True):
        """
        Initialize the backtester
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate (percentage)
            slippage: Slippage rate (percentage)
            min_commission: Minimum commission per trade
            position_size: Position sizing method
            max_positions: Maximum concurrent positions
            margin_requirement: Margin requirement for positions
            short_selling: Allow short selling
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.min_commission = min_commission
        self.position_size = position_size
        self.max_positions = max_positions
        self.margin_requirement = margin_requirement
        self.short_selling = short_selling
        
        # State variables
        self.reset()
    
    def reset(self):
        """Reset backtester state"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_history = []
        self.position_history = []
        self.cash_history = []
    
    def run(self,
            data: pd.DataFrame,
            signals: pd.DataFrame,
            benchmark: Optional[pd.Series] = None,
            strategy_params: Optional[Dict] = None) -> BacktestResult:
        """
        Run backtest on historical data
        
        Args:
            data: OHLCV data
            signals: Trading signals (-1, 0, 1)
            benchmark: Benchmark returns for comparison
            strategy_params: Additional strategy parameters
            
        Returns:
            BacktestResult object
        """
        self.reset()
        
        # Ensure data and signals are aligned
        if not data.index.equals(signals.index):
            raise ValueError("Data and signals must have the same index")
        
        # Initialize tracking
        equity_curve = pd.Series(index=data.index, dtype=float)
        position_tracker = pd.DataFrame(index=data.index, columns=['position', 'value'])
        
        # Run backtest
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Current prices
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            volume = row.get('volume', 0)
            
            # Get signal
            signal = signals.iloc[i]
            if isinstance(signal, pd.Series):
                signal = signal.iloc[0]
            
            # Update existing positions
            self._update_positions(close_price, timestamp)
            
            # Process signals
            if signal != 0:
                self._process_signal(
                    signal=signal,
                    price=close_price,
                    timestamp=timestamp,
                    volume=volume,
                    high=high_price,
                    low=low_price
                )
            
            # Record state
            total_value = self._calculate_portfolio_value(close_price)
            equity_curve[timestamp] = total_value
            position_tracker.loc[timestamp, 'position'] = len(self.positions)
            position_tracker.loc[timestamp, 'value'] = total_value - self.capital
            
            self.equity_history.append(total_value)
            self.cash_history.append(self.capital)
        
        # Close all remaining positions
        last_price = data['close'].iloc[-1]
        last_timestamp = data.index[-1]
        self._close_all_positions(last_price, last_timestamp, reason="end_of_backtest")
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Calculate metrics
        metrics = self._calculate_metrics(equity_curve, returns, self.trades)
        
        # Create result
        result = BacktestResult(
            equity_curve=equity_curve,
            trades=self.trades,
            returns=returns,
            positions=position_tracker,
            metrics=metrics,
            signals=signals,
            benchmark_returns=benchmark
        )
        
        return result
    
    def _process_signal(self, 
                       signal: int,
                       price: float,
                       timestamp: pd.Timestamp,
                       volume: float,
                       high: float,
                       low: float):
        """Process trading signal"""
        # Check if we can open new positions
        if len(self.positions) >= self.max_positions:
            return
        
        # Calculate position size
        position_value = self._calculate_position_size(price)
        if position_value > self.capital * self.margin_requirement:
            return
        
        # Calculate costs
        quantity = position_value / price
        commission_cost = max(position_value * self.commission, self.min_commission)
        slippage_cost = position_value * self.slippage
        
        # Check if we have enough capital
        total_cost = position_value + commission_cost + slippage_cost
        if total_cost > self.capital:
            return
        
        # Open position
        if signal == 1:  # Long
            self._open_position(
                timestamp=timestamp,
                side="long",
                price=price,
                quantity=quantity,
                commission=commission_cost,
                slippage=slippage_cost
            )
        elif signal == -1 and self.short_selling:  # Short
            self._open_position(
                timestamp=timestamp,
                side="short",
                price=price,
                quantity=quantity,
                commission=commission_cost,
                slippage=slippage_cost
            )
    
    def _open_position(self,
                      timestamp: pd.Timestamp,
                      side: str,
                      price: float,
                      quantity: float,
                      commission: float,
                      slippage: float):
        """Open a new position"""
        # Create trade
        trade = Trade(
            entry_time=timestamp,
            symbol="",  # Could be passed as parameter
            side=side,
            entry_price=price,
            quantity=quantity,
            commission=commission,
            slippage=slippage
        )
        
        # Update capital
        if side == "long":
            self.capital -= (price * quantity + commission + slippage)
        else:  # short
            # For shorts, we receive the sale proceeds but need margin
            self.capital += price * quantity - commission - slippage
            self.capital -= price * quantity * self.margin_requirement
        
        # Store position
        self.positions[timestamp] = trade
    
    def _update_positions(self, current_price: float, timestamp: pd.Timestamp):
        """Update existing positions (e.g., check stop losses)"""
        positions_to_close = []
        
        for entry_time, trade in self.positions.items():
            # Simple exit logic - can be extended
            # Example: Close if position is old or hit stop loss
            days_held = (timestamp - entry_time).days
            
            # Time-based exit
            if days_held > 30:
                positions_to_close.append(entry_time)
                continue
            
            # Stop loss (example: 5% loss)
            if trade.side == "long":
                current_return = (current_price - trade.entry_price) / trade.entry_price
            else:
                current_return = (trade.entry_price - current_price) / trade.entry_price
            
            if current_return < -0.05:
                positions_to_close.append(entry_time)
        
        # Close positions
        for entry_time in positions_to_close:
            self._close_position(entry_time, current_price, timestamp, reason="stop_loss")
    
    def _close_position(self,
                       entry_time: pd.Timestamp,
                       exit_price: float,
                       exit_time: pd.Timestamp,
                       reason: str = "signal"):
        """Close a position"""
        if entry_time not in self.positions:
            return
        
        trade = self.positions[entry_time]
        
        # Calculate exit costs
        exit_value = exit_price * trade.quantity
        commission_cost = max(exit_value * self.commission, self.min_commission)
        slippage_cost = exit_value * self.slippage
        
        # Close trade
        trade.close(
            exit_time=exit_time,
            exit_price=exit_price,
            commission=commission_cost,
            slippage=slippage_cost,
            reason=reason
        )
        
        # Update capital
        if trade.side == "long":
            self.capital += exit_price * trade.quantity - commission_cost - slippage_cost
        else:  # short
            # Return margin and pay/receive difference
            self.capital += trade.entry_price * trade.quantity * self.margin_requirement
            self.capital -= (exit_price - trade.entry_price) * trade.quantity
            self.capital -= commission_cost + slippage_cost
        
        # Record trade
        self.trades.append(trade)
        del self.positions[entry_time]
    
    def _close_all_positions(self, price: float, timestamp: pd.Timestamp, reason: str):
        """Close all open positions"""
        positions_to_close = list(self.positions.keys())
        for entry_time in positions_to_close:
            self._close_position(entry_time, price, timestamp, reason)
    
    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on strategy"""
        if isinstance(self.position_size, float):
            # Fixed percentage of capital
            return self.capital * self.position_size
        elif self.position_size == "kelly":
            # Kelly criterion (simplified)
            return self._kelly_position_size(price)
        elif self.position_size == "equal":
            # Equal weight
            return self.capital / self.max_positions
        else:
            # Default to fixed percentage
            return self.capital * 0.1
    
    def _kelly_position_size(self, price: float) -> float:
        """Calculate position size using Kelly criterion"""
        if len(self.trades) < 10:
            # Not enough history, use default
            return self.capital * 0.05
        
        # Calculate win rate and average win/loss
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]
        
        if not wins or not losses:
            return self.capital * 0.05
        
        win_rate = len(wins) / len(self.trades)
        avg_win = np.mean([t.return_pct for t in wins])
        avg_loss = abs(np.mean([t.return_pct for t in losses]))
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        if avg_loss == 0:
            return self.capital * 0.05
        
        b = avg_win / avg_loss
        q = 1 - win_rate
        
        kelly_pct = (win_rate * b - q) / b
        
        # Apply Kelly fraction (usually 0.25 to be conservative)
        kelly_pct = max(0, min(kelly_pct * 0.25, 0.25))
        
        return self.capital * kelly_pct
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value"""
        position_value = 0
        
        for trade in self.positions.values():
            if trade.side == "long":
                position_value += current_price * trade.quantity
            else:  # short
                # Short position value = initial value - (current - entry) * quantity
                position_value += trade.entry_price * trade.quantity * 2 - current_price * trade.quantity
        
        return self.capital + position_value
    
    def _calculate_metrics(self, 
                          equity_curve: pd.Series,
                          returns: pd.Series,
                          trades: List[Trade]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital
        metrics['annualized_return'] = self._annualized_return(returns)
        metrics['volatility'] = returns.std() * np.sqrt(252)
        
        # Risk metrics
        metrics['sharpe_ratio'] = self._sharpe_ratio(returns)
        metrics['sortino_ratio'] = self._sortino_ratio(returns)
        metrics['max_drawdown'] = self._max_drawdown(equity_curve)
        metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        # Trade metrics
        if trades:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            metrics['total_trades'] = len(trades)
            metrics['winning_trades'] = len(winning_trades)
            metrics['losing_trades'] = len(losing_trades)
            metrics['win_rate'] = len(winning_trades) / len(trades) if trades else 0
            
            metrics['avg_win'] = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            metrics['avg_loss'] = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            metrics['profit_factor'] = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else 0
            
            metrics['largest_win'] = max([t.pnl for t in trades]) if trades else 0
            metrics['largest_loss'] = min([t.pnl for t in trades]) if trades else 0
            
            # Duration metrics
            durations = [(t.exit_time - t.entry_time).days for t in trades if t.exit_time]
            metrics['avg_trade_duration'] = np.mean(durations) if durations else 0
        
        # Additional metrics
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        metrics['var_95'] = returns.quantile(0.05)
        metrics['cvar_95'] = returns[returns <= returns.quantile(0.05)].mean()
        
        return metrics
    
    def _annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        if len(returns) == 0:
            return 0
        
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        
        if n_years == 0:
            return 0
        
        return (1 + total_return) ** (1 / n_years) - 1
    
    def _sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def _sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return 0
        
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0
        
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    def _max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(equity_curve) == 0:
            return 0
        
        cumulative = equity_curve / equity_curve.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    def plot_results(self, result: BacktestResult, figsize: Tuple[int, int] = (15, 10)):
        """Plot comprehensive backtest results"""
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle('Backtest Results', fontsize=16)
        
        # Equity curve
        ax = axes[0, 0]
        result.equity_curve.plot(ax=ax, label='Strategy')
        if result.benchmark_returns is not None:
            benchmark_equity = (1 + result.benchmark_returns).cumprod() * self.initial_capital
            benchmark_equity.plot(ax=ax, label='Benchmark', alpha=0.7)
        ax.set_title('Equity Curve')
        ax.set_ylabel('Portfolio Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Drawdown
        ax = axes[0, 1]
        cumulative = result.equity_curve / result.equity_curve.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        drawdown.plot(ax=ax, color='red', alpha=0.7)
        ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax.set_title('Drawdown')
        ax.set_ylabel('Drawdown %')
        ax.grid(True, alpha=0.3)
        
        # Returns distribution
        ax = axes[1, 0]
        result.returns.hist(bins=50, ax=ax, alpha=0.7, color='blue')
        ax.axvline(result.returns.mean(), color='red', linestyle='--', label='Mean')
        ax.set_title('Returns Distribution')
        ax.set_xlabel('Daily Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cumulative returns
        ax = axes[1, 1]
        cum_returns = (1 + result.returns).cumprod() - 1
        cum_returns.plot(ax=ax, label='Strategy')
        if result.benchmark_returns is not None:
            cum_benchmark = (1 + result.benchmark_returns).cumprod() - 1
            cum_benchmark.plot(ax=ax, label='Benchmark', alpha=0.7)
        ax.set_title('Cumulative Returns')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Trade P&L
        ax = axes[2, 0]
        if result.trades:
            trade_pnls = [t.pnl for t in result.trades]
            trade_dates = [t.exit_time for t in result.trades if t.exit_time]
            
            if trade_dates and len(trade_dates) == len(trade_pnls):
                colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
                ax.bar(range(len(trade_pnls)), trade_pnls, color=colors, alpha=0.7)
                ax.set_title('Trade P&L')
                ax.set_xlabel('Trade Number')
                ax.set_ylabel('P&L')
            else:
                ax.text(0.5, 0.5, 'No completed trades', ha='center', va='center')
                ax.set_title('Trade P&L')
        ax.grid(True, alpha=0.3)
        
        # Monthly returns heatmap
        ax = axes[2, 1]
        monthly_returns = result.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        if len(monthly_returns) > 0:
            monthly_pivot = pd.DataFrame(monthly_returns)
            monthly_pivot['Year'] = monthly_pivot.index.year
            monthly_pivot['Month'] = monthly_pivot.index.month
            monthly_pivot = monthly_pivot.pivot(index='Year', columns='Month', values=0)
            
            sns.heatmap(monthly_pivot * 100, annot=True, fmt='.1f', 
                       cmap='RdYlGn', center=0, ax=ax,
                       cbar_kws={'label': 'Return %'})
            ax.set_title('Monthly Returns Heatmap')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            ax.set_title('Monthly Returns Heatmap')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, result: BacktestResult) -> str:
        """Generate a text report of backtest results"""
        report = []
        report.append("=" * 60)
        report.append("BACKTEST REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Performance Summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 30)
        report.append(f"Total Return: {result.metrics['total_return']:.2%}")
        report.append(f"Annualized Return: {result.metrics['annualized_return']:.2%}")
        report.append(f"Volatility: {result.metrics['volatility']:.2%}")
        report.append(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
        report.append(f"Sortino Ratio: {result.metrics['sortino_ratio']:.2f}")
        report.append(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
        report.append(f"Calmar Ratio: {result.metrics['calmar_ratio']:.2f}")
        report.append("")
        
        # Trade Statistics
        report.append("TRADE STATISTICS")
        report.append("-" * 30)
        report.append(f"Total Trades: {result.metrics.get('total_trades', 0)}")
        report.append(f"Winning Trades: {result.metrics.get('winning_trades', 0)}")
        report.append(f"Losing Trades: {result.metrics.get('losing_trades', 0)}")
        report.append(f"Win Rate: {result.metrics.get('win_rate', 0):.2%}")
        report.append(f"Average Win: ${result.metrics.get('avg_win', 0):.2f}")
        report.append(f"Average Loss: ${result.metrics.get('avg_loss', 0):.2f}")
        report.append(f"Profit Factor: {result.metrics.get('profit_factor', 0):.2f}")
        report.append(f"Average Trade Duration: {result.metrics.get('avg_trade_duration', 0):.1f} days")
        report.append("")
        
        # Risk Metrics
        report.append("RISK METRICS")
        report.append("-" * 30)
        report.append(f"95% VaR: {result.metrics.get('var_95', 0):.2%}")
        report.append(f"95% CVaR: {result.metrics.get('cvar_95', 0):.2%}")
        report.append(f"Skewness: {result.metrics.get('skewness', 0):.2f}")
        report.append(f"Kurtosis: {result.metrics.get('kurtosis', 0):.2f}")
        
        return "\n".join(report)
    
    def optimize_parameters(self,
                           data: pd.DataFrame,
                           signal_function: Callable,
                           param_grid: Dict[str, List],
                           metric: str = 'sharpe_ratio') -> Dict:
        """
        Optimize strategy parameters
        
        Args:
            data: Historical data
            signal_function: Function that generates signals
            param_grid: Parameter grid for optimization
            metric: Metric to optimize
            
        Returns:
            Best parameters and results
        """
        best_metric = -np.inf
        best_params = None
        best_result = None
        all_results = []
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        from itertools import product
        param_combinations = list(product(*param_values))
        
        # Test each combination
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            
            # Generate signals with current parameters
            signals = signal_function(data, **param_dict)
            
            # Run backtest
            result = self.run(data, signals)
            
            # Track results
            current_metric = result.metrics.get(metric, -np.inf)
            all_results.append({
                'params': param_dict,
                'metric': current_metric,
                'metrics': result.metrics
            })
            
            # Update best
            if current_metric > best_metric:
                best_metric = current_metric
                best_params = param_dict
                best_result = result
        
        return {
            'best_params': best_params,
            'best_metric': best_metric,
            'best_result': best_result,
            'all_results': pd.DataFrame(all_results)
        }