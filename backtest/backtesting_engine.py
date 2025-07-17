"""
Backtesting Engine
Custom backtesting engine with user-defined metrics and comprehensive analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from config import config

class BacktestingEngine:
    """Custom backtesting engine for trading strategies"""
    
    def __init__(self, initial_capital=10000, commission=0.001, slippage=0.001):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital
            commission: Commission per trade
            slippage: Slippage per trade
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.reset()
    
    def reset(self):
        """Reset backtest state"""
        self.capital = self.initial_capital
        self.position = 0
        self.position_value = 0
        self.trades = []
        self.portfolio_history = []
        self.current_price = 0
        self.entry_price = 0
        self.entry_date = None
        
    def run_backtest(self, model, data, symbol):
        """
        Run backtest on historical data
        
        Args:
            model: Trained trading model
            data: Historical price data
            symbol: Asset symbol
            
        Returns:
            results: Backtest results
        """
        try:
            self.reset()
            
            if len(data) < 30:
                raise ValueError("Not enough data for backtesting")
            
            # Iterate through historical data
            for i, (date, row) in enumerate(data.iterrows()):
                self.current_price = row['close']
                
                # Get signal from model
                try:
                    # Use data up to current point for prediction
                    historical_data = data.iloc[:i+1]
                    
                    if len(historical_data) >= 30:  # Minimum data for prediction
                        signal, confidence = model.get_signal(historical_data)
                        
                        # Execute trade based on signal
                        self._execute_trade(signal, confidence, date, row)
                    
                except Exception as e:
                    # Continue if model fails for this point
                    continue
                
                # Update portfolio value
                portfolio_value = self.capital + (self.position * self.current_price)
                self.portfolio_history.append({
                    'date': date,
                    'price': self.current_price,
                    'capital': self.capital,
                    'position': self.position,
                    'position_value': self.position * self.current_price,
                    'portfolio_value': portfolio_value
                })
            
            # Close any open position at the end
            if self.position != 0:
                self._close_position(data.index[-1], data.iloc[-1])
            
            return {
                'trades': self.trades,
                'portfolio_history': self.portfolio_history,
                'initial_capital': self.initial_capital,
                'final_capital': self.capital + (self.position * self.current_price),
                'symbol': symbol
            }
            
        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")
            return None
    
    def _execute_trade(self, signal, confidence, date, row):
        """Execute trade based on signal"""
        try:
            # Minimum confidence threshold
            if confidence < 0.5:
                return
            
            # Buy signal
            if signal == 'buy' and self.position == 0:
                # Calculate position size based on available capital
                position_size = (self.capital * 0.95) / self.current_price  # 95% of capital
                
                if position_size > 0:
                    # Account for commission and slippage
                    total_cost = position_size * self.current_price * (1 + self.commission + self.slippage)
                    
                    if total_cost <= self.capital:
                        self.position = position_size
                        self.capital -= total_cost
                        self.entry_price = self.current_price
                        self.entry_date = date
                        
                        # Record trade
                        self.trades.append({
                            'date': date,
                            'action': 'buy',
                            'price': self.current_price,
                            'quantity': position_size,
                            'value': total_cost,
                            'signal_confidence': confidence,
                            'capital_after': self.capital
                        })
            
            # Sell signal
            elif signal == 'sell' and self.position > 0:
                self._close_position(date, row, confidence)
                
        except Exception as e:
            print(f"Error executing trade: {str(e)}")
    
    def _close_position(self, date, row, confidence=1.0):
        """Close current position"""
        try:
            if self.position > 0:
                # Calculate proceeds after commission and slippage
                proceeds = self.position * self.current_price * (1 - self.commission - self.slippage)
                
                # Calculate P&L
                pnl = proceeds - (self.position * self.entry_price)
                pnl_percentage = (pnl / (self.position * self.entry_price)) * 100
                
                # Update capital
                self.capital += proceeds
                
                # Record trade
                self.trades.append({
                    'date': date,
                    'action': 'sell',
                    'price': self.current_price,
                    'quantity': self.position,
                    'value': proceeds,
                    'signal_confidence': confidence,
                    'capital_after': self.capital,
                    'pnl': pnl,
                    'pnl_percentage': pnl_percentage,
                    'entry_price': self.entry_price,
                    'entry_date': self.entry_date,
                    'hold_days': (date - self.entry_date).days if self.entry_date else 0
                })
                
                # Reset position
                self.position = 0
                self.entry_price = 0
                self.entry_date = None
                
        except Exception as e:
            print(f"Error closing position: {str(e)}")
    
    def run_strategy_backtest(self, strategy_func, data, params=None):
        """
        Run backtest with custom strategy function
        
        Args:
            strategy_func: Custom strategy function
            data: Historical data
            params: Strategy parameters
            
        Returns:
            results: Backtest results
        """
        try:
            self.reset()
            
            # Iterate through data
            for i, (date, row) in enumerate(data.iterrows()):
                self.current_price = row['close']
                
                # Get signal from strategy
                historical_data = data.iloc[:i+1]
                signal = strategy_func(historical_data, params)
                
                # Execute trade
                self._execute_trade(signal, 1.0, date, row)
                
                # Update portfolio
                portfolio_value = self.capital + (self.position * self.current_price)
                self.portfolio_history.append({
                    'date': date,
                    'price': self.current_price,
                    'capital': self.capital,
                    'position': self.position,
                    'portfolio_value': portfolio_value
                })
            
            # Close final position
            if self.position != 0:
                self._close_position(data.index[-1], data.iloc[-1])
            
            return {
                'trades': self.trades,
                'portfolio_history': self.portfolio_history,
                'initial_capital': self.initial_capital,
                'final_capital': self.capital + (self.position * self.current_price)
            }
            
        except Exception as e:
            st.error(f"Error running strategy backtest: {str(e)}")
            return None
    
    def optimize_parameters(self, model, data, param_grid):
        """
        Optimize model parameters using grid search
        
        Args:
            model: Model to optimize
            data: Historical data
            param_grid: Parameter grid to search
            
        Returns:
            best_params: Best parameters found
            results: Optimization results
        """
        try:
            best_score = -np.inf
            best_params = None
            results = []
            
            # Generate parameter combinations
            param_combinations = []
            keys = list(param_grid.keys())
            
            def generate_combinations(params, index=0):
                if index == len(keys):
                    param_combinations.append(params.copy())
                    return
                
                key = keys[index]
                for value in param_grid[key]:
                    params[key] = value
                    generate_combinations(params, index + 1)
            
            generate_combinations({})
            
            # Test each combination
            for params in param_combinations:
                try:
                    # Update model parameters (if applicable)
                    if hasattr(model, 'update_parameters'):
                        model.update_parameters(params)
                    
                    # Run backtest
                    result = self.run_backtest(model, data, 'optimization')
                    
                    if result:
                        # Calculate score (total return)
                        score = (result['final_capital'] / result['initial_capital']) - 1
                        
                        results.append({
                            'parameters': params.copy(),
                            'score': score,
                            'final_capital': result['final_capital'],
                            'num_trades': len(result['trades'])
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = params.copy()
                
                except Exception as e:
                    print(f"Error testing parameters {params}: {str(e)}")
                    continue
            
            return best_params, results
            
        except Exception as e:
            st.error(f"Error optimizing parameters: {str(e)}")
            return None, []
    
    def monte_carlo_simulation(self, model, data, num_simulations=1000):
        """
        Run Monte Carlo simulation
        
        Args:
            model: Trading model
            data: Historical data
            num_simulations: Number of simulations
            
        Returns:
            simulation_results: Monte Carlo results
        """
        try:
            results = []
            
            for i in range(num_simulations):
                # Add random noise to data
                noisy_data = data.copy()
                noise = np.random.normal(0, 0.01, len(data))
                noisy_data['close'] = noisy_data['close'] * (1 + noise)
                
                # Run backtest
                result = self.run_backtest(model, noisy_data, f'simulation_{i}')
                
                if result:
                    final_return = (result['final_capital'] / result['initial_capital']) - 1
                    results.append({
                        'simulation': i,
                        'final_return': final_return,
                        'final_capital': result['final_capital'],
                        'num_trades': len(result['trades'])
                    })
            
            return results
            
        except Exception as e:
            st.error(f"Error running Monte Carlo simulation: {str(e)}")
            return []
    
    def walk_forward_analysis(self, model, data, train_size=252, test_size=63):
        """
        Perform walk-forward analysis
        
        Args:
            model: Trading model
            data: Historical data
            train_size: Training window size
            test_size: Test window size
            
        Returns:
            wfa_results: Walk-forward analysis results
        """
        try:
            results = []
            
            start_index = train_size
            while start_index + test_size < len(data):
                # Split data
                train_data = data.iloc[start_index - train_size:start_index]
                test_data = data.iloc[start_index:start_index + test_size]
                
                try:
                    # Retrain model
                    model.train(train_data)
                    
                    # Test on out-of-sample data
                    test_result = self.run_backtest(model, test_data, 'walk_forward')
                    
                    if test_result:
                        results.append({
                            'train_start': train_data.index[0],
                            'train_end': train_data.index[-1],
                            'test_start': test_data.index[0],
                            'test_end': test_data.index[-1],
                            'return': (test_result['final_capital'] / test_result['initial_capital']) - 1,
                            'num_trades': len(test_result['trades'])
                        })
                
                except Exception as e:
                    print(f"Error in walk-forward step: {str(e)}")
                
                start_index += test_size
            
            return results
            
        except Exception as e:
            st.error(f"Error in walk-forward analysis: {str(e)}")
            return []
    
    def get_trade_statistics(self, results):
        """
        Calculate detailed trade statistics
        
        Args:
            results: Backtest results
            
        Returns:
            stats: Trade statistics
        """
        try:
            if not results or not results['trades']:
                return {}
            
            trades = results['trades']
            buy_trades = [t for t in trades if t['action'] == 'buy']
            sell_trades = [t for t in trades if t['action'] == 'sell']
            
            if not sell_trades:
                return {}
            
            # Calculate statistics
            returns = [t['pnl_percentage'] for t in sell_trades if 'pnl_percentage' in t]
            
            if not returns:
                return {}
            
            winning_trades = [r for r in returns if r > 0]
            losing_trades = [r for r in returns if r < 0]
            
            stats = {
                'total_trades': len(sell_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(sell_trades) if sell_trades else 0,
                'average_return': np.mean(returns),
                'average_winning_return': np.mean(winning_trades) if winning_trades else 0,
                'average_losing_return': np.mean(losing_trades) if losing_trades else 0,
                'best_trade': max(returns) if returns else 0,
                'worst_trade': min(returns) if returns else 0,
                'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else np.inf,
                'average_hold_days': np.mean([t['hold_days'] for t in sell_trades if 'hold_days' in t])
            }
            
            return stats
            
        except Exception as e:
            print(f"Error calculating trade statistics: {str(e)}")
            return {}
