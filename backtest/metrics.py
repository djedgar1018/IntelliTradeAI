"""
Backtesting Metrics
Comprehensive metrics calculation and visualization for backtest results
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
import scipy.stats as stats

class BacktestMetrics:
    """Class for calculating and visualizing backtest metrics"""
    
    @staticmethod
    def calculate_metrics(results):
        """
        Calculate comprehensive backtest metrics
        
        Args:
            results: Backtest results from BacktestingEngine
            
        Returns:
            metrics: Dictionary with calculated metrics
        """
        try:
            if not results or not results['portfolio_history']:
                return {}
            
            # Convert to DataFrame for easier calculation
            portfolio_df = pd.DataFrame(results['portfolio_history'])
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df.set_index('date', inplace=True)
            
            # Calculate returns
            portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
            portfolio_df['cumulative_return'] = (portfolio_df['portfolio_value'] / results['initial_capital']) - 1
            
            # Basic metrics
            initial_capital = results['initial_capital']
            final_capital = results['final_capital']
            total_return = (final_capital / initial_capital) - 1
            
            # Time metrics
            start_date = portfolio_df.index[0]
            end_date = portfolio_df.index[-1]
            trading_days = len(portfolio_df)
            total_days = (end_date - start_date).days
            
            # Return metrics
            daily_returns = portfolio_df['daily_return'].dropna()
            
            if len(daily_returns) > 1:
                # Annualized return
                annualized_return = (1 + total_return) ** (252 / trading_days) - 1
                
                # Volatility
                volatility = daily_returns.std() * np.sqrt(252)
                
                # Sharpe ratio (assuming risk-free rate of 2%)
                risk_free_rate = 0.02
                sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
                
                # Sortino ratio
                downside_returns = daily_returns[daily_returns < 0]
                downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
                sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
                
                # Maximum drawdown
                rolling_max = portfolio_df['portfolio_value'].expanding().max()
                drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                
                # Calmar ratio
                calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
                
                # VaR (Value at Risk) 95%
                var_95 = daily_returns.quantile(0.05)
                
                # Maximum drawdown duration
                drawdown_duration = BacktestMetrics._calculate_drawdown_duration(portfolio_df['portfolio_value'])
                
            else:
                annualized_return = 0
                volatility = 0
                sharpe_ratio = 0
                sortino_ratio = 0
                max_drawdown = 0
                calmar_ratio = 0
                var_95 = 0
                drawdown_duration = 0
            
            # Trade metrics
            trade_metrics = BacktestMetrics._calculate_trade_metrics(results['trades'])
            
            # Compile all metrics
            metrics = {
                # Basic metrics
                'initial_capital': initial_capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'annualized_return': annualized_return,
                
                # Risk metrics
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_duration': drawdown_duration,
                'var_95': var_95,
                
                # Time metrics
                'start_date': start_date,
                'end_date': end_date,
                'trading_days': trading_days,
                'total_days': total_days,
                
                # Trade metrics
                'total_trades': trade_metrics['total_trades'],
                'win_rate': trade_metrics['win_rate'],
                'profit_factor': trade_metrics['profit_factor'],
                'average_trade_return': trade_metrics['average_trade_return'],
                'best_trade': trade_metrics['best_trade'],
                'worst_trade': trade_metrics['worst_trade'],
                
                # Additional metrics
                'beta': BacktestMetrics._calculate_beta(daily_returns),
                'alpha': BacktestMetrics._calculate_alpha(daily_returns, annualized_return),
                'information_ratio': BacktestMetrics._calculate_information_ratio(daily_returns),
                'skewness': stats.skew(daily_returns) if len(daily_returns) > 2 else 0,
                'kurtosis': stats.kurtosis(daily_returns) if len(daily_returns) > 2 else 0
            }
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    @staticmethod
    def _calculate_drawdown_duration(portfolio_values):
        """Calculate maximum drawdown duration"""
        try:
            rolling_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - rolling_max) / rolling_max
            
            # Find drawdown periods
            in_drawdown = drawdown < 0
            drawdown_periods = []
            start_idx = None
            
            for i, is_down in enumerate(in_drawdown):
                if is_down and start_idx is None:
                    start_idx = i
                elif not is_down and start_idx is not None:
                    duration = i - start_idx
                    drawdown_periods.append(duration)
                    start_idx = None
            
            # Handle case where drawdown continues to the end
            if start_idx is not None:
                duration = len(portfolio_values) - start_idx
                drawdown_periods.append(duration)
            
            return max(drawdown_periods) if drawdown_periods else 0
            
        except Exception:
            return 0
    
    @staticmethod
    def _calculate_trade_metrics(trades):
        """Calculate trade-specific metrics"""
        try:
            if not trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'average_trade_return': 0,
                    'best_trade': 0,
                    'worst_trade': 0
                }
            
            # Filter completed trades (sell orders)
            completed_trades = [t for t in trades if t['action'] == 'sell' and 'pnl_percentage' in t]
            
            if not completed_trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'average_trade_return': 0,
                    'best_trade': 0,
                    'worst_trade': 0
                }
            
            returns = [t['pnl_percentage'] for t in completed_trades]
            winning_trades = [r for r in returns if r > 0]
            losing_trades = [r for r in returns if r < 0]
            
            total_trades = len(completed_trades)
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            gross_profit = sum(winning_trades) if winning_trades else 0
            gross_loss = abs(sum(losing_trades)) if losing_trades else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            average_trade_return = np.mean(returns) if returns else 0
            best_trade = max(returns) if returns else 0
            worst_trade = min(returns) if returns else 0
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'average_trade_return': average_trade_return,
                'best_trade': best_trade,
                'worst_trade': worst_trade
            }
            
        except Exception:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'average_trade_return': 0,
                'best_trade': 0,
                'worst_trade': 0
            }
    
    @staticmethod
    def _calculate_beta(returns, market_returns=None):
        """Calculate beta (systematic risk)"""
        try:
            if market_returns is None:
                # Use a simple proxy for market returns
                market_returns = pd.Series(np.random.normal(0.0008, 0.02, len(returns)))
            
            if len(returns) > 10 and len(market_returns) > 10:
                covariance = np.cov(returns, market_returns)[0, 1]
                market_variance = np.var(market_returns)
                beta = covariance / market_variance if market_variance > 0 else 0
                return beta
            
            return 0
            
        except Exception:
            return 0
    
    @staticmethod
    def _calculate_alpha(returns, strategy_return, market_return=0.1, risk_free_rate=0.02):
        """Calculate alpha (excess return)"""
        try:
            beta = BacktestMetrics._calculate_beta(returns)
            alpha = strategy_return - (risk_free_rate + beta * (market_return - risk_free_rate))
            return alpha
            
        except Exception:
            return 0
    
    @staticmethod
    def _calculate_information_ratio(returns):
        """Calculate information ratio"""
        try:
            if len(returns) > 1:
                excess_returns = returns - returns.mean()
                tracking_error = excess_returns.std()
                if tracking_error > 0:
                    return excess_returns.mean() / tracking_error
            
            return 0
            
        except Exception:
            return 0
    
    @staticmethod
    def plot_performance(results):
        """
        Plot performance charts
        
        Args:
            results: Backtest results
        """
        try:
            if not results or not results['portfolio_history']:
                st.error("No data to plot")
                return
            
            # Convert to DataFrame
            portfolio_df = pd.DataFrame(results['portfolio_history'])
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df.set_index('date', inplace=True)
            
            # Calculate metrics for plotting
            portfolio_df['cumulative_return'] = (portfolio_df['portfolio_value'] / results['initial_capital'] - 1) * 100
            portfolio_df['drawdown'] = ((portfolio_df['portfolio_value'] - portfolio_df['portfolio_value'].expanding().max()) / portfolio_df['portfolio_value'].expanding().max()) * 100
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                subplot_titles=('Portfolio Value', 'Cumulative Return (%)', 'Drawdown (%)'),
                vertical_spacing=0.1,
                row_heights=[0.4, 0.3, 0.3]
            )
            
            # Portfolio value
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['portfolio_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Add initial capital line
            fig.add_hline(
                y=results['initial_capital'],
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Capital",
                row=1, col=1
            )
            
            # Cumulative return
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['cumulative_return'],
                    mode='lines',
                    name='Cumulative Return',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
            
            # Drawdown
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['drawdown'],
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='red'),
                    fill='tonexty'
                ),
                row=3, col=1
            )
            
            # Add trade markers
            if results['trades']:
                buy_trades = [t for t in results['trades'] if t['action'] == 'buy']
                sell_trades = [t for t in results['trades'] if t['action'] == 'sell']
                
                if buy_trades:
                    buy_dates = [t['date'] for t in buy_trades]
                    buy_prices = [t['price'] for t in buy_trades]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=buy_dates,
                            y=buy_prices,
                            mode='markers',
                            name='Buy Signals',
                            marker=dict(color='green', size=10, symbol='triangle-up')
                        ),
                        row=1, col=1
                    )
                
                if sell_trades:
                    sell_dates = [t['date'] for t in sell_trades]
                    sell_prices = [t['price'] for t in sell_trades]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=sell_dates,
                            y=sell_prices,
                            mode='markers',
                            name='Sell Signals',
                            marker=dict(color='red', size=10, symbol='triangle-down')
                        ),
                        row=1, col=1
                    )
            
            # Update layout
            fig.update_layout(
                title='Backtest Performance Analysis',
                height=800,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Return (%)", row=2, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error plotting performance: {str(e)}")
    
    @staticmethod
    def get_trades_summary(results):
        """
        Get summary of all trades
        
        Args:
            results: Backtest results
            
        Returns:
            trades_df: DataFrame with trade summary
        """
        try:
            if not results or not results['trades']:
                return pd.DataFrame()
            
            trades = results['trades']
            completed_trades = [t for t in trades if t['action'] == 'sell' and 'pnl' in t]
            
            if not completed_trades:
                return pd.DataFrame()
            
            # Create trades summary
            trades_data = []
            for trade in completed_trades:
                trades_data.append({
                    'Entry Date': trade.get('entry_date', ''),
                    'Exit Date': trade['date'],
                    'Entry Price': trade.get('entry_price', 0),
                    'Exit Price': trade['price'],
                    'Quantity': trade['quantity'],
                    'P&L ($)': trade['pnl'],
                    'P&L (%)': trade['pnl_percentage'],
                    'Hold Days': trade.get('hold_days', 0),
                    'Confidence': trade.get('signal_confidence', 0)
                })
            
            trades_df = pd.DataFrame(trades_data)
            
            # Format columns
            if not trades_df.empty:
                trades_df['Entry Price'] = trades_df['Entry Price'].round(4)
                trades_df['Exit Price'] = trades_df['Exit Price'].round(4)
                trades_df['Quantity'] = trades_df['Quantity'].round(6)
                trades_df['P&L ($)'] = trades_df['P&L ($)'].round(2)
                trades_df['P&L (%)'] = trades_df['P&L (%)'].round(2)
                trades_df['Confidence'] = trades_df['Confidence'].round(3)
            
            return trades_df
            
        except Exception as e:
            st.error(f"Error creating trades summary: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def create_metrics_dashboard(metrics):
        """
        Create comprehensive metrics dashboard
        
        Args:
            metrics: Calculated metrics dictionary
        """
        try:
            if not metrics:
                st.error("No metrics available")
                return
            
            st.subheader("📊 Performance Metrics")
            
            # Key performance indicators
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Return",
                    f"{metrics.get('total_return', 0):.2%}",
                    f"${metrics.get('final_capital', 0) - metrics.get('initial_capital', 0):,.2f}"
                )
            
            with col2:
                st.metric(
                    "Annualized Return",
                    f"{metrics.get('annualized_return', 0):.2%}"
                )
            
            with col3:
                st.metric(
                    "Sharpe Ratio",
                    f"{metrics.get('sharpe_ratio', 0):.2f}"
                )
            
            with col4:
                st.metric(
                    "Max Drawdown",
                    f"{metrics.get('max_drawdown', 0):.2%}"
                )
            
            # Risk metrics
            st.subheader("🛡️ Risk Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Volatility", f"{metrics.get('volatility', 0):.2%}")
            
            with col2:
                st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
            
            with col3:
                st.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.2f}")
            
            with col4:
                st.metric("VaR 95%", f"{metrics.get('var_95', 0):.2%}")
            
            # Trading metrics
            st.subheader("📈 Trading Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", metrics.get('total_trades', 0))
            
            with col2:
                st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2%}")
            
            with col3:
                st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
            
            with col4:
                st.metric("Avg Trade Return", f"{metrics.get('average_trade_return', 0):.2%}")
            
            # Additional metrics
            st.subheader("📋 Additional Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Market Metrics:**")
                st.write(f"• Beta: {metrics.get('beta', 0):.2f}")
                st.write(f"• Alpha: {metrics.get('alpha', 0):.2%}")
                st.write(f"• Information Ratio: {metrics.get('information_ratio', 0):.2f}")
                st.write(f"• Best Trade: {metrics.get('best_trade', 0):.2%}")
                st.write(f"• Worst Trade: {metrics.get('worst_trade', 0):.2%}")
            
            with col2:
                st.write("**Statistical Metrics:**")
                st.write(f"• Skewness: {metrics.get('skewness', 0):.2f}")
                st.write(f"• Kurtosis: {metrics.get('kurtosis', 0):.2f}")
                st.write(f"• Trading Days: {metrics.get('trading_days', 0)}")
                st.write(f"• Max DD Duration: {metrics.get('max_drawdown_duration', 0)} days")
            
        except Exception as e:
            st.error(f"Error creating metrics dashboard: {str(e)}")
