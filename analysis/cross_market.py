"""
Cross-Market Analysis Module
Provides cross-market correlation analysis and arbitrage opportunities
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta
import scipy.stats as stats
from config import config

class CrossMarketAnalysis:
    """Cross-market analysis and correlation tools"""
    
    def __init__(self):
        self.crypto_symbols = config.DATA_CONFIG["crypto_symbols"]
        self.stock_symbols = config.DATA_CONFIG["stock_symbols"]
        self.correlation_threshold = 0.7
        self.arbitrage_threshold = 0.05
    
    def calculate_correlations(self, data):
        """
        Calculate correlations between different assets
        
        Args:
            data: Dictionary with asset data
            
        Returns:
            correlations: Correlation analysis results
        """
        try:
            # Prepare price data
            price_data = {}
            
            for symbol, df in data.items():
                if df is not None and not df.empty and 'close' in df.columns:
                    price_data[symbol] = df['close'].pct_change().dropna()
            
            if len(price_data) < 2:
                return {}
            
            # Align data by common dates
            aligned_data = pd.DataFrame(price_data)
            aligned_data = aligned_data.dropna()
            
            if aligned_data.empty:
                return {}
            
            # Calculate correlation matrix
            correlation_matrix = aligned_data.corr()
            
            # Separate crypto and stock correlations
            crypto_data = {k: v for k, v in price_data.items() if k in self.crypto_symbols}
            stock_data = {k: v for k, v in price_data.items() if k in self.stock_symbols}
            
            # Cross-asset correlations
            cross_correlations = []
            
            for crypto_symbol in crypto_data.keys():
                for stock_symbol in stock_data.keys():
                    if crypto_symbol in aligned_data.columns and stock_symbol in aligned_data.columns:
                        corr = correlation_matrix.loc[crypto_symbol, stock_symbol]
                        
                        if not np.isnan(corr):
                            cross_correlations.append({
                                'crypto_asset': crypto_symbol,
                                'stock_asset': stock_symbol,
                                'correlation': corr,
                                'significance': 'High' if abs(corr) > self.correlation_threshold else 'Low'
                            })
            
            # High correlations within asset classes
            high_correlations = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    symbol1 = correlation_matrix.columns[i]
                    symbol2 = correlation_matrix.columns[j]
                    corr = correlation_matrix.iloc[i, j]
                    
                    if not np.isnan(corr) and abs(corr) > self.correlation_threshold:
                        high_correlations.append({
                            'asset1': symbol1,
                            'asset2': symbol2,
                            'correlation': corr,
                            'asset1_type': 'Crypto' if symbol1 in self.crypto_symbols else 'Stock',
                            'asset2_type': 'Crypto' if symbol2 in self.crypto_symbols else 'Stock'
                        })
            
            return {
                'correlation_matrix': correlation_matrix,
                'cross_correlations': cross_correlations,
                'high_correlations': high_correlations,
                'aligned_data': aligned_data
            }
            
        except Exception as e:
            st.error(f"Error calculating correlations: {str(e)}")
            return {}
    
    def analyze_volatility_patterns(self, data):
        """
        Analyze volatility patterns across markets
        
        Args:
            data: Dictionary with asset data
            
        Returns:
            volatility_analysis: Volatility analysis results
        """
        try:
            volatility_data = {}
            
            for symbol, df in data.items():
                if df is not None and not df.empty and 'close' in df.columns:
                    returns = df['close'].pct_change().dropna()
                    
                    if len(returns) > 30:
                        # Calculate various volatility measures
                        volatility_data[symbol] = {
                            'daily_volatility': returns.std(),
                            'annualized_volatility': returns.std() * np.sqrt(252),
                            'rolling_volatility_30': returns.rolling(30).std(),
                            'volatility_of_volatility': returns.rolling(30).std().std(),
                            'asset_type': 'Crypto' if symbol in self.crypto_symbols else 'Stock'
                        }
            
            if not volatility_data:
                return {}
            
            # Compare crypto vs stock volatility
            crypto_volatilities = [v['annualized_volatility'] for s, v in volatility_data.items() 
                                 if v['asset_type'] == 'Crypto']
            stock_volatilities = [v['annualized_volatility'] for s, v in volatility_data.items() 
                                if v['asset_type'] == 'Stock']
            
            analysis = {
                'volatility_data': volatility_data,
                'crypto_avg_volatility': np.mean(crypto_volatilities) if crypto_volatilities else 0,
                'stock_avg_volatility': np.mean(stock_volatilities) if stock_volatilities else 0,
                'volatility_ratio': (np.mean(crypto_volatilities) / np.mean(stock_volatilities)) 
                                  if crypto_volatilities and stock_volatilities else 0
            }
            
            return analysis
            
        except Exception as e:
            st.error(f"Error analyzing volatility patterns: {str(e)}")
            return {}
    
    def detect_arbitrage_opportunities(self, data):
        """
        Detect potential arbitrage opportunities
        
        Args:
            data: Dictionary with asset data
            
        Returns:
            arbitrage_opportunities: Detected arbitrage opportunities
        """
        try:
            opportunities = []
            
            # Price momentum arbitrage
            for symbol, df in data.items():
                if df is not None and not df.empty and len(df) > 2:
                    current_price = df['close'].iloc[-1]
                    previous_price = df['close'].iloc[-2]
                    
                    price_change = (current_price - previous_price) / previous_price
                    
                    # Look for significant price movements
                    if abs(price_change) > self.arbitrage_threshold:
                        opportunities.append({
                            'type': 'momentum',
                            'asset': symbol,
                            'current_price': current_price,
                            'price_change': price_change,
                            'signal': 'Strong Buy' if price_change > 0 else 'Strong Sell',
                            'confidence': min(abs(price_change) * 10, 1.0)
                        })
            
            # Cross-asset arbitrage (pairs trading)
            correlation_results = self.calculate_correlations(data)
            
            if 'high_correlations' in correlation_results:
                for corr_pair in correlation_results['high_correlations']:
                    asset1 = corr_pair['asset1']
                    asset2 = corr_pair['asset2']
                    
                    if asset1 in data and asset2 in data:
                        df1 = data[asset1]
                        df2 = data[asset2]
                        
                        if (df1 is not None and not df1.empty and 
                            df2 is not None and not df2.empty and
                            len(df1) > 1 and len(df2) > 1):
                            
                            # Calculate price ratio
                            price1 = df1['close'].iloc[-1]
                            price2 = df2['close'].iloc[-1]
                            
                            # Calculate historical average ratio
                            min_len = min(len(df1), len(df2))
                            recent_data1 = df1['close'].tail(min_len)
                            recent_data2 = df2['close'].tail(min_len)
                            
                            ratios = recent_data1 / recent_data2
                            avg_ratio = ratios.mean()
                            current_ratio = price1 / price2
                            
                            # Check for ratio deviation
                            ratio_deviation = abs(current_ratio - avg_ratio) / avg_ratio
                            
                            if ratio_deviation > 0.1:  # 10% deviation threshold
                                opportunities.append({
                                    'type': 'pairs_trading',
                                    'asset1': asset1,
                                    'asset2': asset2,
                                    'current_ratio': current_ratio,
                                    'average_ratio': avg_ratio,
                                    'deviation': ratio_deviation,
                                    'signal': 'Long ' + asset1 + ', Short ' + asset2 if current_ratio < avg_ratio 
                                            else 'Short ' + asset1 + ', Long ' + asset2,
                                    'confidence': min(ratio_deviation * 5, 1.0)
                                })
            
            return opportunities
            
        except Exception as e:
            st.error(f"Error detecting arbitrage opportunities: {str(e)}")
            return []
    
    def analyze_market_sentiment(self, data):
        """
        Analyze cross-market sentiment
        
        Args:
            data: Dictionary with asset data
            
        Returns:
            sentiment_analysis: Market sentiment analysis
        """
        try:
            sentiment_data = {}
            
            for symbol, df in data.items():
                if df is not None and not df.empty and len(df) > 10:
                    # Calculate sentiment indicators
                    returns = df['close'].pct_change().dropna()
                    
                    if len(returns) > 5:
                        # Recent performance
                        recent_return = returns.tail(5).mean()
                        
                        # Momentum
                        momentum = returns.tail(10).mean()
                        
                        # Volatility
                        volatility = returns.tail(20).std()
                        
                        # Sentiment score
                        sentiment_score = (recent_return * 0.4 + momentum * 0.4 - volatility * 0.2)
                        
                        sentiment_data[symbol] = {
                            'sentiment_score': sentiment_score,
                            'recent_return': recent_return,
                            'momentum': momentum,
                            'volatility': volatility,
                            'sentiment_label': self._classify_sentiment(sentiment_score),
                            'asset_type': 'Crypto' if symbol in self.crypto_symbols else 'Stock'
                        }
            
            if not sentiment_data:
                return {}
            
            # Aggregate sentiment by asset type
            crypto_sentiment = [s['sentiment_score'] for s, d in sentiment_data.items() 
                              if d['asset_type'] == 'Crypto']
            stock_sentiment = [s['sentiment_score'] for s, d in sentiment_data.items() 
                             if d['asset_type'] == 'Stock']
            
            analysis = {
                'individual_sentiment': sentiment_data,
                'crypto_avg_sentiment': np.mean(crypto_sentiment) if crypto_sentiment else 0,
                'stock_avg_sentiment': np.mean(stock_sentiment) if stock_sentiment else 0,
                'market_divergence': abs(np.mean(crypto_sentiment) - np.mean(stock_sentiment)) 
                                   if crypto_sentiment and stock_sentiment else 0
            }
            
            return analysis
            
        except Exception as e:
            st.error(f"Error analyzing market sentiment: {str(e)}")
            return {}
    
    def _classify_sentiment(self, score):
        """Classify sentiment score into categories"""
        if score > 0.02:
            return 'Very Bullish'
        elif score > 0.01:
            return 'Bullish'
        elif score > -0.01:
            return 'Neutral'
        elif score > -0.02:
            return 'Bearish'
        else:
            return 'Very Bearish'
    
    def run_cross_market_analysis(self, data):
        """
        Run comprehensive cross-market analysis
        
        Args:
            data: Dictionary with asset data
            
        Returns:
            analysis_results: Complete analysis results
        """
        try:
            # Run all analysis components
            correlation_analysis = self.calculate_correlations(data)
            volatility_analysis = self.analyze_volatility_patterns(data)
            arbitrage_opportunities = self.detect_arbitrage_opportunities(data)
            sentiment_analysis = self.analyze_market_sentiment(data)
            
            # Market overview
            market_overview = {
                'total_assets': len(data),
                'crypto_assets': len([s for s in data.keys() if s in self.crypto_symbols]),
                'stock_assets': len([s for s in data.keys() if s in self.stock_symbols]),
                'high_correlations': len(correlation_analysis.get('high_correlations', [])),
                'arbitrage_opportunities': len(arbitrage_opportunities),
                'analysis_timestamp': datetime.now()
            }
            
            return {
                'market_overview': market_overview,
                'correlations': correlation_analysis,
                'volatility': volatility_analysis,
                'arbitrage': arbitrage_opportunities,
                'sentiment': sentiment_analysis
            }
            
        except Exception as e:
            st.error(f"Error running cross-market analysis: {str(e)}")
            return {}
    
    def create_correlation_heatmap(self, correlation_matrix):
        """Create correlation heatmap"""
        try:
            if correlation_matrix.empty:
                return None
            
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="Cross-Market Correlation Matrix",
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0
            )
            
            fig.update_layout(
                height=500,
                xaxis_title="Assets",
                yaxis_title="Assets"
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {str(e)}")
            return None
    
    def create_volatility_comparison(self, volatility_data):
        """Create volatility comparison chart"""
        try:
            if not volatility_data:
                return None
            
            # Prepare data
            symbols = list(volatility_data.keys())
            volatilities = [v['annualized_volatility'] for v in volatility_data.values()]
            asset_types = [v['asset_type'] for v in volatility_data.values()]
            
            # Create chart
            fig = go.Figure()
            
            # Color by asset type
            colors = ['#1f77b4' if t == 'Crypto' else '#ff7f0e' for t in asset_types]
            
            fig.add_trace(go.Bar(
                x=symbols,
                y=volatilities,
                marker_color=colors,
                text=[f'{v:.1%}' for v in volatilities],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Volatility: %{y:.1%}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Annualized Volatility Comparison',
                xaxis_title='Assets',
                yaxis_title='Volatility',
                xaxis_tickangle=-45,
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating volatility comparison: {str(e)}")
            return None
    
    def create_sentiment_dashboard(self, sentiment_data):
        """Create sentiment analysis dashboard"""
        try:
            if not sentiment_data:
                return None
            
            # Prepare data
            symbols = list(sentiment_data.keys())
            sentiments = [s['sentiment_score'] for s in sentiment_data.values()]
            labels = [s['sentiment_label'] for s in sentiment_data.values()]
            asset_types = [s['asset_type'] for s in sentiment_data.values()]
            
            # Create sentiment chart
            fig = go.Figure()
            
            # Color by sentiment
            color_map = {
                'Very Bullish': '#00cc00',
                'Bullish': '#66ff66',
                'Neutral': '#ffff00',
                'Bearish': '#ff6666',
                'Very Bearish': '#cc0000'
            }
            
            colors = [color_map.get(label, '#gray') for label in labels]
            
            fig.add_trace(go.Bar(
                x=symbols,
                y=sentiments,
                marker_color=colors,
                text=labels,
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Sentiment: %{text}<br>Score: %{y:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Cross-Market Sentiment Analysis',
                xaxis_title='Assets',
                yaxis_title='Sentiment Score',
                xaxis_tickangle=-45,
                height=400
            )
            
            # Add neutral line
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating sentiment dashboard: {str(e)}")
            return None
    
    def display_analysis_dashboard(self, analysis_results):
        """Display comprehensive cross-market analysis dashboard"""
        try:
            if not analysis_results:
                st.error("No analysis results available")
                return
            
            # Market overview
            st.subheader("🌐 Market Overview")
            
            overview = analysis_results['market_overview']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Assets", overview['total_assets'])
            
            with col2:
                st.metric("Crypto Assets", overview['crypto_assets'])
            
            with col3:
                st.metric("Stock Assets", overview['stock_assets'])
            
            with col4:
                st.metric("Arbitrage Opportunities", overview['arbitrage_opportunities'])
            
            # Correlation Analysis
            st.subheader("🔗 Correlation Analysis")
            
            correlations = analysis_results.get('correlations', {})
            
            if 'correlation_matrix' in correlations and not correlations['correlation_matrix'].empty:
                # Correlation heatmap
                heatmap = self.create_correlation_heatmap(correlations['correlation_matrix'])
                if heatmap:
                    st.plotly_chart(heatmap, use_container_width=True)
                
                # High correlations table
                if 'high_correlations' in correlations:
                    st.subheader("High Correlations (>70%)")
                    
                    high_corr_data = []
                    for corr in correlations['high_correlations']:
                        high_corr_data.append({
                            'Asset 1': corr['asset1'],
                            'Asset 2': corr['asset2'],
                            'Correlation': f"{corr['correlation']:.3f}",
                            'Asset 1 Type': corr['asset1_type'],
                            'Asset 2 Type': corr['asset2_type']
                        })
                    
                    if high_corr_data:
                        st.dataframe(pd.DataFrame(high_corr_data), use_container_width=True)
                    else:
                        st.info("No high correlations found")
            
            # Cross-asset correlations
            if 'cross_correlations' in correlations:
                st.subheader("📊 Crypto-Stock Correlations")
                
                cross_corr_data = []
                for corr in correlations['cross_correlations']:
                    cross_corr_data.append({
                        'Crypto Asset': corr['crypto_asset'],
                        'Stock Asset': corr['stock_asset'],
                        'Correlation': f"{corr['correlation']:.3f}",
                        'Significance': corr['significance']
                    })
                
                if cross_corr_data:
                    st.dataframe(pd.DataFrame(cross_corr_data), use_container_width=True)
                else:
                    st.info("No cross-asset correlations calculated")
            
            # Volatility Analysis
            st.subheader("📈 Volatility Analysis")
            
            volatility = analysis_results.get('volatility', {})
            
            if 'volatility_data' in volatility:
                # Volatility comparison chart
                vol_chart = self.create_volatility_comparison(volatility['volatility_data'])
                if vol_chart:
                    st.plotly_chart(vol_chart, use_container_width=True)
                
                # Volatility metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avg Crypto Volatility", f"{volatility.get('crypto_avg_volatility', 0):.1%}")
                
                with col2:
                    st.metric("Avg Stock Volatility", f"{volatility.get('stock_avg_volatility', 0):.1%}")
                
                with col3:
                    st.metric("Volatility Ratio", f"{volatility.get('volatility_ratio', 0):.2f}")
            
            # Arbitrage Opportunities
            st.subheader("💰 Arbitrage Opportunities")
            
            arbitrage = analysis_results.get('arbitrage', [])
            
            if arbitrage:
                arb_data = []
                for opp in arbitrage:
                    if opp['type'] == 'momentum':
                        arb_data.append({
                            'Type': 'Momentum',
                            'Asset': opp['asset'],
                            'Signal': opp['signal'],
                            'Price Change': f"{opp['price_change']:.2%}",
                            'Confidence': f"{opp['confidence']:.2f}"
                        })
                    elif opp['type'] == 'pairs_trading':
                        arb_data.append({
                            'Type': 'Pairs Trading',
                            'Asset': f"{opp['asset1']} / {opp['asset2']}",
                            'Signal': opp['signal'],
                            'Deviation': f"{opp['deviation']:.2%}",
                            'Confidence': f"{opp['confidence']:.2f}"
                        })
                
                st.dataframe(pd.DataFrame(arb_data), use_container_width=True)
            else:
                st.info("No arbitrage opportunities detected")
            
            # Sentiment Analysis
            st.subheader("💭 Market Sentiment")
            
            sentiment = analysis_results.get('sentiment', {})
            
            if 'individual_sentiment' in sentiment:
                # Sentiment chart
                sentiment_chart = self.create_sentiment_dashboard(sentiment['individual_sentiment'])
                if sentiment_chart:
                    st.plotly_chart(sentiment_chart, use_container_width=True)
                
                # Sentiment metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    crypto_sentiment = sentiment.get('crypto_avg_sentiment', 0)
                    st.metric("Crypto Sentiment", f"{crypto_sentiment:.4f}")
                
                with col2:
                    stock_sentiment = sentiment.get('stock_avg_sentiment', 0)
                    st.metric("Stock Sentiment", f"{stock_sentiment:.4f}")
                
                with col3:
                    divergence = sentiment.get('market_divergence', 0)
                    st.metric("Market Divergence", f"{divergence:.4f}")
            
            # Analysis timestamp
            st.info(f"Analysis completed at: {overview['analysis_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            st.error(f"Error displaying analysis dashboard: {str(e)}")
