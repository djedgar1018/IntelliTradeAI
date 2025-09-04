"""
AI Analysis Tab - User-friendly intelligent trading recommendations
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

def render_ai_analysis_tab(market_data, ai_advisor, period, interval):
    """Render the AI analysis tab with intelligent recommendations"""
    
    st.markdown("### 🤖 AI-Powered Trading Analysis")
    st.markdown("Let me analyze your assets and provide clear trading recommendations with explanations.")
    
    if not market_data:
        st.info("👆 **First, load some market data** in the 'Overview & Data' tab to get started!")
        st.markdown("---")
        st.markdown("### 📋 What I'll analyze for you:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **📊 Technical Analysis:**
            - Price trends and moving averages
            - Support and resistance levels
            - RSI and momentum indicators
            - Volume analysis
            """)
        with col2:
            st.markdown("""
            **🤖 AI Predictions:**
            - Machine learning price forecasts
            - Confidence levels for each prediction
            - Risk assessment
            - Clear buy/sell/hold recommendations
            """)
        return
    
    # Asset selection
    st.markdown("### 📈 Choose Asset to Analyze")
    asset_options = list(market_data.keys())
    
    if len(asset_options) == 1:
        selected_asset = asset_options[0]
        st.info(f"Analyzing your only loaded asset: **{selected_asset}**")
    else:
        selected_asset = st.selectbox(
            "Which asset would you like me to analyze?",
            asset_options,
            help="I'll provide detailed analysis and recommendations for your selected asset"
        )
    
    if selected_asset and selected_asset in market_data:
        asset_data = market_data[selected_asset]
        
        # Get AI analysis
        with st.spinner(f"🧠 Running AI analysis on {selected_asset}..."):
            analysis = ai_advisor.analyze_asset(selected_asset, asset_data)
        
        # Display recommendation prominently
        rec = analysis['recommendation']
        
        # Create prominent recommendation card
        decision_colors = {
            'BUY': '🟢', 'DCA_IN': '🔵', 'SELL': '🔴', 
            'DCA_OUT': '🟠', 'HOLD': '⚪'
        }
        
        decision_backgrounds = {
            'BUY': '#d4edda', 'DCA_IN': '#cce7ff', 'SELL': '#f8d7da',
            'DCA_OUT': '#fff3cd', 'HOLD': '#f8f9fa'
        }
        
        # Main recommendation card
        st.markdown(f"""
        <div style="background-color: {decision_backgrounds.get(rec['decision'], '#f8f9fa')}; 
                    padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h2 style="margin: 0; text-align: center;">
                {decision_colors.get(rec['decision'], '⚪')} <strong>{rec['decision']}</strong>
            </h2>
            <p style="text-align: center; font-size: 18px; margin: 10px 0;">
                <strong>{rec['action_explanation']}</strong>
            </p>
            <p style="text-align: center; color: #666;">
                Confidence: {rec['confidence_level']} | Risk Level: {rec['risk_level']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        st.markdown("### 📊 Asset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price", 
                f"${analysis['current_price']:,.2f}",
                f"{analysis['price_change_24h']:+.1f}%" if analysis['price_change_24h'] != 0 else None
            )
        
        with col2:
            confidence_color = {
                'Very High': '🟢', 'High': '🟢', 'Medium': '🟡',
                'Low': '🟠', 'Very Low': '🔴'
            }
            st.metric(
                "AI Confidence", 
                rec['confidence_level'],
                confidence_color.get(rec['confidence_level'], '⚪')
            )
        
        with col3:
            risk_color = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
            st.metric(
                "Risk Level", 
                rec['risk_level'],
                risk_color.get(rec['risk_level'], '⚪')
            )
        
        with col4:
            data_quality = "📊 Rich Data" if len(asset_data) > 30 else "📈 Limited Data" if len(asset_data) > 1 else "📍 Current Only"
            st.metric("Data Quality", data_quality)
        
        # Explanation section
        st.markdown("### 💡 Why This Recommendation?")
        st.markdown("Here's my detailed reasoning:")
        
        for i, explanation in enumerate(rec['detailed_explanation'], 1):
            st.markdown(f"**{i}.** {explanation}")
        
        # Action items
        st.markdown("### 📋 What Should You Do?")
        st.markdown("**Specific actions you can take:**")
        
        for i, action in enumerate(rec['suggested_actions'], 1):
            st.markdown(f"✅ **{i}.** {action}")
        
        # Technical details in expandable section
        with st.expander("📈 Technical Analysis Details"):
            tech = analysis.get('technical_analysis', {})
            if tech.get('explanation'):
                st.markdown("**Technical indicators I analyzed:**")
                for explanation in tech['explanation']:
                    st.markdown(f"• {explanation}")
            else:
                st.info("Technical analysis requires more historical data")
        
        # Model details if available
        if analysis.get('model_analysis'):
            with st.expander("🤖 AI Model Details"):
                model = analysis['model_analysis']
                st.markdown(f"**Model Prediction:** {model['signal']}")
                st.markdown(f"**Confidence Level:** {model['confidence']}")
                st.markdown(f"**Explanation:** {model['explanation']}")
        
        # Chart in expandable section as requested
        if len(asset_data) > 1:
            with st.expander("📊 See Chart"):
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                # Price line
                fig.add_trace(go.Scatter(
                    x=asset_data.index,
                    y=asset_data['close'],
                    mode='lines',
                    name=f'{selected_asset} Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Add moving average if enough data
                if len(asset_data) >= 20:
                    ma20 = asset_data['close'].rolling(window=20).mean()
                    fig.add_trace(go.Scatter(
                        x=asset_data.index,
                        y=ma20,
                        mode='lines',
                        name='20-day Moving Average',
                        line=dict(color='orange', width=1)
                    ))
                
                # Add volume if available
                if 'volume' in asset_data.columns and asset_data['volume'].sum() > 0:
                    # Create subplot with secondary y-axis
                    from plotly.subplots import make_subplots
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.05,
                        subplot_titles=('Price', 'Volume'),
                        shared_xaxes=True
                    )
                    
                    # Price chart
                    fig.add_trace(
                        go.Scatter(
                            x=asset_data.index,
                            y=asset_data['close'],
                            mode='lines',
                            name=f'{selected_asset} Price',
                            line=dict(color='#1f77b4', width=2)
                        ), row=1, col=1
                    )
                    
                    # Moving average
                    if len(asset_data) >= 20:
                        ma20 = asset_data['close'].rolling(window=20).mean()
                        fig.add_trace(
                            go.Scatter(
                                x=asset_data.index,
                                y=ma20,
                                mode='lines',
                                name='20-day MA',
                                line=dict(color='orange', width=1)
                            ), row=1, col=1
                        )
                    
                    # Volume chart
                    fig.add_trace(
                        go.Bar(
                            x=asset_data.index,
                            y=asset_data['volume'],
                            name='Volume',
                            marker_color='lightblue',
                            opacity=0.6
                        ), row=2, col=1
                    )
                    
                    fig.update_layout(
                        title=f'{selected_asset} Price & Volume Analysis',
                        height=600,
                        showlegend=True,
                        hovermode='x unified'
                    )
                    
                    fig.update_xaxes(title_text="Date", row=2, col=1)
                    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
                    fig.update_yaxes(title_text="Volume", row=2, col=1)
                    
                else:
                    fig.update_layout(
                        title=f'{selected_asset} Price Analysis',
                        xaxis_title='Date',
                        yaxis_title='Price (USD)',
                        height=400,
                        showlegend=True,
                        hovermode='x unified'
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Chart insights
                price_change = ((asset_data['close'].iloc[-1] / asset_data['close'].iloc[0]) - 1) * 100
                volatility = asset_data['close'].pct_change().std() * 100
                
                st.markdown(f"""
                **📊 Chart Insights:**
                - **Total Change:** {price_change:+.1f}% over the period
                - **Price Volatility:** {volatility:.1f}% (daily standard deviation)
                - **Data Points:** {len(asset_data)} observations
                - **Period:** {asset_data.index[0].date()} to {asset_data.index[-1].date()}
                """)
        
        # Risk warning
        st.markdown("---")
        st.warning("""
        ⚠️ **Important Disclaimer:** 
        This analysis is for educational purposes only. Always do your own research and consider your risk tolerance. 
        Never invest more than you can afford to lose. Past performance doesn't guarantee future results.
        """)
        
        # Analysis timestamp
        st.caption(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save analysis option
        if st.button("💾 Save This Analysis"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{selected_asset}_{timestamp}.txt"
            
            analysis_text = f"""
AI Trading Analysis - {selected_asset}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RECOMMENDATION: {rec['decision']}
Confidence: {rec['confidence_level']} | Risk: {rec['risk_level']}

EXPLANATION:
{rec['action_explanation']}

DETAILED REASONING:
""" + "\n".join([f"{i}. {exp}" for i, exp in enumerate(rec['detailed_explanation'], 1)]) + f"""

SUGGESTED ACTIONS:
""" + "\n".join([f"{i}. {act}" for i, act in enumerate(rec['suggested_actions'], 1)])
            
            st.download_button(
                label="📄 Download Analysis Report",
                data=analysis_text,
                file_name=filename,
                mime="text/plain"
            )
    
    # Multiple asset comparison
    if len(asset_options) > 1:
        st.markdown("---")
        st.markdown("### 🔄 Quick Compare All Assets")
        
        if st.button("🚀 Analyze All Loaded Assets"):
            comparison_results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, symbol in enumerate(asset_options):
                status_text.text(f"Analyzing {symbol}...")
                
                asset_data = market_data[symbol]
                analysis = ai_advisor.analyze_asset(symbol, asset_data)
                comparison_results[symbol] = analysis
                
                progress_bar.progress((i + 1) / len(asset_options))
            
            status_text.text("Analysis complete!")
            
            # Display comparison table
            st.markdown("#### 📊 Asset Comparison Results")
            
            comparison_data = []
            for symbol, analysis in comparison_results.items():
                rec = analysis['recommendation']
                comparison_data.append({
                    'Asset': symbol,
                    'Price': f"${analysis['current_price']:,.2f}",
                    'Change 24h': f"{analysis['price_change_24h']:+.1f}%",
                    'Recommendation': f"{decision_colors.get(rec['decision'], '⚪')} {rec['decision']}",
                    'Confidence': rec['confidence_level'],
                    'Risk': rec['risk_level']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Summary insights
            buy_count = sum(1 for res in comparison_results.values() if res['recommendation']['decision'] in ['BUY', 'DCA_IN'])
            sell_count = sum(1 for res in comparison_results.values() if res['recommendation']['decision'] in ['SELL', 'DCA_OUT'])
            hold_count = sum(1 for res in comparison_results.values() if res['recommendation']['decision'] == 'HOLD')
            
            st.markdown("#### 🎯 Portfolio Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Buy/DCA In Signals", buy_count, f"🟢 {buy_count}/{len(asset_options)}")
            with col2:
                st.metric("Hold Signals", hold_count, f"⚪ {hold_count}/{len(asset_options)}")
            with col3:
                st.metric("Sell/DCA Out Signals", sell_count, f"🔴 {sell_count}/{len(asset_options)}")

def render_model_training_tab(market_data, period, interval):
    """Render the model training tab"""
    st.markdown("### 🧠 AI Model Training")
    st.markdown("Train machine learning models to improve prediction accuracy.")
    
    if not market_data:
        st.info("👆 **Load market data first** to train AI models!")
        return
    
    # Select asset for training
    suitable_assets = {k: v for k, v in market_data.items() if len(v) > 30}
    
    if not suitable_assets:
        st.warning("⚠️ **Need more data for training**")
        st.markdown("AI models need at least 30+ data points for training. Current data:")
        for symbol, data in market_data.items():
            st.markdown(f"- **{symbol}**: {len(data)} data points")
        st.info("💡 Try loading stock data (AAPL, MSFT, NVDA) which provides full historical data.")
        return
    
    st.success(f"✅ Found {len(suitable_assets)} assets with enough data for training!")
    
    training_asset = st.selectbox(
        "Choose asset to train models on:",
        list(suitable_assets.keys()),
        help="I'll train AI models using this asset's historical data"
    )
    
    if st.button("🚀 Start AI Model Training", type="primary"):
        with st.spinner("🤖 Training AI models... This may take a minute..."):
            try:
                from models.model_comparison import compare_models_safe
                
                df = suitable_assets[training_asset]
                st.info(f"Training models on {training_asset} with {len(df)} data points...")
                
                # Run model comparison (without LSTM to avoid TensorFlow issues)
                scoreboard, best_model, best_path = compare_models_safe(df)
                
                st.success(f"🎉 Training complete! Best model: **{best_model}**")
                
                # Display results
                st.markdown("### 📊 Model Performance Results")
                st.dataframe(scoreboard, use_container_width=True)
                
                # Model interpretation
                st.markdown("### 🔍 What This Means")
                
                best_row = scoreboard.iloc[0]  # Best model is first row
                accuracy = best_row['accuracy']
                
                if accuracy > 0.65:
                    performance_desc = "Excellent! The AI model shows strong predictive capability."
                    performance_color = "🟢"
                elif accuracy > 0.55:
                    performance_desc = "Good performance. The model can provide helpful insights."
                    performance_color = "🟡"
                else:
                    performance_desc = "Moderate performance. Use predictions with caution."
                    performance_color = "🟠"
                
                st.markdown(f"""
                {performance_color} **Model Performance:** {accuracy:.1%} accuracy
                
                **Interpretation:** {performance_desc}
                
                The **{best_model}** model has been saved and will be used for future predictions on {training_asset}.
                """)
                
                # Save training results
                import json
                
                timestamp = int(datetime.now().timestamp())
                training_record = {
                    "timestamp": timestamp,
                    "asset": training_asset,
                    "best_model": best_model,
                    "best_path": best_path,
                    "scoreboard": scoreboard.to_dict("records"),
                    "data_points": len(df),
                    "period": period
                }
                
                # Store in session state
                if 'training_history' not in st.session_state:
                    st.session_state.training_history = []
                st.session_state.training_history.append(training_record)
                
                st.info(f"💾 Training results saved. Model ready for predictions!")
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.info("This might be due to insufficient data or system limitations.")
    
    # Show training history if available
    if 'training_history' in st.session_state and st.session_state.training_history:
        st.markdown("---")
        st.markdown("### 📜 Training History")
        
        for record in reversed(st.session_state.training_history[-3:]):  # Show last 3
            timestamp = datetime.fromtimestamp(record['timestamp'])
            st.markdown(f"""
            **{record['asset']}** - {timestamp.strftime('%Y-%m-%d %H:%M')}  
            Best Model: {record['best_model']} | Data Points: {record['data_points']}
            """)

def render_backtest_tab(market_data):
    """Render the backtesting tab"""
    st.markdown("### 📊 Strategy Backtesting")
    st.markdown("Test how well the AI predictions would have performed historically.")
    
    if not market_data:
        st.info("👆 **Load market data first** to run backtests!")
        return
    
    # Check for trained models
    if 'training_history' not in st.session_state or not st.session_state.training_history:
        st.warning("🧠 **Train an AI model first** in the 'Model Training' tab!")
        st.info("Backtesting requires a trained model to generate trading signals.")
        return
    
    # Get latest training record
    latest_training = st.session_state.training_history[-1]
    model_asset = latest_training['asset']
    model_path = latest_training['best_path']
    
    st.success(f"✅ Using trained model: **{latest_training['best_model']}** (trained on {model_asset})")
    
    # Backtest settings
    st.markdown("### ⚙️ Backtest Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        threshold = st.slider(
            "🎯 Trading Signal Threshold", 
            0.45, 0.75, 0.60, 0.01,
            help="Higher = more conservative (fewer trades), Lower = more aggressive (more trades)"
        )
    
    with col2:
        initial_capital = st.number_input(
            "💰 Starting Capital", 
            1000, 100000, 10000, 1000,
            help="How much money to start the backtest with"
        )
    
    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("📈 Running backtest simulation..."):
            try:
                # Get data for backtesting
                if model_asset in market_data:
                    test_data = market_data[model_asset]
                    
                    if len(test_data) < 20:
                        st.error("❌ Need more historical data for meaningful backtest")
                        return
                    
                    # Load the trained model
                    if os.path.exists(model_path):
                        model = joblib.load(model_path)
                    else:
                        st.error(f"❌ Model file not found: {model_path}")
                        return
                    
                    # Feature engineering (simplified)
                    from backtest.features.feature_engineering import build_features
                    X, y, features, processed = build_features(test_data, horizon=1)
                    
                    if len(X) == 0:
                        st.error("❌ Could not generate features from data")
                        return
                    
                    # Generate predictions
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(X)[:, 1]
                    else:
                        predictions = model.predict(X)
                        probabilities = predictions.astype(float)
                    
                    # Convert to signals
                    signals = (probabilities >= threshold).astype(int)
                    
                    # Run backtest
                    from backtest.backtesting_engine import simulate_long_flat
                    
                    # Align data
                    price_series = processed['close'].iloc[:len(signals)]
                    signal_series = pd.Series(signals, index=price_series.index)
                    
                    metrics, equity_curve, trades = simulate_long_flat(
                        price_series, signal_series, start_capital=initial_capital
                    )
                    
                    # Display results
                    st.markdown("### 🏆 Backtest Results")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        final_return = metrics['total_return_pct']
                        st.metric(
                            "Total Return", 
                            f"{final_return:+.1f}%",
                            f"${metrics['final_equity'] - initial_capital:+,.0f}"
                        )
                    
                    with col2:
                        st.metric("Number of Trades", f"{metrics['n_trades']:,}")
                    
                    with col3:
                        sharpe = metrics.get('sharpe', 0)
                        sharpe_quality = "Excellent" if sharpe > 2 else "Good" if sharpe > 1 else "Poor"
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}", sharpe_quality)
                    
                    with col4:
                        max_dd = metrics.get('max_drawdown', 0) * 100
                        st.metric("Max Drawdown", f"{max_dd:.1f}%")
                    
                    # Performance interpretation
                    st.markdown("### 📈 Performance Analysis")
                    
                    if final_return > 20:
                        performance_emoji = "🚀"
                        performance_text = "Excellent performance! The strategy significantly outperformed."
                    elif final_return > 10:
                        performance_emoji = "📈"
                        performance_text = "Good performance! The strategy showed positive results."
                    elif final_return > 0:
                        performance_emoji = "👍"
                        performance_text = "Modest gains. The strategy was profitable."
                    else:
                        performance_emoji = "📉"
                        performance_text = "The strategy underperformed. Consider adjusting parameters."
                    
                    st.markdown(f"{performance_emoji} **{performance_text}**")
                    
                    # Trading activity
                    if metrics['n_trades'] > 0:
                        avg_trade_return = final_return / metrics['n_trades']
                        st.markdown(f"💼 **Average return per trade:** {avg_trade_return:.2f}%")
                        
                        if metrics['n_trades'] > 50:
                            st.info("📊 High trading frequency - Consider if transaction costs would impact returns")
                        elif metrics['n_trades'] < 5:
                            st.info("📊 Low trading frequency - Strategy is quite conservative")
                    
                    # Equity curve chart
                    st.markdown("### 📊 Portfolio Growth")
                    
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=equity_curve['date'],
                        y=equity_curve['equity'],
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='green', width=2)
                    ))
                    
                    # Add starting capital line
                    fig.add_hline(
                        y=initial_capital, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text=f"Starting Capital: ${initial_capital:,}"
                    )
                    
                    fig.update_layout(
                        title=f'{model_asset} Strategy Performance',
                        xaxis_title='Date',
                        yaxis_title='Portfolio Value (USD)',
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Strategy settings summary
                    with st.expander("⚙️ Backtest Settings"):
                        st.markdown(f"""
                        - **Asset:** {model_asset}
                        - **Model:** {latest_training['best_model']}
                        - **Signal Threshold:** {threshold}
                        - **Starting Capital:** ${initial_capital:,}
                        - **Data Points:** {len(test_data)}
                        - **Test Period:** {test_data.index[0].date()} to {test_data.index[-1].date()}
                        """)
                    
                    # Risk warning
                    st.warning("""
                    ⚠️ **Backtest Limitations:**
                    - Past performance doesn't guarantee future results
                    - Real trading involves fees, slippage, and market impact
                    - Market conditions change over time
                    - Always use proper risk management
                    """)
                
                else:
                    st.error(f"❌ No data available for {model_asset}")
                    
            except Exception as e:
                st.error(f"❌ Backtest failed: {str(e)}")
                st.info("This might be due to insufficient data or model compatibility issues.")