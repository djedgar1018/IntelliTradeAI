"""
IntelliTradeAI Dashboard
Streamlit interface for the AI trading agent
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_trainer import RobustModelTrainer
from data.data_ingestion import DataIngestion

# Page configuration
st.set_page_config(
    page_title="IntelliTradeAI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def get_components():
    trainer = RobustModelTrainer()
    ingestor = DataIngestion()
    return trainer, ingestor

trainer, ingestor = get_components()

# Main title
st.title("📊 IntelliTradeAI Dashboard")
st.markdown("AI-Powered Trading Agent with Advanced Machine Learning")

# Sidebar menu
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox(
    "Select Operation", 
    ["Overview", "Fetch Data", "Retrain Models", "Make Predictions", "Model Analysis"]
)

# Overview Section
if menu == "Overview":
    st.header("🎯 System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Status", "Active", "🟢")
    
    with col2:
        try:
            model_summary = trainer.get_model_summary()
            st.metric("Trained Models", model_summary["total_models"])
        except:
            st.metric("Trained Models", "0")
    
    with col3:
        st.metric("Supported Assets", "Stocks + Crypto")
    
    st.markdown("---")
    
    st.subheader("🚀 Features")
    
    features_col1, features_col2 = st.columns(2)
    
    with features_col1:
        st.markdown("""
        **Data Processing:**
        - Real-time market data fetching
        - 80+ technical indicators
        - Advanced feature engineering
        - Multi-timeframe analysis
        """)
        
        st.markdown("""
        **Machine Learning:**
        - Random Forest models
        - XGBoost algorithms
        - Feature selection & scaling
        - Cross-validation
        """)
    
    with features_col2:
        st.markdown("""
        **Supported Markets:**
        - Cryptocurrencies (BTC, ETH, LTC)
        - US Stocks (NASDAQ, NYSE)
        - Real-time price feeds
        - Historical data analysis
        """)
        
        st.markdown("""
        **Trading Signals:**
        - Buy/Sell predictions
        - Confidence scoring
        - Feature importance
        - Model explanations
        """)
    
    st.markdown("---")
    st.info("💡 **Quick Start:** Use the sidebar to fetch data, train models, or get predictions for your assets.")

# Fetch Data Section
elif menu == "Fetch Data":
    st.header("📈 Market Data Fetcher")
    
    # Data type selection
    col1, col2 = st.columns(2)
    
    with col1:
        data_type = st.selectbox("Data Type", ["Mixed", "Cryptocurrency", "Stocks"])
    
    with col2:
        if data_type == "Cryptocurrency":
            symbols = st.multiselect("Crypto Symbols", ["BTC", "ETH", "LTC"], default=["BTC"])
        elif data_type == "Stocks":
            symbols_input = st.text_input("Stock Symbols (comma-separated)", "AAPL,GOOGL,MSFT")
            symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        else:
            symbols_input = st.text_input("Mixed Symbols (comma-separated)", "BTC,AAPL,ETH")
            symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    if st.button("Fetch Data", type="primary"):
        if symbols:
            with st.spinner("Fetching market data..."):
                try:
                    data_results = {}
                    
                    # Separate crypto and stock symbols
                    crypto_symbols = [s for s in symbols if s in ["BTC", "ETH", "LTC"]]
                    stock_symbols = [s for s in symbols if s not in ["BTC", "ETH", "LTC"]]
                    
                    # Fetch crypto data
                    if crypto_symbols:
                        crypto_data = ingestor.fetch_crypto_data(crypto_symbols)
                        if crypto_data:
                            data_results.update(crypto_data)
                    
                    # Fetch stock data  
                    if stock_symbols:
                        stock_data = ingestor.fetch_stock_data(stock_symbols)
                        if stock_data:
                            data_results.update(stock_data)
                    
                    if data_results:
                        st.success(f"✅ Successfully fetched data for {len(data_results)} symbols")
                        
                        # Display data summary
                        for symbol, data in data_results.items():
                            st.subheader(f"{symbol} Data")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Latest Price", f"${data['close'].iloc[-1]:.2f}")
                            with col2:
                                st.metric("Data Points", len(data))
                            with col3:
                                price_change = ((data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]) * 100
                                st.metric("24h Change", f"{price_change:.2f}%")
                            with col4:
                                st.metric("Volume", f"{data['volume'].iloc[-1]:,.0f}")
                            
                            # Price chart
                            fig = go.Figure()
                            fig.add_trace(go.Candlestick(
                                x=data.index,
                                open=data['open'],
                                high=data['high'],
                                low=data['low'],
                                close=data['close'],
                                name=symbol
                            ))
                            fig.update_layout(
                                title=f"{symbol} Price Chart",
                                yaxis_title="Price ($)",
                                xaxis_title="Date",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("---")
                    else:
                        st.error("❌ No data could be fetched for the specified symbols")
                        
                except Exception as e:
                    st.error(f"❌ Error fetching data: {str(e)}")
        else:
            st.warning("⚠️ Please enter at least one symbol")

# Retrain Models Section
elif menu == "Retrain Models":
    st.header("🔄 Model Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.selectbox("Asset Symbol", ["BTC", "ETH", "LTC", "AAPL", "GOOGL", "MSFT", "TSLA"])
    
    with col2:
        algorithms = st.multiselect(
            "Algorithms", 
            ["random_forest", "xgboost"], 
            default=["random_forest"]
        )
    
    optimize = st.checkbox("Enable Hyperparameter Optimization", value=False)
    
    if st.button("Start Training", type="primary"):
        if symbol and algorithms:
            with st.spinner(f"Training models for {symbol}..."):
                try:
                    # Fetch training data
                    if symbol in ["BTC", "ETH", "LTC"]:
                        data_dict = ingestor.fetch_crypto_data([symbol])
                    else:
                        data_dict = ingestor.fetch_stock_data([symbol])
                    
                    if data_dict and symbol in data_dict:
                        data = data_dict[symbol]
                        
                        # Start training
                        results = trainer.run_comprehensive_training(
                            data,
                            symbol,
                            algorithms=algorithms,
                            optimize_hyperparams=optimize
                        )
                        
                        st.success("✅ Model training completed!")
                        
                        # Display results
                        st.subheader("Training Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Training Samples", results["data_shape"]["train"][0])
                        with col2:
                            st.metric("Test Samples", results["data_shape"]["test"][0])
                        with col3:
                            st.metric("Features Used", results["data_shape"]["train"][1])
                        
                        # Model performance
                        st.subheader("Model Performance")
                        
                        performance_data = []
                        for algorithm, result in results["models"].items():
                            if result["status"] == "success":
                                metrics = result["metrics"]
                                performance_data.append({
                                    "Algorithm": algorithm.upper(),
                                    "Accuracy": f"{metrics['accuracy']:.4f}",
                                    "Precision": f"{metrics.get('precision', 0):.4f}",
                                    "Recall": f"{metrics.get('recall', 0):.4f}",
                                    "F1 Score": f"{metrics.get('f1_score', 0):.4f}",
                                    "AUC": f"{metrics.get('test_auc', 0):.4f}"
                                })
                        
                        if performance_data:
                            st.dataframe(pd.DataFrame(performance_data), use_container_width=True)
                            
                            # Best model
                            best_model = results["best_model"]
                            st.success(f"🏆 Best Model: {best_model['algorithm'].upper()} (Accuracy: {best_model['score']:.4f})")
                        
                    else:
                        st.error(f"❌ Could not fetch data for {symbol}")
                        
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
        else:
            st.warning("⚠️ Please select a symbol and at least one algorithm")

# Make Predictions Section
elif menu == "Make Predictions":
    st.header("🔮 Trading Predictions")
    
    # Symbol selection
    symbol = st.selectbox("Select Asset", ["BTC", "ETH", "LTC", "AAPL", "GOOGL", "MSFT", "TSLA"])
    
    if st.button("Generate Predictions", type="primary"):
        with st.spinner(f"Generating predictions for {symbol}..."):
            try:
                # Check for available models
                model_summary = trainer.get_model_summary()
                available_models = [key for key in model_summary["models"].keys() if key.startswith(symbol)]
                
                if not available_models:
                    st.warning(f"⚠️ No trained models found for {symbol}. Please train a model first.")
                else:
                    # Fetch current data
                    if symbol in ["BTC", "ETH", "LTC"]:
                        data_dict = ingestor.fetch_crypto_data([symbol])
                    else:
                        data_dict = ingestor.fetch_stock_data([symbol])
                    
                    if data_dict and symbol in data_dict:
                        data = data_dict[symbol]
                        
                        # Current price info
                        current_price = data['close'].iloc[-1]
                        price_change = ((data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Price", f"${current_price:.2f}")
                        with col2:
                            st.metric("24h Change", f"{price_change:.2f}%", f"{price_change:.2f}%")
                        with col3:
                            st.metric("Available Models", len(available_models))
                        
                        # Generate features and predictions
                        features = trainer.engineer_features(data, symbol)
                        
                        st.subheader("🎯 Prediction Results")
                        
                        prediction_results = []
                        
                        for model_key in available_models:
                            model_info = model_summary["models"][model_key]
                            algorithm = model_info["algorithm"]
                            
                            try:
                                # Make prediction
                                predictions = trainer.make_predictions(features, symbol, algorithm.lower().replace("classifier", ""))
                                
                                if predictions is not None and len(predictions) > 0:
                                    latest_signal = predictions[-1]
                                    signal_text = "🟢 BUY" if latest_signal == 1 else "🔴 SELL"
                                    
                                    prediction_results.append({
                                        "Model": algorithm,
                                        "Signal": signal_text,
                                        "Confidence": f"{abs(latest_signal):.2f}",
                                        "Features": model_info["features"]
                                    })
                                    
                            except Exception as model_error:
                                prediction_results.append({
                                    "Model": algorithm,
                                    "Signal": "❌ Error",
                                    "Confidence": "N/A",
                                    "Features": model_info["features"]
                                })
                        
                        if prediction_results:
                            st.dataframe(pd.DataFrame(prediction_results), use_container_width=True)
                            
                            # Consensus signal
                            buy_signals = sum(1 for r in prediction_results if "🟢" in r["Signal"])
                            total_signals = len([r for r in prediction_results if "❌" not in r["Signal"]])
                            
                            if total_signals > 0:
                                consensus = "🟢 BUY" if buy_signals > total_signals/2 else "🔴 SELL"
                                consensus_strength = max(buy_signals, total_signals - buy_signals) / total_signals
                                
                                st.subheader("📊 Consensus Signal")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Consensus", consensus)
                                with col2:
                                    st.metric("Strength", f"{consensus_strength:.2f}")
                        
                    else:
                        st.error(f"❌ Could not fetch current data for {symbol}")
                    
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

# Model Analysis Section
elif menu == "Model Analysis":
    st.header("🔍 Model Analysis")
    
    try:
        model_summary = trainer.get_model_summary()
        
        if model_summary["total_models"] > 0:
            st.metric("Total Trained Models", model_summary["total_models"])
            
            # Model overview table
            model_data = []
            for model_key, info in model_summary["models"].items():
                symbol = model_key.split("_")[0]
                algorithm = model_key.split("_")[1]
                
                model_data.append({
                    "Model ID": model_key,
                    "Symbol": symbol,
                    "Algorithm": algorithm.upper(),
                    "Features": info["features"],
                    "Has Scaler": "✅" if info["has_scaler"] else "❌",
                    "Has Selector": "✅" if info["has_selector"] else "❌"
                })
            
            st.subheader("📋 Model Registry")
            st.dataframe(pd.DataFrame(model_data), use_container_width=True)
            
            # Model performance comparison
            st.subheader("📊 Performance Analysis")
            st.info("💡 Performance metrics are available after making predictions with the models")
            
        else:
            st.warning("⚠️ No trained models found. Use the 'Retrain Models' section to train your first model.")
            
    except Exception as e:
        st.error(f"❌ Error analyzing models: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**IntelliTradeAI** | AI-Powered Trading Agent | Built with Streamlit & FastAPI")