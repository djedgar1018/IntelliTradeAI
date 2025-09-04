# --- make project root importable when running from app/ ---
import os, sys, time, json, glob
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from data.data_ingestion import DataIngestion
from models.model_comparison import compare_models
from backtest.features.feature_engineering import build_features
from backtest.backtesting_engine import simulate_long_flat, proba_to_signal
from ai_advisor.trading_intelligence import TradingIntelligence
from app.ai_analysis_tab import render_ai_analysis_tab, render_model_training_tab, render_backtest_tab

EXPERIMENTS_DIR = os.getenv("EXPERIMENTS_DIR", "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

def latest_compare_record():
    """Return the last compare_*.json dict or None."""
    paths = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, "compare_*.json")))
    if not paths:
        return None
    with open(paths[-1], "r") as f:
        return json.load(f)

def save_compare_record(symbol, scoreboard_df, best_model, best_path):
    ts = int(time.time())
    rec = {
        "ts": ts,
        "symbol": symbol,
        "best_model": best_model,
        "best_model_path": best_path,
        "scoreboard": scoreboard_df.to_dict("records"),
    }
    out = os.path.join(EXPERIMENTS_DIR, f"compare_{ts}.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    return out

st.set_page_config(page_title="🤖 AI Trading Advisor", layout="wide")
st.title("🤖 AI Trading Advisor")
st.caption("Your intelligent assistant for data-driven trading decisions")

# Initialize AI advisor
if 'ai_advisor' not in st.session_state:
    st.session_state.ai_advisor = TradingIntelligence()

# Welcome message
with st.container():
    st.markdown("""
    ### 👋 Welcome to your AI Trading Advisor!
    
    I analyze market data using advanced AI models and provide clear **BUY**, **SELL**, **HOLD**, or **DCA** recommendations with detailed explanations.
    
    📊 **What I do for you:**
    - Analyze price trends and technical indicators
    - Use machine learning models to predict price movements  
    - Provide clear trading recommendations with confidence levels
    - Explain my reasoning in simple terms
    - Suggest specific actions you should take
    
    💡 **Get started:** Choose your assets below and I'll analyze them for you!
    """)

# --- Sidebar controls ---
st.sidebar.markdown("### 🎛️ Configuration")
st.sidebar.markdown("**Choose the assets you want me to analyze:**")
crypto_text = st.sidebar.text_input("🪙 Crypto symbols (comma-separated)", "BTC,ETH,FET", help="Enter crypto symbols like BTC, ETH, ADA")
stock_text  = st.sidebar.text_input("📈 Stock symbols (comma-separated)",  "AAPL,MSFT,NVDA", help="Enter stock symbols like AAPL, GOOGL, TSLA")
st.sidebar.markdown("**Analysis Settings:**")
period   = st.sidebar.selectbox("📅 How much history to analyze?", ["1mo","3mo","6mo","1y"], index=3, help="More history = better analysis")
interval = st.sidebar.selectbox("⏰ Data frequency", ["1d","1h","30m","5m"], index=0, help="Daily data recommended for most analysis")

# Add helpful info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.info("""
**For best results:**
- Use stock symbols for full historical analysis
- Crypto shows current prices (free API limits)
- Try AAPL, MSFT, NVDA for demo
- Run AI analysis for detailed insights
""")

# Parse user lists
crypto_syms = [s.strip().upper() for s in crypto_text.split(",") if s.strip()]
stock_syms  = [s.strip().upper() for s in stock_text.split(",") if s.strip()]

ing = DataIngestion()

# Create user-friendly tabs
tab_overview, tab_ai_analysis, tab_models, tab_backtest = st.tabs([
    "🏠 Overview & Data", 
    "🤖 AI Analysis", 
    "🧠 Model Training", 
    "📊 Backtest Results"
])

# =========================
# 1) DATA TAB
# =========================
with tab_overview:
    st.markdown("### 📊 Market Data Overview")
    st.markdown("Get the latest market data for your chosen assets. I'll load this data for analysis.")
    
    # Store data in session state
    if 'market_data' not in st.session_state:
        st.session_state.market_data = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Load Your Portfolio Data")
        if st.button("🔄 Load All Selected Assets", type="primary"):
            with st.spinner("📡 Fetching market data for analysis..."):
                try:
                    MIX = ing.fetch_mixed_data(
                        crypto_symbols=crypto_syms if crypto_syms else [],
                        stock_symbols=stock_syms if stock_syms else [],
                        period=period, interval=interval
                    )
                    st.session_state.market_data = MIX
                    st.success(f"✅ Successfully loaded data for {len(MIX)} assets!")
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
                    st.session_state.market_data = {}
    
    with col2:
        st.markdown("### ₿ Quick Bitcoin Check")
        if st.button("⚡ Get Bitcoin Price"):
            with st.spinner("Getting Bitcoin data..."):
                try:
                    BTC = ing.fetch_crypto_data(["BTC"], period=period, interval=interval)
                    if BTC and "BTC" in BTC:
                        st.session_state.market_data["BTC"] = BTC["BTC"]
                        df = BTC["BTC"]
                        st.success(f"₿ Bitcoin: ${df['close'].iloc[-1]:,.2f}")
                    else:
                        st.warning("⚠️ Could not fetch Bitcoin data (API limitations)")
                except Exception as e:
                    st.error(f"❌ Bitcoin fetch error: {str(e)}")
    
    # Display loaded data
    if st.session_state.market_data:
        st.markdown("### 📋 Currently Loaded Assets")
        
        # Create summary metrics
        asset_list = list(st.session_state.market_data.items())
        cols = st.columns(min(len(asset_list), 4))
        for i, (symbol, df) in enumerate(asset_list):
            with cols[i % 4]:
                current_price = df['close'].iloc[-1]
                price_change = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100 if len(df) > 1 else 0
                
                st.metric(
                    label=f"📊 {symbol}",
                    value=f"${current_price:,.2f}",
                    delta=f"{price_change:+.1f}%" if len(df) > 1 else "Current"
                )
        
        # Show data quality info with charts
        st.markdown("**📈 Asset Details:**")
        
        for symbol, df in st.session_state.market_data.items():
            data_quality = "📊 Full Historical" if len(df) > 30 else "⚠️ Limited Data" if len(df) > 1 else "📍 Current Price Only"
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{symbol}:** {len(df)} data points | {data_quality}")
            
            with col2:
                if len(df) > 1:  # Only show chart option if we have historical data
                    with st.expander(f"📊 See Chart"):
                        import plotly.graph_objects as go
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['close'],
                            mode='lines',
                            name=f'{symbol} Price',
                            line=dict(color='#1f77b4', width=2)
                        ))
                        
                        fig.update_layout(
                            title=f'{symbol} Price History',
                            xaxis_title='Date',
                            yaxis_title='Price (USD)',
                            height=300,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Quick stats
                        price_change = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
                        st.caption(f"Total return: {price_change:+.1f}% | Period: {df.index[0].date()} to {df.index[-1].date()}")
    
    else:
        st.info("👆 Click 'Load All Selected Assets' above to get started with your market analysis!")
    

# =========================
# 2) AI ANALYSIS TAB
# =========================
with tab_ai_analysis:
    market_data = st.session_state.get('market_data', {})
    render_ai_analysis_tab(market_data, st.session_state.ai_advisor, period, interval)

# =========================
# 3) MODEL TRAINING TAB
# =========================
with tab_models:
    market_data = st.session_state.get('market_data', {})
    render_model_training_tab(market_data, period, interval)

# =========================
# 4) BACKTEST RESULTS TAB
# =========================
with tab_backtest:
    market_data = st.session_state.get('market_data', {})
    render_backtest_tab(market_data)