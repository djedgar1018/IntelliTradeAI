"""
IntelliTradeAI Enhanced Trading Plan Features
Sector rankings, popup charts, price alerts, and options suggestions based on tier
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class SectorRankings:
    """Sector and ETF rankings for the trading plan"""
    
    STOCK_SECTORS = {
        "Technology": {"etf": "XLK", "score": 92, "trend": "Bullish", "volatility": "High", "recommendation": "Strong Buy"},
        "Healthcare": {"etf": "XLV", "score": 85, "trend": "Bullish", "volatility": "Medium", "recommendation": "Buy"},
        "Financials": {"etf": "XLF", "score": 78, "trend": "Neutral", "volatility": "Medium", "recommendation": "Hold"},
        "Consumer Discretionary": {"etf": "XLY", "score": 75, "trend": "Bullish", "volatility": "High", "recommendation": "Buy"},
        "Industrials": {"etf": "XLI", "score": 72, "trend": "Neutral", "volatility": "Medium", "recommendation": "Hold"},
        "Energy": {"etf": "XLE", "score": 68, "trend": "Bearish", "volatility": "High", "recommendation": "Caution"},
        "Materials": {"etf": "XLB", "score": 65, "trend": "Neutral", "volatility": "Medium", "recommendation": "Hold"},
        "Consumer Staples": {"etf": "XLP", "score": 70, "trend": "Neutral", "volatility": "Low", "recommendation": "Hold"},
        "Utilities": {"etf": "XLU", "score": 62, "trend": "Neutral", "volatility": "Low", "recommendation": "Hold"},
        "Real Estate": {"etf": "XLRE", "score": 58, "trend": "Bearish", "volatility": "Medium", "recommendation": "Caution"},
        "Communication Services": {"etf": "XLC", "score": 80, "trend": "Bullish", "volatility": "High", "recommendation": "Buy"}
    }
    
    MAJOR_ETFS = {
        "SPY": {"name": "S&P 500", "score": 82, "trend": "Bullish", "type": "Broad Market"},
        "QQQ": {"name": "NASDAQ-100", "score": 90, "trend": "Bullish", "type": "Tech Heavy"},
        "IWM": {"name": "Russell 2000", "score": 65, "trend": "Neutral", "type": "Small Cap"},
        "DIA": {"name": "Dow Jones", "score": 75, "trend": "Neutral", "type": "Blue Chip"},
        "VTI": {"name": "Total Stock Market", "score": 80, "trend": "Bullish", "type": "Broad Market"},
        "ARKK": {"name": "ARK Innovation", "score": 78, "trend": "Bullish", "type": "Growth/Innovation"},
        "VGT": {"name": "Vanguard Tech", "score": 88, "trend": "Bullish", "type": "Technology"},
        "VOO": {"name": "Vanguard S&P 500", "score": 82, "trend": "Bullish", "type": "Broad Market"},
        "GLD": {"name": "Gold", "score": 70, "trend": "Neutral", "type": "Commodity"},
        "TLT": {"name": "20+ Year Treasury", "score": 55, "trend": "Bearish", "type": "Bonds"}
    }
    
    @classmethod
    def get_sector_rankings_df(cls) -> pd.DataFrame:
        """Get sector rankings as DataFrame"""
        data = []
        rank = 1
        for sector, info in sorted(cls.STOCK_SECTORS.items(), key=lambda x: x[1]["score"], reverse=True):
            data.append({
                "Rank": rank,
                "Sector": sector,
                "ETF": info["etf"],
                "AI Score": info["score"],
                "Trend": info["trend"],
                "Volatility": info["volatility"],
                "Recommendation": info["recommendation"]
            })
            rank += 1
        return pd.DataFrame(data)
    
    @classmethod
    def get_etf_rankings_df(cls) -> pd.DataFrame:
        """Get ETF rankings as DataFrame"""
        data = []
        rank = 1
        for etf, info in sorted(cls.MAJOR_ETFS.items(), key=lambda x: x[1]["score"], reverse=True):
            data.append({
                "Rank": rank,
                "Symbol": etf,
                "Name": info["name"],
                "Type": info["type"],
                "AI Score": info["score"],
                "Trend": info["trend"]
            })
            rank += 1
        return pd.DataFrame(data)
    
    @classmethod
    def render_sector_rankings_table(cls):
        """Render the sector rankings table"""
        st.subheader("Stock Sector Rankings")
        st.caption("AI-powered sector analysis based on momentum, fundamentals, and market conditions")
        
        df = cls.get_sector_rankings_df()
        
        def color_score(val):
            if val >= 80:
                color = '#16c784'
            elif val >= 60:
                color = '#f5a623'
            else:
                color = '#ea3943'
            return f'color: {color}; font-weight: bold'
        
        def color_recommendation(val):
            colors = {
                "Strong Buy": "#16c784",
                "Buy": "#4ade80",
                "Hold": "#f5a623",
                "Caution": "#ea3943"
            }
            return f'color: {colors.get(val, "#ffffff")}; font-weight: bold'
        
        styled_df = df.style.applymap(color_score, subset=['AI Score'])\
                           .applymap(color_recommendation, subset=['Recommendation'])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    @classmethod
    def render_etf_rankings_table(cls):
        """Render the ETF rankings table"""
        st.subheader("Major ETF Rankings")
        st.caption("Ranked by AI analysis of momentum and relative strength")
        
        df = cls.get_etf_rankings_df()
        
        def color_score(val):
            if val >= 80:
                color = '#16c784'
            elif val >= 60:
                color = '#f5a623'
            else:
                color = '#ea3943'
            return f'color: {color}; font-weight: bold'
        
        styled_df = df.style.applymap(color_score, subset=['AI Score'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)


class OptimalLevelCharts:
    """Pop-up charts showing optimal buy/sell levels for assets"""
    
    @classmethod
    def generate_sample_price_data(cls, symbol: str, current_price: float = 100.0) -> pd.DataFrame:
        """Generate sample price data for demo"""
        dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
        np.random.seed(hash(symbol) % 100)
        
        returns = np.random.randn(60) * 0.02
        prices = [current_price]
        for r in returns[:-1]:
            prices.append(prices[-1] * (1 + r))
        prices = prices[::-1]
        
        high = [p * (1 + abs(np.random.randn() * 0.01)) for p in prices]
        low = [p * (1 - abs(np.random.randn() * 0.01)) for p in prices]
        
        return pd.DataFrame({
            'date': dates,
            'open': [p * (1 + np.random.randn() * 0.005) for p in prices],
            'high': high,
            'low': low,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 60)
        })
    
    @classmethod
    def calculate_optimal_levels(cls, df: pd.DataFrame) -> Dict:
        """Calculate support, resistance, and optimal entry/exit levels"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        current = close[-1]
        
        sma_20 = np.mean(close[-20:])
        sma_50 = np.mean(close[-50:]) if len(close) >= 50 else np.mean(close)
        
        std = np.std(close[-20:])
        bb_upper = sma_20 + (2 * std)
        bb_lower = sma_20 - (2 * std)
        
        recent_low = np.min(low[-20:])
        recent_high = np.max(high[-20:])
        
        support_1 = recent_low
        support_2 = recent_low * 0.95
        resistance_1 = recent_high
        resistance_2 = recent_high * 1.05
        
        optimal_buy = support_1 * 1.01
        optimal_sell = resistance_1 * 0.99
        stop_loss = support_1 * 0.97
        take_profit = resistance_1 * 1.02
        
        return {
            "current_price": current,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "support_1": support_1,
            "support_2": support_2,
            "resistance_1": resistance_1,
            "resistance_2": resistance_2,
            "optimal_buy": optimal_buy,
            "optimal_sell": optimal_sell,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }
    
    @classmethod
    def create_optimal_levels_chart(cls, symbol: str, df: pd.DataFrame, levels: Dict) -> go.Figure:
        """Create a chart with optimal levels highlighted"""
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#16c784',
            decreasing_line_color='#ea3943'
        ))
        
        fig.add_hline(y=levels["support_1"], line_dash="dash", line_color="#16c784", 
                      annotation_text=f"Support 1: ${levels['support_1']:.2f}")
        fig.add_hline(y=levels["support_2"], line_dash="dot", line_color="#16c784",
                      annotation_text=f"Support 2: ${levels['support_2']:.2f}")
        
        fig.add_hline(y=levels["resistance_1"], line_dash="dash", line_color="#ea3943",
                      annotation_text=f"Resistance 1: ${levels['resistance_1']:.2f}")
        fig.add_hline(y=levels["resistance_2"], line_dash="dot", line_color="#ea3943",
                      annotation_text=f"Resistance 2: ${levels['resistance_2']:.2f}")
        
        fig.add_hline(y=levels["optimal_buy"], line_dash="solid", line_color="#00ff88", line_width=2,
                      annotation_text=f"Optimal Buy: ${levels['optimal_buy']:.2f}")
        fig.add_hline(y=levels["optimal_sell"], line_dash="solid", line_color="#ff4444", line_width=2,
                      annotation_text=f"Optimal Sell: ${levels['optimal_sell']:.2f}")
        
        fig.update_layout(
            title=f"{symbol} - Optimal Trading Levels",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            template="plotly_dark",
            height=500,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    @classmethod
    def render_asset_chart_popup(cls, symbol: str, current_price: float = None):
        """Render a popup-style chart for an asset with optimal levels"""
        if current_price is None:
            current_price = 100.0 + (hash(symbol) % 1000)
        
        df = cls.generate_sample_price_data(symbol, current_price)
        levels = cls.calculate_optimal_levels(df)
        
        with st.expander(f"View {symbol} Chart with Optimal Levels", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current", f"${levels['current_price']:.2f}")
            with col2:
                st.metric("Optimal Buy", f"${levels['optimal_buy']:.2f}", 
                          delta=f"{((levels['optimal_buy']/levels['current_price'])-1)*100:.1f}%")
            with col3:
                st.metric("Optimal Sell", f"${levels['optimal_sell']:.2f}",
                          delta=f"{((levels['optimal_sell']/levels['current_price'])-1)*100:.1f}%")
            with col4:
                st.metric("Stop Loss", f"${levels['stop_loss']:.2f}")
            
            fig = cls.create_optimal_levels_chart(symbol, df, levels)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("**Key Levels Summary:**")
            levels_df = pd.DataFrame({
                "Level": ["Support 1", "Support 2", "Resistance 1", "Resistance 2", "Take Profit"],
                "Price": [f"${levels['support_1']:.2f}", f"${levels['support_2']:.2f}", 
                         f"${levels['resistance_1']:.2f}", f"${levels['resistance_2']:.2f}",
                         f"${levels['take_profit']:.2f}"],
                "Action": ["Consider buying", "Strong buy zone", "Consider selling", "Strong resistance", "Target exit"]
            })
            st.dataframe(levels_df, use_container_width=True, hide_index=True)


class PriceAlerts:
    """Price alert functionality for trading plan assets"""
    
    @classmethod
    def render_price_alerts_section(cls, recommended_assets: List[str]):
        """Render price alerts section for recommended assets"""
        st.subheader("Price Alerts")
        st.caption("Set alerts to be notified when assets reach your target prices")
        
        if 'price_alerts' not in st.session_state:
            st.session_state.price_alerts = []
        
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            alert_asset = st.selectbox("Select Asset", recommended_assets, key="alert_asset_select")
        
        with col2:
            alert_type = st.selectbox("Alert Type", 
                                      ["Price Above", "Price Below", "Percent Change Up", "Percent Change Down"],
                                      key="alert_type_select")
        
        with col3:
            if "Percent" in alert_type:
                alert_value = st.number_input("Percentage (%)", min_value=0.1, max_value=100.0, value=5.0, 
                                              step=0.5, key="alert_value_input")
            else:
                alert_value = st.number_input("Target Price ($)", min_value=0.01, max_value=1000000.0, 
                                              value=100.0, step=1.0, key="alert_value_input")
        
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Add Alert", type="primary", key="add_alert_btn"):
                alert = {
                    "id": len(st.session_state.price_alerts) + 1,
                    "asset": alert_asset,
                    "type": alert_type,
                    "value": alert_value,
                    "created": datetime.now().isoformat(),
                    "active": True
                }
                st.session_state.price_alerts.append(alert)
                st.success(f"Alert added for {alert_asset}!")
                st.rerun()
        
        if st.session_state.price_alerts:
            st.markdown("---")
            st.markdown("**Active Alerts:**")
            
            alerts_df = pd.DataFrame([
                {
                    "Asset": a["asset"],
                    "Type": a["type"],
                    "Target": f"${a['value']:.2f}" if "Price" in a["type"] else f"{a['value']}%",
                    "Status": "Active" if a["active"] else "Triggered"
                }
                for a in st.session_state.price_alerts
            ])
            
            st.dataframe(alerts_df, use_container_width=True, hide_index=True)
            
            if st.button("Clear All Alerts", key="clear_alerts_btn"):
                st.session_state.price_alerts = []
                st.rerun()


class OptionsRecommendations:
    """Options trading recommendations based on risk tier"""
    
    TIER_OPTIONS_STRATEGIES = {
        "conservative": {
            "allowed_strategies": ["Covered Calls", "Cash-Secured Puts"],
            "call_recommendation": "In-the-money calls for downside protection with income potential",
            "put_recommendation": "Cash-secured puts on stocks you want to own at lower prices",
            "max_allocation": 10,
            "prefer_direction": "neutral_to_bullish",
            "example_calls": [
                {"symbol": "AAPL", "strategy": "Covered Call", "strike": "ITM", "expiry": "30-45 days"},
                {"symbol": "MSFT", "strategy": "Covered Call", "strike": "ITM", "expiry": "30-45 days"}
            ],
            "example_puts": [
                {"symbol": "JNJ", "strategy": "Cash-Secured Put", "strike": "OTM 5-10%", "expiry": "30-45 days"},
                {"symbol": "PG", "strategy": "Cash-Secured Put", "strike": "OTM 5-10%", "expiry": "30-45 days"}
            ]
        },
        "moderate": {
            "allowed_strategies": ["Covered Calls", "Cash-Secured Puts", "Protective Puts", "Bull Call Spreads"],
            "call_recommendation": "At-the-money calls for balanced risk/reward on bullish positions",
            "put_recommendation": "Protective puts to hedge existing long positions",
            "max_allocation": 20,
            "prefer_direction": "balanced",
            "example_calls": [
                {"symbol": "SPY", "strategy": "Bull Call Spread", "strike": "ATM", "expiry": "30-60 days"},
                {"symbol": "QQQ", "strategy": "Long Call", "strike": "ATM", "expiry": "45-60 days"}
            ],
            "example_puts": [
                {"symbol": "SPY", "strategy": "Protective Put", "strike": "OTM 5%", "expiry": "60-90 days"},
                {"symbol": "IWM", "strategy": "Long Put", "strike": "ATM", "expiry": "30-45 days"}
            ]
        },
        "growth": {
            "allowed_strategies": ["Long Calls", "Long Puts", "Spreads", "Straddles"],
            "call_recommendation": "Long calls on high-conviction bullish plays with defined risk",
            "put_recommendation": "Long puts for bearish momentum plays or hedging",
            "max_allocation": 30,
            "prefer_direction": "directional",
            "example_calls": [
                {"symbol": "NVDA", "strategy": "Long Call", "strike": "Slightly OTM", "expiry": "45-90 days"},
                {"symbol": "TSLA", "strategy": "Bull Call Spread", "strike": "ATM", "expiry": "30-45 days"}
            ],
            "example_puts": [
                {"symbol": "ARKK", "strategy": "Long Put", "strike": "ATM", "expiry": "30-45 days"},
                {"symbol": "XLE", "strategy": "Bear Put Spread", "strike": "ATM", "expiry": "30-45 days"}
            ]
        },
        "aggressive": {
            "allowed_strategies": ["All Basic Strategies", "Iron Condors", "Butterflies", "LEAPS"],
            "call_recommendation": "Out-of-the-money calls for high-leverage bullish bets",
            "put_recommendation": "Out-of-the-money puts for speculative downside plays",
            "max_allocation": 40,
            "prefer_direction": "momentum",
            "example_calls": [
                {"symbol": "COIN", "strategy": "Long Call", "strike": "OTM 10-15%", "expiry": "14-30 days"},
                {"symbol": "AMD", "strategy": "LEAPS Call", "strike": "OTM 20%", "expiry": "6-12 months"}
            ],
            "example_puts": [
                {"symbol": "MSTR", "strategy": "Long Put", "strike": "OTM 10%", "expiry": "14-30 days"},
                {"symbol": "GME", "strategy": "Bear Put Spread", "strike": "ATM", "expiry": "7-14 days"}
            ]
        },
        "speculative": {
            "allowed_strategies": ["All Strategies Including High Risk"],
            "call_recommendation": "Weekly OTM calls on volatile stocks for maximum leverage",
            "put_recommendation": "Weekly OTM puts for quick bearish plays on overextended stocks",
            "max_allocation": 50,
            "prefer_direction": "all_opportunities",
            "example_calls": [
                {"symbol": "0DTE SPY", "strategy": "Weekly Call", "strike": "OTM 1-2%", "expiry": "Same day"},
                {"symbol": "TSLA", "strategy": "Weekly Call", "strike": "OTM 5-10%", "expiry": "0-3 days"}
            ],
            "example_puts": [
                {"symbol": "0DTE SPY", "strategy": "Weekly Put", "strike": "OTM 1-2%", "expiry": "Same day"},
                {"symbol": "NVDA", "strategy": "Weekly Put", "strike": "OTM 5-10%", "expiry": "0-3 days"}
            ]
        }
    }
    
    @classmethod
    def get_options_suggestions(cls, risk_tier: str, market_outlook: str = "neutral") -> Dict:
        """Get options suggestions based on risk tier and market outlook"""
        tier_config = cls.TIER_OPTIONS_STRATEGIES.get(risk_tier.lower(), 
                                                       cls.TIER_OPTIONS_STRATEGIES["moderate"])
        
        call_suggestions = []
        put_suggestions = []
        
        if market_outlook in ["bullish", "neutral"]:
            call_suggestions = [
                {"strategy": "Bull Call Spread", "description": "Limited risk bullish play", 
                 "risk": "Defined", "reward": "Moderate"},
                {"strategy": "Long Call", "description": "Direct bullish exposure",
                 "risk": "Premium paid", "reward": "Unlimited"}
            ]
        
        if market_outlook in ["bearish", "neutral"]:
            put_suggestions = [
                {"strategy": "Bear Put Spread", "description": "Limited risk bearish play",
                 "risk": "Defined", "reward": "Moderate"},
                {"strategy": "Long Put", "description": "Direct bearish exposure or hedge",
                 "risk": "Premium paid", "reward": "Substantial"}
            ]
        
        if risk_tier.lower() in ["aggressive", "speculative"]:
            call_suggestions.append({
                "strategy": "Naked Call (Advanced)", "description": "Unlimited risk - not for beginners",
                "risk": "Unlimited", "reward": "Premium collected"
            })
            put_suggestions.append({
                "strategy": "Straddle", "description": "Profit from big moves in either direction",
                "risk": "Both premiums", "reward": "Unlimited on either side"
            })
        
        return {
            "tier": risk_tier,
            "market_outlook": market_outlook,
            "allowed_strategies": tier_config["allowed_strategies"],
            "call_guidance": tier_config["call_recommendation"],
            "put_guidance": tier_config["put_recommendation"],
            "max_allocation": tier_config["max_allocation"],
            "call_suggestions": call_suggestions,
            "put_suggestions": put_suggestions
        }
    
    @classmethod
    def render_options_suggestions(cls, risk_tier: str):
        """Render options suggestions section"""
        st.subheader("Options Trading Suggestions")
        st.caption(f"Based on your {risk_tier.title()} risk profile")
        
        tier_config = cls.TIER_OPTIONS_STRATEGIES.get(risk_tier.lower(), 
                                                       cls.TIER_OPTIONS_STRATEGIES["moderate"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            market_outlook = st.selectbox("Current Market Outlook", 
                                          ["neutral", "bullish", "bearish"],
                                          key="options_market_outlook")
        
        with col2:
            st.info(f"Max options allocation: {tier_config.get('max_allocation', 20)}% of portfolio")
        
        suggestions = cls.get_options_suggestions(risk_tier, market_outlook)
        
        st.markdown("**Allowed Strategies for Your Tier:**")
        st.write(", ".join(suggestions["allowed_strategies"]))
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Call Option Suggestions")
            st.info(suggestions["call_guidance"])
            
            if suggestions["call_suggestions"]:
                for s in suggestions["call_suggestions"]:
                    with st.container():
                        st.markdown(f"**{s['strategy']}**")
                        st.write(s["description"])
                        st.caption(f"Risk: {s['risk']} | Reward: {s['reward']}")
                        st.markdown("---")
            
            st.markdown("**Educational Examples for Your Tier:**")
            st.caption("These are example strategies suited to your risk profile. Always research before trading.")
            example_calls = tier_config.get("example_calls", [])
            if example_calls:
                calls_df = pd.DataFrame(example_calls)
                st.dataframe(calls_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### Put Option Suggestions")
            st.info(suggestions["put_guidance"])
            
            if suggestions["put_suggestions"]:
                for s in suggestions["put_suggestions"]:
                    with st.container():
                        st.markdown(f"**{s['strategy']}**")
                        st.write(s["description"])
                        st.caption(f"Risk: {s['risk']} | Reward: {s['reward']}")
                        st.markdown("---")
            
            st.markdown("**Educational Examples for Your Tier:**")
            st.caption("These are example strategies suited to your risk profile. Always research before trading.")
            example_puts = tier_config.get("example_puts", [])
            if example_puts:
                puts_df = pd.DataFrame(example_puts)
                st.dataframe(puts_df, use_container_width=True, hide_index=True)
