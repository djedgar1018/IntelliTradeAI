"""
IntelliTradeAI Hover Tooltip Definitions
Provides contextual definitions that pop up when hovering over trading terms
"""
import streamlit as st
from typing import Dict, Optional

class TooltipTerms:
    """Trading terms with definitions for hover tooltips"""
    
    TERMS = {
        "BUY": "A recommendation to purchase an asset because the AI predicts its price will increase.",
        "SELL": "A recommendation to sell an asset because the AI predicts its price will decrease.",
        "HOLD": "A recommendation to maintain your current position without buying or selling.",
        "Confidence": "How certain the AI is about its prediction. Higher percentages mean more certainty.",
        "Support": "A price 'floor' where the asset tends to stop dropping and buyers step in.",
        "Resistance": "A price 'ceiling' where the asset tends to stop climbing and sellers take profits.",
        "RSI": "Relative Strength Index - measures if an asset is overbought (>70) or oversold (<30).",
        "MACD": "Moving Average Convergence Divergence - shows whether momentum is increasing or decreasing.",
        "Bollinger Bands": "Bands showing normal price range. Prices outside may be extreme.",
        "Stop-Loss": "Automatic sell order that limits your loss if price drops too much.",
        "Take-Profit": "Automatic sell order that locks in profits when price reaches your goal.",
        "Position Size": "How much of your money you put into one trade.",
        "Portfolio": "All of your investments combined - stocks, crypto, options, etc.",
        "Volatility": "How 'wild' or stable the price movements are. High volatility means bigger swings.",
        "Market Cap": "Total value of all shares/coins - shows how 'big' a company or crypto is.",
        "Liquidity": "How easily you can buy or sell. High liquidity means fast, easy trades.",
        "Bull Market": "When prices are generally going up and investors are optimistic.",
        "Bear Market": "When prices are generally going down and investors are worried.",
        "ETF": "Exchange-Traded Fund - a basket of stocks that trades like a single stock.",
        "Options": "Contracts giving you the right to buy (call) or sell (put) at a specific price.",
        "Call Option": "The right to BUY an asset at a fixed price before expiration.",
        "Put Option": "The right to SELL an asset at a fixed price before expiration.",
        "Strike Price": "The price at which you can exercise your option contract.",
        "Premium": "The cost you pay to buy an options contract.",
        "Delta": "How much the option price moves when the stock price moves $1.",
        "Theta": "How much value the option loses each day (time decay).",
        "Implied Volatility": "Market's expectation of how much the stock price will move.",
        "DeFi": "Decentralized Finance - financial services on blockchain without intermediaries.",
        "Layer 1": "Main blockchain networks like Bitcoin, Ethereum, Solana.",
        "Layer 2": "Scaling solutions built on top of Layer 1 chains for faster transactions.",
        "Stablecoin": "Cryptocurrencies designed to maintain a stable value, usually $1.",
        "Risk Tolerance": "Your ability and willingness to lose money on investments.",
        "Diversification": "Spreading investments across different assets to reduce risk.",
        "Breakeven": "The price at which you neither make nor lose money on a trade.",
        "Greeks": "Mathematical values (Delta, Gamma, Theta, Vega) measuring option risk.",
        "Sector": "A category of stocks in the same industry (Technology, Healthcare, etc.).",
        "Index": "A measurement of a section of the market (S&P 500, NASDAQ, etc.).",
        "Dividend": "A payment made by a company to its shareholders from profits.",
        "Conservative": "Low-risk strategy focused on protecting your capital.",
        "Aggressive": "High-risk strategy focused on maximizing growth potential.",
        "Speculative": "Very high-risk strategy for experienced traders only."
    }
    
    @classmethod
    def get_definition(cls, term: str) -> Optional[str]:
        """Get definition for a term"""
        return cls.TERMS.get(term)
    
    @classmethod
    def inject_tooltip_css(cls):
        """Inject CSS for hover tooltips with 3-second delay"""
        st.markdown("""
        <style>
        .tooltip-term {
            position: relative;
            display: inline;
            border-bottom: 1px dotted #1f77b4;
            cursor: help;
            color: #1f77b4;
            font-weight: 500;
        }
        
        .tooltip-term .tooltip-content {
            visibility: hidden;
            opacity: 0;
            width: 280px;
            background-color: #1a1a2e;
            color: #fff;
            text-align: left;
            border-radius: 8px;
            padding: 12px 15px;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            margin-left: -140px;
            font-size: 14px;
            font-weight: normal;
            line-height: 1.5;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            border: 1px solid #0f3460;
            transition: opacity 0.3s ease-in-out, visibility 0.3s ease-in-out;
            transition-delay: 3s;
        }
        
        .tooltip-term .tooltip-content::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -8px;
            border-width: 8px;
            border-style: solid;
            border-color: #1a1a2e transparent transparent transparent;
        }
        
        .tooltip-term:hover .tooltip-content {
            visibility: visible;
            opacity: 1;
        }
        
        .tooltip-term .tooltip-header {
            color: #e94560;
            font-weight: bold;
            margin-bottom: 5px;
            font-size: 15px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @classmethod
    def wrap_term(cls, term: str, display_text: Optional[str] = None) -> str:
        """Wrap a term with tooltip HTML"""
        definition = cls.TERMS.get(term, "")
        display = display_text or term
        if definition:
            return f'''<span class="tooltip-term">{display}<span class="tooltip-content"><div class="tooltip-header">{term}</div>{definition}</span></span>'''
        return display
    
    @classmethod
    def render_text_with_tooltips(cls, text: str) -> str:
        """Replace known terms in text with tooltip-wrapped versions"""
        result = text
        for term in sorted(cls.TERMS.keys(), key=len, reverse=True):
            if term in result:
                wrapped = cls.wrap_term(term)
                result = result.replace(term, wrapped)
        return result


def inject_global_tooltips():
    """Inject tooltip CSS and JavaScript into the page"""
    TooltipTerms.inject_tooltip_css()


def tooltip(term: str, display_text: Optional[str] = None) -> str:
    """Convenience function to create a tooltip-wrapped term"""
    return TooltipTerms.wrap_term(term, display_text)
