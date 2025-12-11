"""
IntelliTradeAI Interactive Trading Dictionary
Provides contextual definitions and explanations for trading terms
"""
import streamlit as st
from typing import Dict, List, Optional

class TradingDictionary:
    """Interactive dictionary for trading terms and concepts"""
    
    TERMS = {
        "BUY Signal": {
            "definition": "A recommendation to purchase an asset because the AI predicts its price will increase.",
            "simple": "The AI thinks this is a good time to buy because the price might go up.",
            "example": "If Bitcoin shows a BUY signal with 75% confidence, the AI believes there's a 75% chance the price will rise.",
            "category": "Signals"
        },
        "SELL Signal": {
            "definition": "A recommendation to sell an asset because the AI predicts its price will decrease.",
            "simple": "The AI thinks you should sell because the price might go down.",
            "example": "A SELL signal on Apple stock suggests the AI sees signs of a potential price drop.",
            "category": "Signals"
        },
        "HOLD Signal": {
            "definition": "A recommendation to maintain your current position without buying or selling.",
            "simple": "The AI suggests waiting - don't buy or sell right now.",
            "example": "A HOLD signal means the AI doesn't see a clear opportunity to profit from trading at this moment.",
            "category": "Signals"
        },
        "Confidence Score": {
            "definition": "A percentage indicating how certain the AI is about its prediction, based on agreement between multiple analysis methods.",
            "simple": "How sure the AI is about its recommendation. Higher percentages mean more certainty.",
            "example": "A 85% confidence means the AI's different models strongly agree on the prediction.",
            "category": "Signals"
        },
        "Support Level": {
            "definition": "A price point where an asset historically stops falling and may bounce back up due to increased buying interest.",
            "simple": "A 'floor' price where the asset tends to stop dropping and buyers step in.",
            "example": "If Bitcoin has support at $40,000, it often bounces back when it drops to that price.",
            "category": "Technical Analysis"
        },
        "Resistance Level": {
            "definition": "A price point where an asset historically stops rising due to increased selling pressure.",
            "simple": "A 'ceiling' price where the asset tends to stop climbing and sellers take profits.",
            "example": "If a stock has resistance at $100, it has struggled to rise above that price in the past.",
            "category": "Technical Analysis"
        },
        "RSI (Relative Strength Index)": {
            "definition": "A momentum indicator measuring the speed and magnitude of recent price changes on a scale of 0-100.",
            "simple": "A number from 0-100 showing if an asset is overbought (too expensive) or oversold (potentially cheap).",
            "example": "RSI above 70 suggests overbought (price might drop). RSI below 30 suggests oversold (price might rise).",
            "category": "Technical Indicators"
        },
        "MACD": {
            "definition": "Moving Average Convergence Divergence - a trend-following indicator showing the relationship between two moving averages.",
            "simple": "A tool that shows whether an asset's momentum is increasing or decreasing.",
            "example": "When MACD crosses above its signal line, it often indicates upward momentum.",
            "category": "Technical Indicators"
        },
        "Bollinger Bands": {
            "definition": "A volatility indicator consisting of a moving average with upper and lower bands based on standard deviation.",
            "simple": "Bands that show the normal price range. Prices outside the bands may be extreme.",
            "example": "When price touches the upper band, the asset might be overbought. Lower band might mean oversold.",
            "category": "Technical Indicators"
        },
        "Moving Average": {
            "definition": "The average price of an asset over a specific time period, updated as new data becomes available.",
            "simple": "A smoothed line showing the average price trend over time (like 20 days or 50 days).",
            "example": "A 50-day moving average shows the average closing price over the last 50 trading days.",
            "category": "Technical Indicators"
        },
        "Stop-Loss": {
            "definition": "An order to automatically sell an asset when it reaches a specified price to limit potential losses.",
            "simple": "A safety net that automatically sells if the price drops too much, limiting your loss.",
            "example": "Setting a 5% stop-loss means your position sells automatically if it drops 5% from your entry.",
            "category": "Risk Management"
        },
        "Take-Profit": {
            "definition": "An order to automatically sell an asset when it reaches a specified profit target.",
            "simple": "An automatic sell order that locks in your profits when the price reaches your goal.",
            "example": "A 15% take-profit means your position sells automatically when you've gained 15%.",
            "category": "Risk Management"
        },
        "Position Size": {
            "definition": "The amount of capital allocated to a single trade, typically expressed as a percentage of total portfolio.",
            "simple": "How much of your money you put into one trade.",
            "example": "If you have $10,000 and use 10% position sizing, each trade would be $1,000 maximum.",
            "category": "Risk Management"
        },
        "Portfolio": {
            "definition": "The collection of all investments held by an individual or institution.",
            "simple": "All of your investments combined - stocks, crypto, options, etc.",
            "example": "Your portfolio might contain Bitcoin, Apple stock, and some ETFs.",
            "category": "General"
        },
        "Volatility": {
            "definition": "A measure of how much and how quickly an asset's price fluctuates over time.",
            "simple": "How 'wild' or stable the price movements are. High volatility means bigger swings.",
            "example": "Cryptocurrencies often have high volatility (big price swings). Government bonds have low volatility.",
            "category": "Risk Management"
        },
        "Market Cap": {
            "definition": "The total market value of an asset, calculated by multiplying price by total shares or coins outstanding.",
            "simple": "The total value of all shares or coins - shows how 'big' a company or crypto is.",
            "example": "If a company has 1 million shares at $100 each, its market cap is $100 million.",
            "category": "General"
        },
        "Liquidity": {
            "definition": "The ease with which an asset can be bought or sold without significantly affecting its price.",
            "simple": "How easily you can buy or sell something. High liquidity means fast, easy trades.",
            "example": "Bitcoin has high liquidity (easy to trade). Small altcoins may have low liquidity (harder to sell).",
            "category": "General"
        },
        "Bull Market": {
            "definition": "A market condition characterized by rising prices and optimistic investor sentiment.",
            "simple": "When prices are generally going up and investors are optimistic.",
            "example": "A bull market might see stocks rising 20% or more over several months.",
            "category": "Market Conditions"
        },
        "Bear Market": {
            "definition": "A market condition characterized by falling prices (typically 20%+ decline) and pessimistic sentiment.",
            "simple": "When prices are generally going down and investors are worried.",
            "example": "A bear market might see stocks falling 20% or more from recent highs.",
            "category": "Market Conditions"
        },
        "Fear & Greed Index": {
            "definition": "A sentiment indicator measuring market emotion on a scale of 0 (Extreme Fear) to 100 (Extreme Greed).",
            "simple": "A number showing if investors are scared (might be time to buy) or greedy (might be time to be cautious).",
            "example": "Index at 20 suggests extreme fear (potential buying opportunity). At 80, extreme greed (potential risk).",
            "category": "Sentiment"
        },
        "SHAP Values": {
            "definition": "A method to explain AI predictions by showing how much each factor contributed to the decision.",
            "simple": "Shows which factors (like price, volume, news) most influenced the AI's recommendation.",
            "example": "SHAP might show that RSI contributed 30% to a BUY signal, while MACD contributed 25%.",
            "category": "AI/ML"
        },
        "Ensemble Model": {
            "definition": "A machine learning approach that combines multiple models to make more accurate predictions.",
            "simple": "Using multiple AI 'brains' together - like getting opinions from several experts.",
            "example": "IntelliTradeAI combines Random Forest, XGBoost, and pattern recognition for better accuracy.",
            "category": "AI/ML"
        },
        "Backtesting": {
            "definition": "Testing a trading strategy using historical data to evaluate how it would have performed.",
            "simple": "Testing the AI's strategy on past data to see how well it would have worked.",
            "example": "Backtesting might show a strategy would have earned 15% return over the past year.",
            "category": "AI/ML"
        },
        "Call Option": {
            "definition": "A contract giving the buyer the right (not obligation) to buy an asset at a specified price before expiration.",
            "simple": "A bet that a stock's price will go UP. You can buy the stock at a locked-in price.",
            "example": "A $100 call on Apple lets you buy Apple at $100 even if it rises to $120.",
            "category": "Options"
        },
        "Put Option": {
            "definition": "A contract giving the buyer the right (not obligation) to sell an asset at a specified price before expiration.",
            "simple": "A bet that a stock's price will go DOWN. You can sell at a locked-in price.",
            "example": "A $100 put on Apple lets you sell Apple at $100 even if it falls to $80.",
            "category": "Options"
        },
        "Strike Price": {
            "definition": "The price at which an option can be exercised to buy or sell the underlying asset.",
            "simple": "The locked-in price in an options contract where you can buy or sell.",
            "example": "A call with $150 strike means you can buy the stock at $150 regardless of market price.",
            "category": "Options"
        },
        "Implied Volatility (IV)": {
            "definition": "The market's forecast of how much an asset's price is expected to move, reflected in option prices.",
            "simple": "How much the market expects prices to swing - affects how expensive options are.",
            "example": "High IV before earnings means the market expects big price movement (options cost more).",
            "category": "Options"
        },
        "Greeks (Delta, Gamma, Theta, Vega)": {
            "definition": "Measurements of how option prices change relative to various factors like price, time, and volatility.",
            "simple": "Numbers that show how sensitive an option is to different changes in the market.",
            "example": "Delta 0.50 means the option gains $0.50 for every $1 the stock rises.",
            "category": "Options"
        },
        "DeFi (Decentralized Finance)": {
            "definition": "Financial services built on blockchain technology that operate without traditional intermediaries like banks.",
            "simple": "Banking and financial services that run on blockchain - no banks needed.",
            "example": "Lending your crypto to earn interest through a protocol like Aave or Compound.",
            "category": "Crypto"
        },
        "RWA (Real World Assets)": {
            "definition": "Physical assets like real estate, commodities, or bonds that are tokenized and traded on blockchain.",
            "simple": "Real things (like buildings or gold) represented as digital tokens you can trade.",
            "example": "A tokenized real estate fund lets you own a fraction of buildings through crypto tokens.",
            "category": "Crypto"
        },
        "Meme Coin": {
            "definition": "Cryptocurrencies that originated from internet memes or jokes, often with high volatility and speculative nature.",
            "simple": "Crypto coins based on internet jokes or trends. Very risky but can have big moves.",
            "example": "Dogecoin and Shiba Inu are popular meme coins that gained mainstream attention.",
            "category": "Crypto"
        },
        "Stablecoin": {
            "definition": "Cryptocurrencies designed to maintain a stable value, typically pegged to a fiat currency like the US dollar.",
            "simple": "Crypto that stays at a fixed price (usually $1). Used to store value without volatility.",
            "example": "USDT and USDC are stablecoins that aim to always equal $1.",
            "category": "Crypto"
        },
        "Layer 1": {
            "definition": "The base blockchain network that processes and finalizes transactions (like Bitcoin or Ethereum).",
            "simple": "The main blockchain network that everything else is built on.",
            "example": "Ethereum, Solana, and Avalanche are Layer 1 blockchains.",
            "category": "Crypto"
        },
        "Layer 2": {
            "definition": "Solutions built on top of Layer 1 to improve speed and reduce transaction costs.",
            "simple": "Extra layers on top of main blockchains that make transactions faster and cheaper.",
            "example": "Polygon and Arbitrum are Layer 2 solutions for Ethereum.",
            "category": "Crypto"
        },
        "ETF (Exchange-Traded Fund)": {
            "definition": "An investment fund that trades on stock exchanges and holds a basket of assets like stocks, bonds, or commodities.",
            "simple": "A single investment that contains many stocks or assets - easy diversification.",
            "example": "SPY is an ETF that tracks the S&P 500, giving you exposure to 500 companies.",
            "category": "Stocks"
        },
        "Dividend": {
            "definition": "A portion of a company's profits paid to shareholders, typically on a quarterly basis.",
            "simple": "Cash payments companies give to shareholders from their profits.",
            "example": "If a stock pays a $1 quarterly dividend and you own 100 shares, you receive $100 every quarter.",
            "category": "Stocks"
        },
        "P/E Ratio": {
            "definition": "Price-to-Earnings ratio - a valuation metric comparing stock price to earnings per share.",
            "simple": "Shows how expensive a stock is compared to its profits. Lower might mean better value.",
            "example": "A P/E of 20 means investors pay $20 for every $1 of company earnings.",
            "category": "Stocks"
        }
    }
    
    CATEGORIES = [
        "All",
        "Signals",
        "Technical Analysis", 
        "Technical Indicators",
        "Risk Management",
        "General",
        "Market Conditions",
        "Sentiment",
        "AI/ML",
        "Options",
        "Crypto",
        "Stocks"
    ]
    
    @staticmethod
    def get_term(term: str) -> Optional[Dict]:
        """Get definition for a specific term"""
        return TradingDictionary.TERMS.get(term)
    
    @staticmethod
    def search_terms(query: str) -> List[str]:
        """Search for terms matching query"""
        query_lower = query.lower()
        matches = []
        for term, data in TradingDictionary.TERMS.items():
            if (query_lower in term.lower() or 
                query_lower in data["definition"].lower() or
                query_lower in data["simple"].lower()):
                matches.append(term)
        return matches
    
    @staticmethod
    def get_terms_by_category(category: str) -> Dict[str, Dict]:
        """Get all terms in a specific category"""
        if category == "All":
            return TradingDictionary.TERMS
        return {
            term: data for term, data in TradingDictionary.TERMS.items()
            if data["category"] == category
        }
    
    @staticmethod
    def display_term_popup(term: str):
        """Display a term definition in a popup/expander"""
        data = TradingDictionary.get_term(term)
        if data:
            with st.expander(f"What is '{term}'?"):
                st.markdown(f"**Simple Explanation:** {data['simple']}")
                st.markdown(f"**Technical Definition:** {data['definition']}")
                st.markdown(f"**Example:** {data['example']}")
                st.caption(f"Category: {data['category']}")
    
    @staticmethod
    def display_dictionary_page():
        """Display full dictionary interface"""
        st.header("Trading Dictionary")
        st.markdown("Search or browse definitions for trading terms and concepts.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input(
                "Search terms",
                placeholder="Type to search (e.g., 'RSI', 'stop-loss', 'crypto')",
                key="dict_search"
            )
        
        with col2:
            category_filter = st.selectbox(
                "Filter by category",
                TradingDictionary.CATEGORIES,
                key="dict_category"
            )
        
        if search_query:
            matching_terms = TradingDictionary.search_terms(search_query)
            terms_to_display = {
                term: TradingDictionary.TERMS[term] 
                for term in matching_terms
            }
            st.caption(f"Found {len(matching_terms)} matching terms")
        else:
            terms_to_display = TradingDictionary.get_terms_by_category(category_filter)
        
        if not terms_to_display:
            st.info("No terms found. Try a different search or category.")
            return
        
        for term, data in sorted(terms_to_display.items()):
            with st.expander(f"{term}", expanded=False):
                st.markdown(f"**In Simple Terms:**")
                st.info(data["simple"])
                
                st.markdown(f"**Technical Definition:**")
                st.markdown(data["definition"])
                
                st.markdown(f"**Example:**")
                st.success(data["example"])
                
                st.caption(f"Category: {data['category']}")
    
    @staticmethod
    def get_contextual_help(context: str) -> str:
        """Get contextual help text based on current screen/feature"""
        context_help = {
            "trading_signals": "Trading signals (BUY/SELL/HOLD) are AI-generated recommendations based on our tri-signal fusion analysis.",
            "options_chain": "The options chain shows available call and put options with their prices, Greeks, and implied volatility.",
            "fear_greed": "The Fear & Greed Index measures market sentiment from 0 (extreme fear) to 100 (extreme greed).",
            "backtesting": "Backtesting simulates how a trading strategy would have performed using historical data.",
            "auto_trading": "Automatic trading lets the AI execute trades on your behalf based on your configured rules.",
            "technical_analysis": "Technical analysis uses price charts and indicators to identify trading opportunities.",
            "sentiment_analysis": "Sentiment analysis gauges market mood from news and social media to predict price movements."
        }
        return context_help.get(context, "")
    
    @staticmethod
    def display_inline_definition(term: str) -> None:
        """Display a small inline definition tooltip"""
        data = TradingDictionary.get_term(term)
        if data:
            st.markdown(f"**{term}**: {data['simple']}")
