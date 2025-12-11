"""
IntelliTradeAI User Onboarding and Risk Assessment
Personalized trading plans based on investment amount and risk tolerance
"""
import streamlit as st
from typing import Dict, Optional, List
from datetime import datetime
from enum import Enum

class RiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    GROWTH = "growth"
    AGGRESSIVE = "aggressive"
    SPECULATIVE = "speculative"

class TradingPlan:
    """Represents a personalized trading plan"""
    
    PLANS = {
        RiskLevel.CONSERVATIVE: {
            "name": "Capital Preservation Plan",
            "description": "Focus on protecting your investment with steady, low-risk growth",
            "target_return": "4-8% annually",
            "max_drawdown": "5%",
            "time_horizon": "3-5+ years",
            "stock_allocation": 40,
            "crypto_allocation": 5,
            "stablecoin_allocation": 40,
            "cash_allocation": 15,
            "recommended_stocks": {
                "sectors": ["Consumer Staples", "Utilities", "Healthcare", "Dividend Aristocrats"],
                "examples": ["JNJ", "WMT", "KO", "PG", "VZ"],
                "focus": "Blue-chip dividend-paying stocks with low volatility"
            },
            "recommended_crypto": {
                "sectors": ["Stablecoins", "Large Cap Only"],
                "examples": ["USDC", "USDT", "BTC", "ETH"],
                "focus": "Only top 2 cryptocurrencies by market cap, heavy stablecoin allocation"
            },
            "options_allowed": False,
            "auto_trade_settings": {
                "min_confidence": 85,
                "max_position_percent": 5,
                "stop_loss_percent": 3,
                "take_profit_percent": 8,
                "max_daily_trades": 3
            },
            "signal_filter": "Only show HIGH confidence BUY signals on low-volatility assets"
        },
        RiskLevel.MODERATE: {
            "name": "Balanced Growth Plan",
            "description": "Balance between growth potential and risk management",
            "target_return": "8-15% annually",
            "max_drawdown": "10%",
            "time_horizon": "2-4 years",
            "stock_allocation": 50,
            "crypto_allocation": 15,
            "stablecoin_allocation": 20,
            "cash_allocation": 15,
            "recommended_stocks": {
                "sectors": ["Technology", "Healthcare", "Consumer Discretionary", "Financials", "ETFs"],
                "examples": ["AAPL", "MSFT", "JPM", "V", "SPY", "QQQ"],
                "focus": "Mix of growth and value stocks with strong fundamentals"
            },
            "recommended_crypto": {
                "sectors": ["Large Cap", "Layer 1", "DeFi Blue Chips"],
                "examples": ["BTC", "ETH", "SOL", "AVAX", "LINK"],
                "focus": "Established cryptocurrencies with proven track records"
            },
            "options_allowed": True,
            "options_strategy": "Covered calls and cash-secured puts only",
            "auto_trade_settings": {
                "min_confidence": 75,
                "max_position_percent": 8,
                "stop_loss_percent": 5,
                "take_profit_percent": 12,
                "max_daily_trades": 5
            },
            "signal_filter": "Show MEDIUM-HIGH confidence signals on established assets"
        },
        RiskLevel.GROWTH: {
            "name": "Growth Accelerator Plan",
            "description": "Focus on capital appreciation with managed risk exposure",
            "target_return": "15-30% annually",
            "max_drawdown": "20%",
            "time_horizon": "1-3 years",
            "stock_allocation": 45,
            "crypto_allocation": 35,
            "stablecoin_allocation": 10,
            "cash_allocation": 10,
            "recommended_stocks": {
                "sectors": ["Technology", "Semiconductors", "AI/Cloud", "Biotech", "Growth ETFs"],
                "examples": ["NVDA", "AMD", "TSLA", "META", "NFLX", "CRM"],
                "focus": "High-growth technology and innovation leaders"
            },
            "recommended_crypto": {
                "sectors": ["Large Cap", "Layer 1", "Layer 2", "DeFi", "AI Tokens"],
                "examples": ["BTC", "ETH", "SOL", "AVAX", "MATIC", "NEAR", "FET", "RNDR"],
                "focus": "Mix of established and emerging blockchain projects"
            },
            "options_allowed": True,
            "options_strategy": "Directional calls/puts, spreads",
            "auto_trade_settings": {
                "min_confidence": 70,
                "max_position_percent": 12,
                "stop_loss_percent": 8,
                "take_profit_percent": 20,
                "max_daily_trades": 8
            },
            "signal_filter": "Show all confidence levels with sector diversification"
        },
        RiskLevel.AGGRESSIVE: {
            "name": "High Growth Plan",
            "description": "Maximize growth potential with higher risk tolerance",
            "target_return": "30-60% annually",
            "max_drawdown": "35%",
            "time_horizon": "6 months - 2 years",
            "stock_allocation": 35,
            "crypto_allocation": 50,
            "stablecoin_allocation": 5,
            "cash_allocation": 10,
            "recommended_stocks": {
                "sectors": ["High-Growth Tech", "Biotech", "Small Cap Growth", "SPACs", "Leveraged ETFs"],
                "examples": ["NVDA", "AMD", "TSLA", "COIN", "MSTR", "SQ"],
                "focus": "High-beta stocks with significant upside potential"
            },
            "recommended_crypto": {
                "sectors": ["Layer 1", "Layer 2", "DeFi", "AI Tokens", "Gaming/Metaverse", "RWA"],
                "examples": ["SOL", "AVAX", "INJ", "FET", "RNDR", "IMX", "SAND", "ONDO"],
                "focus": "Emerging protocols with high growth potential"
            },
            "options_allowed": True,
            "options_strategy": "Full options strategies including leveraged positions",
            "auto_trade_settings": {
                "min_confidence": 65,
                "max_position_percent": 15,
                "stop_loss_percent": 12,
                "take_profit_percent": 30,
                "max_daily_trades": 12
            },
            "signal_filter": "All signals with emphasis on momentum plays"
        },
        RiskLevel.SPECULATIVE: {
            "name": "Maximum Opportunity Plan",
            "description": "High-risk, high-reward strategy for experienced traders",
            "target_return": "60%+ annually (highly variable)",
            "max_drawdown": "50%+",
            "time_horizon": "Short-term (days to months)",
            "stock_allocation": 25,
            "crypto_allocation": 65,
            "stablecoin_allocation": 5,
            "cash_allocation": 5,
            "recommended_stocks": {
                "sectors": ["Momentum Plays", "Meme Stocks", "Biotech Catalysts", "Options Heavy"],
                "examples": ["TSLA", "COIN", "MSTR", "GME", "Biotech catalysts"],
                "focus": "High-volatility opportunities with significant leverage"
            },
            "recommended_crypto": {
                "sectors": ["Mid Cap", "Meme Coins", "New Listings", "AI Tokens", "Narrative Plays"],
                "examples": ["PEPE", "SHIB", "DOGE", "BONK", "WIF", "New launches"],
                "focus": "Trending tokens and narrative-driven opportunities"
            },
            "options_allowed": True,
            "options_strategy": "All strategies including high-risk leveraged plays",
            "auto_trade_settings": {
                "min_confidence": 60,
                "max_position_percent": 20,
                "stop_loss_percent": 15,
                "take_profit_percent": 50,
                "max_daily_trades": 20
            },
            "signal_filter": "All signals including high-risk opportunities",
            "warning": "This plan involves substantial risk of loss. Only invest what you can afford to lose entirely."
        }
    }

class UserOnboarding:
    """Handles user onboarding and risk assessment"""
    
    INVESTMENT_TIERS = [
        {"min": 0, "max": 1000, "tier": "Starter", "description": "Learning the basics"},
        {"min": 1000, "max": 5000, "tier": "Beginner", "description": "Building your foundation"},
        {"min": 5000, "max": 25000, "tier": "Intermediate", "description": "Growing your portfolio"},
        {"min": 25000, "max": 100000, "tier": "Advanced", "description": "Serious investing"},
        {"min": 100000, "max": float('inf'), "tier": "Professional", "description": "Sophisticated strategies"}
    ]
    
    RISK_QUESTIONS = [
        {
            "id": "investment_goal",
            "question": "What is your primary investment goal?",
            "options": [
                {"text": "Preserve my capital and avoid losses", "score": 1},
                {"text": "Generate steady income with minimal risk", "score": 2},
                {"text": "Grow my wealth over time with balanced risk", "score": 3},
                {"text": "Maximize growth even if it means higher risk", "score": 4},
                {"text": "Seek the highest possible returns regardless of risk", "score": 5}
            ]
        },
        {
            "id": "loss_reaction",
            "question": "If your portfolio dropped 20% in one month, what would you do?",
            "options": [
                {"text": "Sell everything immediately to prevent further losses", "score": 1},
                {"text": "Sell some positions and move to safer investments", "score": 2},
                {"text": "Hold and wait for recovery", "score": 3},
                {"text": "Buy more at the lower prices", "score": 4},
                {"text": "Significantly increase my position to maximize the opportunity", "score": 5}
            ]
        },
        {
            "id": "time_horizon",
            "question": "How long do you plan to keep your money invested?",
            "options": [
                {"text": "Less than 1 year - I might need it soon", "score": 1},
                {"text": "1-2 years", "score": 2},
                {"text": "3-5 years", "score": 3},
                {"text": "5-10 years", "score": 4},
                {"text": "10+ years - I'm investing for the long term", "score": 5}
            ]
        },
        {
            "id": "experience",
            "question": "How would you describe your investing experience?",
            "options": [
                {"text": "I'm completely new to investing", "score": 1},
                {"text": "I've invested in basic products like savings or CDs", "score": 2},
                {"text": "I have some experience with stocks and/or crypto", "score": 3},
                {"text": "I actively trade and understand market dynamics", "score": 4},
                {"text": "I'm an experienced trader familiar with advanced strategies", "score": 5}
            ]
        },
        {
            "id": "income_stability",
            "question": "How stable is your current income?",
            "options": [
                {"text": "Very unstable - I'm not sure about future income", "score": 1},
                {"text": "Somewhat unstable with variable income", "score": 2},
                {"text": "Fairly stable with occasional variations", "score": 3},
                {"text": "Very stable with predictable income", "score": 4},
                {"text": "Multiple income sources, very secure", "score": 5}
            ]
        },
        {
            "id": "crypto_comfort",
            "question": "How comfortable are you with cryptocurrency investments?",
            "options": [
                {"text": "Not comfortable - I prefer to avoid crypto entirely", "score": 1},
                {"text": "Slightly curious but very cautious", "score": 2},
                {"text": "Comfortable with small crypto positions", "score": 3},
                {"text": "Very comfortable with significant crypto exposure", "score": 4},
                {"text": "Crypto-first mindset - I prefer crypto over traditional assets", "score": 5}
            ]
        },
        {
            "id": "volatility_tolerance",
            "question": "How would you react to seeing daily swings of 5-10% in your portfolio?",
            "options": [
                {"text": "Very stressed - I check my portfolio constantly and worry", "score": 1},
                {"text": "Uncomfortable but manageable", "score": 2},
                {"text": "Neutral - it's part of investing", "score": 3},
                {"text": "Excited about opportunities", "score": 4},
                {"text": "Thrilled - I actively seek volatile investments", "score": 5}
            ]
        }
    ]
    
    @staticmethod
    def get_investment_tier(amount: float) -> Dict:
        """Determine investment tier based on amount"""
        for tier in UserOnboarding.INVESTMENT_TIERS:
            if tier["min"] <= amount < tier["max"]:
                return tier
        return UserOnboarding.INVESTMENT_TIERS[-1]
    
    @staticmethod
    def calculate_risk_level(scores: List[int]) -> RiskLevel:
        """Calculate risk level from questionnaire scores"""
        avg_score = sum(scores) / len(scores)
        
        if avg_score <= 1.5:
            return RiskLevel.CONSERVATIVE
        elif avg_score <= 2.5:
            return RiskLevel.MODERATE
        elif avg_score <= 3.5:
            return RiskLevel.GROWTH
        elif avg_score <= 4.2:
            return RiskLevel.AGGRESSIVE
        else:
            return RiskLevel.SPECULATIVE
    
    @staticmethod
    def get_personalized_plan(risk_level: RiskLevel, investment_amount: float) -> Dict:
        """Get personalized trading plan based on risk level and amount"""
        base_plan = TradingPlan.PLANS[risk_level].copy()
        tier = UserOnboarding.get_investment_tier(investment_amount)
        
        base_plan["investment_amount"] = investment_amount
        base_plan["investment_tier"] = tier["tier"]
        base_plan["tier_description"] = tier["description"]
        
        if investment_amount < 1000:
            base_plan["special_notes"] = [
                "With a smaller starting amount, focus on learning and paper trading first",
                "Consider commission-free platforms to maximize your returns",
                "Start with just 1-2 assets to learn the patterns well"
            ]
        elif investment_amount < 5000:
            base_plan["special_notes"] = [
                "Diversify across 3-5 assets to spread risk",
                "Focus on building consistent habits rather than quick gains",
                "Consider dollar-cost averaging into positions"
            ]
        elif investment_amount < 25000:
            base_plan["special_notes"] = [
                "You have enough to build a properly diversified portfolio",
                "Consider allocating across different sectors and asset classes",
                "Options strategies become more viable at this level"
            ]
        elif investment_amount < 100000:
            base_plan["special_notes"] = [
                "Advanced strategies and diversification are fully available",
                "Consider tax-efficient positioning across accounts",
                "Full access to options and sophisticated hedging strategies"
            ]
        else:
            base_plan["special_notes"] = [
                "Consider working with a tax professional for optimization",
                "Advanced portfolio strategies including alternatives may be suitable",
                "Full institutional-level analysis and tools available"
            ]
        
        return base_plan
    
    @staticmethod
    def display_onboarding_survey() -> Optional[Dict]:
        """Display the complete onboarding survey"""
        st.header("Welcome to IntelliTradeAI")
        st.markdown("""
        Let's create your personalized trading plan. This short survey will help us understand 
        your investment goals, risk tolerance, and preferences so we can provide tailored recommendations.
        """)
        
        st.markdown("---")
        
        st.subheader("Step 1: Investment Amount")
        investment_amount = st.number_input(
            "How much money do you plan to invest?",
            min_value=0.0,
            max_value=10000000.0,
            value=10000.0,
            step=1000.0,
            format="%.2f",
            help="Enter the total amount you're comfortable investing"
        )
        
        tier = UserOnboarding.get_investment_tier(investment_amount)
        st.info(f"**Investment Tier**: {tier['tier']} - {tier['description']}")
        
        st.markdown("---")
        
        st.subheader("Step 2: Risk Assessment")
        st.markdown("Please answer these questions honestly to help us determine your risk tolerance.")
        
        scores = []
        for i, q in enumerate(UserOnboarding.RISK_QUESTIONS):
            st.markdown(f"**{i+1}. {q['question']}**")
            options = [opt["text"] for opt in q["options"]]
            selected = st.radio(
                "Select one:",
                options,
                key=f"risk_q_{q['id']}",
                label_visibility="collapsed"
            )
            selected_score = next(
                opt["score"] for opt in q["options"] if opt["text"] == selected
            )
            scores.append(selected_score)
            st.markdown("")
        
        st.markdown("---")
        
        st.subheader("Step 3: Trading Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            interested_in_stocks = st.checkbox("I want to trade stocks", value=True)
            interested_in_crypto = st.checkbox("I want to trade cryptocurrency", value=True)
            interested_in_options = st.checkbox("I want to trade options", value=False)
        
        with col2:
            prefer_auto = st.radio(
                "Trading mode preference:",
                ["Manual (AI recommendations, I decide)", 
                 "Automatic (AI executes trades for me)",
                 "Hybrid (Start manual, maybe go automatic later)"],
                key="trading_mode_pref"
            )
        
        st.markdown("---")
        
        if st.button("Generate My Personalized Plan", type="primary", use_container_width=True):
            risk_level = UserOnboarding.calculate_risk_level(scores)
            plan = UserOnboarding.get_personalized_plan(risk_level, investment_amount)
            
            plan["preferences"] = {
                "stocks": interested_in_stocks,
                "crypto": interested_in_crypto,
                "options": interested_in_options,
                "trading_mode": prefer_auto
            }
            plan["risk_level"] = risk_level.value
            plan["assessment_scores"] = scores
            plan["completed_at"] = datetime.now().isoformat()
            
            return plan
        
        return None
    
    @staticmethod
    def display_trading_plan(plan: Dict):
        """Display the generated trading plan"""
        st.header(f"Your Personalized Trading Plan")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Investment Amount", f"${plan['investment_amount']:,.2f}")
        with col2:
            st.metric("Investor Tier", plan["investment_tier"])
        with col3:
            risk_display = plan["risk_level"].replace("_", " ").title()
            st.metric("Risk Profile", risk_display)
        
        st.markdown("---")
        
        st.subheader(plan["name"])
        st.markdown(plan["description"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Target Returns**")
            st.success(plan["target_return"])
            
            st.markdown("**Maximum Drawdown Tolerance**")
            st.warning(plan["max_drawdown"])
            
            st.markdown("**Recommended Time Horizon**")
            st.info(plan["time_horizon"])
        
        with col2:
            st.markdown("**Recommended Allocation**")
            allocation_data = {
                "Stocks": plan["stock_allocation"],
                "Crypto": plan["crypto_allocation"],
                "Stablecoins": plan["stablecoin_allocation"],
                "Cash": plan["cash_allocation"]
            }
            for asset, pct in allocation_data.items():
                st.progress(pct/100, text=f"{asset}: {pct}%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Recommended Stocks")
            stock_rec = plan["recommended_stocks"]
            st.markdown(f"**Focus Areas:** {', '.join(stock_rec['sectors'])}")
            st.markdown(f"**Example Tickers:** {', '.join(stock_rec['examples'])}")
            st.caption(stock_rec["focus"])
        
        with col2:
            st.subheader("Recommended Crypto")
            crypto_rec = plan["recommended_crypto"]
            st.markdown(f"**Focus Areas:** {', '.join(crypto_rec['sectors'])}")
            st.markdown(f"**Example Tokens:** {', '.join(crypto_rec['examples'])}")
            st.caption(crypto_rec["focus"])
        
        st.markdown("---")
        
        if plan.get("options_allowed"):
            st.subheader("Options Trading")
            st.success(f"Options trading is enabled for your plan")
            if "options_strategy" in plan:
                st.markdown(f"**Recommended Strategies:** {plan['options_strategy']}")
        else:
            st.subheader("Options Trading")
            st.warning("Options trading is not recommended for your risk profile")
        
        st.markdown("---")
        
        st.subheader("Automatic Trading Settings")
        settings = plan["auto_trade_settings"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min Confidence", f"{settings['min_confidence']}%")
            st.metric("Max Position Size", f"{settings['max_position_percent']}%")
        with col2:
            st.metric("Stop Loss", f"{settings['stop_loss_percent']}%")
            st.metric("Take Profit", f"{settings['take_profit_percent']}%")
        with col3:
            st.metric("Max Daily Trades", settings['max_daily_trades'])
        
        st.markdown("---")
        
        if "special_notes" in plan:
            st.subheader("Personalized Recommendations")
            for note in plan["special_notes"]:
                st.markdown(f"- {note}")
        
        if "warning" in plan:
            st.error(f"**Important Warning:** {plan['warning']}")
        
        st.markdown("---")
        
        st.subheader("Signal Filtering")
        st.info(plan["signal_filter"])
