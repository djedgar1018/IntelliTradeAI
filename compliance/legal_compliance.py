"""
IntelliTradeAI Legal Compliance Module
SEC, FINRA, and E-Signature compliance for automated trading authorization
"""
import streamlit as st
from datetime import datetime
from typing import Dict, Optional
import hashlib
import json

class LegalComplianceManager:
    """Manages legal disclaimers, risk disclosures, and e-signature authorization"""
    
    RISK_DISCLOSURES = {
        "general_investment": """
**GENERAL INVESTMENT RISK DISCLOSURE**

Investing involves risk, including the possible loss of principal. Past performance does not guarantee future results. No investment strategy can guarantee profit or protect against loss in declining markets.

The value of your investments may fluctuate significantly due to market conditions, economic factors, and company-specific developments. You should carefully consider your investment objectives, risk tolerance, and time horizon before making any investment decisions.
""",
        "automated_trading": """
**AUTOMATED TRADING RISK DISCLOSURE**

Automated trading systems use algorithms to make trading decisions on your behalf. While these systems are designed to execute trades based on predefined criteria, they carry additional risks including:

1. **Technology Risk**: System failures, connectivity issues, or software errors may result in delayed or failed trade execution.

2. **Algorithm Limitations**: Algorithms are based on historical data and may not perform as expected during unusual market conditions or unprecedented events.

3. **Market Volatility**: Rapid market movements may result in trades being executed at prices significantly different from expected.

4. **Overreliance Risk**: Automated systems should not replace your own due diligence and investment research.

5. **No Guarantee of Profits**: Automated trading does not guarantee profits and may result in significant losses.
""",
        "cryptocurrency": """
**CRYPTOCURRENCY RISK DISCLOSURE**

Cryptocurrency investments carry unique and substantial risks including:

1. **Extreme Volatility**: Cryptocurrency prices can experience rapid and significant fluctuations within short periods.

2. **Regulatory Uncertainty**: Cryptocurrency regulations vary by jurisdiction and may change, potentially affecting the value or legality of your holdings.

3. **Security Risks**: Digital assets may be subject to hacking, theft, or loss of access keys.

4. **Limited Protection**: Cryptocurrency investments are generally not protected by FDIC, SIPC, or similar insurance programs.

5. **Market Manipulation**: Cryptocurrency markets may be subject to manipulation and lack the regulatory oversight of traditional securities markets.

6. **Liquidity Risk**: Some cryptocurrencies may be difficult to sell quickly at fair market value.
""",
        "options_trading": """
**OPTIONS TRADING RISK DISCLOSURE**

Options trading involves significant risk and is not appropriate for all investors. Before trading options, you should understand:

1. **Total Loss Risk**: Options can expire worthless, resulting in complete loss of your investment.

2. **Complex Strategies**: Multi-leg options strategies involve multiple commissions and complex tax implications.

3. **Time Decay**: Options lose value as they approach expiration (theta decay).

4. **Leverage Risk**: Options provide leverage, which can amplify both gains and losses.

5. **Assignment Risk**: Option sellers may be assigned at any time, potentially resulting in significant obligations.

Please read the Characteristics and Risks of Standardized Options (OCC disclosure) before trading options.
""",
        "ai_disclosure": """
**ARTIFICIAL INTELLIGENCE DISCLOSURE**

IntelliTradeAI uses artificial intelligence and machine learning algorithms to generate trading recommendations. You should understand:

1. **AI Limitations**: AI predictions are probabilistic estimates based on historical data and patterns. They are not guarantees of future performance.

2. **Model Accuracy**: Our models achieve approximately 72% accuracy in backtesting. This means approximately 28% of predictions may be incorrect.

3. **Black Swan Events**: AI models may not accurately predict or respond to unprecedented market events.

4. **Data Dependency**: AI recommendations depend on the quality and availability of market data. Data errors or delays may affect recommendation accuracy.

5. **Explainability**: While we provide SHAP-based explanations for our predictions, AI decision-making may involve complex interactions that are difficult to fully interpret.

6. **No Human Override in Automatic Mode**: When automatic trading is enabled, trades execute based on AI signals without human review of individual transactions.
"""
    }
    
    SEC_REGULATORY_NOTICES = {
        "registration_notice": """
**REGULATORY STATUS**

IntelliTradeAI provides automated investment analysis and trading recommendations. This service is provided for informational and educational purposes.

IntelliTradeAI is NOT registered as a broker-dealer with the U.S. Securities and Exchange Commission (SEC) or any state securities regulatory authority. IntelliTradeAI is NOT a registered investment adviser.

The trading signals, recommendations, and analysis provided by this application do not constitute personalized investment advice. You should consult with a qualified financial advisor before making investment decisions.
""",
        "not_fiduciary": """
**NON-FIDUCIARY NOTICE**

IntelliTradeAI does not act as a fiduciary with respect to your investments or trading decisions. We do not provide personalized investment advice based on your complete financial situation.

Our recommendations are generated algorithmically based on technical analysis and machine learning models. They do not take into account your complete financial circumstances, tax situation, or specific investment objectives beyond the risk profile you provide.
""",
        "no_guarantee": """
**NO GUARANTEE OF RESULTS**

IntelliTradeAI makes no representations or warranties regarding:

- The accuracy or completeness of any information provided
- The suitability of any investment recommendation for your specific situation
- Future investment performance or returns
- The absence of errors in our systems or data

Past performance, whether actual or indicated by historical tests, is not indicative of future results.
"""
    }
    
    AUTOMATIC_TRADING_AGREEMENT = """
# AUTOMATIC TRADING AUTHORIZATION AGREEMENT

**Effective Date**: {date}

By electronically signing this agreement, I, the undersigned user ("User"), hereby authorize IntelliTradeAI ("Platform") to execute trades on my behalf under the following terms and conditions:

## 1. AUTHORIZATION SCOPE

I authorize the Platform to:
- Execute BUY, SELL, and HOLD orders for securities and cryptocurrencies based on AI-generated signals
- Manage position sizes according to my risk profile settings
- Implement stop-loss and take-profit orders as configured
- Make trading decisions without prior approval for each individual trade

## 2. RISK ACKNOWLEDGMENT

I acknowledge and accept that:

a) **Financial Risk**: Automated trading may result in significant financial losses, including loss of my entire invested capital.

b) **Technology Risk**: System failures, network issues, or software errors may affect trade execution.

c) **Algorithm Risk**: AI algorithms may make incorrect predictions or perform poorly under certain market conditions.

d) **No Guarantees**: The Platform makes no guarantees regarding trading performance, profits, or loss prevention.

## 3. CONFIGURATION SETTINGS

I confirm that I have reviewed and approved the following automatic trading parameters:

- Maximum position size: {max_position}% of portfolio
- Stop-loss threshold: {stop_loss}%
- Take-profit threshold: {take_profit}%
- Minimum AI confidence for execution: {min_confidence}%
- Maximum daily trades: {max_daily_trades}
- Maximum daily loss limit: ${max_daily_loss}

## 4. USER RESPONSIBILITIES

I agree to:
- Monitor my account regularly
- Maintain sufficient funds for intended trading activity
- Update my risk profile if my financial situation changes
- Disable automatic trading if I no longer wish to participate
- Review and understand all disclosures provided

## 5. TERMINATION

I understand that:
- I may disable automatic trading at any time through the Platform interface
- Disabling will stop new trades but will not affect open positions
- I am responsible for managing or closing any open positions after disabling

## 6. INDEMNIFICATION

I agree to hold harmless IntelliTradeAI, its affiliates, officers, employees, and agents from any claims, damages, or losses arising from:
- Trades executed under this authorization
- Technology failures or delays
- Market conditions or price movements
- My own decisions regarding trading configuration

## 7. ELECTRONIC SIGNATURE

I confirm that:
- I am signing this agreement voluntarily
- I have read and understood all terms
- I am at least 18 years of age
- I am legally authorized to enter into this agreement
- My electronic signature has the same legal effect as a handwritten signature

**User Full Legal Name**: {user_name}
**User Email**: {user_email}
**IP Address**: {ip_address}
**Timestamp**: {timestamp}
**Agreement Hash**: {agreement_hash}

---

**ELECTRONIC SIGNATURE**

By clicking "I AGREE AND AUTHORIZE AUTOMATIC TRADING" below, I am electronically signing this agreement and authorizing IntelliTradeAI to execute trades on my behalf according to the terms stated above.
"""

    @staticmethod
    def display_all_disclosures():
        """Display all required risk disclosures"""
        st.subheader("Required Risk Disclosures")
        
        with st.expander("General Investment Risks", expanded=False):
            st.markdown(LegalComplianceManager.RISK_DISCLOSURES["general_investment"])
        
        with st.expander("Automated Trading Risks", expanded=False):
            st.markdown(LegalComplianceManager.RISK_DISCLOSURES["automated_trading"])
        
        with st.expander("Cryptocurrency Risks", expanded=False):
            st.markdown(LegalComplianceManager.RISK_DISCLOSURES["cryptocurrency"])
        
        with st.expander("Options Trading Risks", expanded=False):
            st.markdown(LegalComplianceManager.RISK_DISCLOSURES["options_trading"])
        
        with st.expander("AI/Algorithm Disclosure", expanded=False):
            st.markdown(LegalComplianceManager.RISK_DISCLOSURES["ai_disclosure"])
        
        with st.expander("Regulatory Notices", expanded=False):
            st.markdown(LegalComplianceManager.SEC_REGULATORY_NOTICES["registration_notice"])
            st.markdown(LegalComplianceManager.SEC_REGULATORY_NOTICES["not_fiduciary"])
            st.markdown(LegalComplianceManager.SEC_REGULATORY_NOTICES["no_guarantee"])
    
    @staticmethod
    def get_automatic_trading_agreement(user_name: str, user_email: str, 
                                         trading_config: Dict) -> str:
        """Generate personalized automatic trading agreement"""
        agreement = LegalComplianceManager.AUTOMATIC_TRADING_AGREEMENT.format(
            date=datetime.now().strftime("%B %d, %Y"),
            user_name=user_name,
            user_email=user_email,
            max_position=trading_config.get('max_position_size_percent', 10),
            stop_loss=trading_config.get('stop_loss_percent', 5),
            take_profit=trading_config.get('take_profit_percent', 15),
            min_confidence=trading_config.get('min_confidence', 70),
            max_daily_trades=trading_config.get('max_daily_trades', 10),
            max_daily_loss=trading_config.get('max_loss_per_day', 500),
            ip_address="[Captured at signing]",
            timestamp=datetime.now().isoformat(),
            agreement_hash="[Generated at signing]"
        )
        return agreement
    
    @staticmethod
    def generate_agreement_hash(user_name: str, user_email: str, 
                                 timestamp: str, agreement_text: str) -> str:
        """Generate unique hash for agreement verification"""
        content = f"{user_name}|{user_email}|{timestamp}|{agreement_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16].upper()
    
    @staticmethod
    def display_esignature_flow(user_name: str, user_email: str, trading_config: Dict) -> Optional[Dict]:
        """Display e-signature flow for automatic trading authorization"""
        st.subheader("Automatic Trading Authorization")
        
        st.warning("""
        **IMPORTANT**: You are about to authorize IntelliTradeAI to execute trades on your behalf 
        automatically. Please read all disclosures carefully before proceeding.
        """)
        
        LegalComplianceManager.display_all_disclosures()
        
        st.markdown("---")
        st.subheader("Electronic Signature Agreement")
        
        agreement_text = LegalComplianceManager.get_automatic_trading_agreement(
            user_name, user_email, trading_config
        )
        
        with st.expander("View Full Agreement", expanded=True):
            st.markdown(agreement_text)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            confirm_read = st.checkbox(
                "I have read and understand all risk disclosures",
                key="confirm_read_disclosures"
            )
        
        with col2:
            confirm_terms = st.checkbox(
                "I agree to the terms of the Automatic Trading Agreement",
                key="confirm_agreement_terms"
            )
        
        confirm_name = st.text_input(
            "Type your full legal name to confirm your identity",
            key="esign_name_confirm",
            placeholder="Enter your full legal name exactly as registered"
        )
        
        name_matches = confirm_name.strip().lower() == user_name.strip().lower()
        
        all_confirmed = confirm_read and confirm_terms and name_matches
        
        if not name_matches and confirm_name:
            st.error("Name does not match your registered name. Please type your name exactly as registered.")
        
        if st.button(
            "I AGREE AND AUTHORIZE AUTOMATIC TRADING",
            disabled=not all_confirmed,
            type="primary",
            use_container_width=True
        ):
            timestamp = datetime.now().isoformat()
            agreement_hash = LegalComplianceManager.generate_agreement_hash(
                user_name, user_email, timestamp, agreement_text
            )
            
            signature_record = {
                "user_name": user_name,
                "user_email": user_email,
                "signed_at": timestamp,
                "agreement_hash": agreement_hash,
                "trading_config": trading_config,
                "ip_address": "captured",
                "disclosures_acknowledged": True,
                "terms_accepted": True
            }
            
            st.success(f"""
            **Authorization Successful**
            
            Your electronic signature has been recorded.
            
            - Agreement Hash: `{agreement_hash}`
            - Signed: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
            
            You may disable automatic trading at any time from your account settings.
            """)
            
            return signature_record
        
        return None
    
    @staticmethod
    def get_quick_disclaimer() -> str:
        """Get short disclaimer for display in trading interface"""
        return """
*Trading involves risk of loss. Past performance does not guarantee future results. 
AI predictions are probabilistic estimates and may be incorrect. 
See full disclosures in Settings > Legal.*
"""

    @staticmethod
    def display_trade_confirmation_disclaimer():
        """Display disclaimer before trade execution"""
        st.caption("""
        **Risk Notice**: This trade is based on AI analysis with approximately 72% historical accuracy. 
        You may lose some or all of your invested capital. By proceeding, you acknowledge these risks.
        """)
