"""
AI Trading Intelligence Module
Provides intelligent trading recommendations with explanations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
from typing import Dict, List, Tuple, Optional

class TradingIntelligence:
    """
    Intelligent trading advisor that provides clear buy/sell/hold/DCA recommendations
    with detailed explanations in plain English
    """
    
    def __init__(self):
        self.confidence_levels = {
            'very_high': 0.85,
            'high': 0.75,
            'medium': 0.65,
            'low': 0.55,
            'very_low': 0.45
        }
        
    def analyze_asset(self, symbol: str, data: pd.DataFrame, model=None, model_proba=None) -> Dict:
        """
        Comprehensive asset analysis with trading recommendations
        
        Returns:
        - Trading decision (BUY, SELL, HOLD, DCA_IN, DCA_OUT)
        - Confidence level
        - Detailed explanation
        - Risk assessment
        - Key metrics
        """
        
        if data is None or len(data) == 0:
            return self._create_no_data_response(symbol)
            
        # Current price analysis
        current_price = float(data['close'].iloc[-1])
        price_change_24h = self._calculate_price_change(data)
        
        # Technical analysis
        technical_signals = self._analyze_technical_indicators(data)
        
        # Model-based prediction if available
        model_signal = None
        if model is not None and model_proba is not None:
            model_signal = self._analyze_model_prediction(model_proba, model)
        
        # Market context
        market_context = self._analyze_market_context(data)
        
        # Generate final recommendation
        recommendation = self._generate_recommendation(
            technical_signals, model_signal, market_context, current_price, price_change_24h
        )
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'price_change_24h': price_change_24h,
            'recommendation': recommendation,
            'analysis_timestamp': datetime.now().isoformat(),
            'technical_analysis': technical_signals,
            'model_analysis': model_signal,
            'market_context': market_context
        }
    
    def _calculate_price_change(self, data: pd.DataFrame) -> float:
        """Calculate 24h price change percentage"""
        if len(data) < 2:
            return 0.0
        
        current = float(data['close'].iloc[-1])
        previous = float(data['close'].iloc[-2])
        return ((current - previous) / previous) * 100
    
    def _analyze_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Analyze technical indicators and generate signals"""
        signals = {
            'overall_signal': 'NEUTRAL',
            'strength': 'medium',
            'indicators': {},
            'explanation': []
        }
        
        if len(data) < 20:
            signals['explanation'].append("Insufficient data for comprehensive technical analysis")
            return signals
        
        # Simple moving averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=min(50, len(data))).mean()
        
        current_price = float(data['close'].iloc[-1])
        sma_20 = float(data['sma_20'].iloc[-1]) if not pd.isna(data['sma_20'].iloc[-1]) else current_price
        sma_50 = float(data['sma_50'].iloc[-1]) if not pd.isna(data['sma_50'].iloc[-1]) else current_price
        
        # Price vs Moving Averages
        above_sma_20 = current_price > sma_20
        above_sma_50 = current_price > sma_50
        sma_20_vs_50 = sma_20 > sma_50
        
        bullish_signals = 0
        bearish_signals = 0
        
        if above_sma_20:
            bullish_signals += 1
            signals['explanation'].append(f"Price (${current_price:,.2f}) is above 20-day average (${sma_20:,.2f}) - Bullish short-term trend")
        else:
            bearish_signals += 1
            signals['explanation'].append(f"Price (${current_price:,.2f}) is below 20-day average (${sma_20:,.2f}) - Bearish short-term trend")
        
        if above_sma_50:
            bullish_signals += 1
            signals['explanation'].append(f"Price is above 50-day average (${sma_50:,.2f}) - Bullish long-term trend")
        else:
            bearish_signals += 1
            signals['explanation'].append(f"Price is below 50-day average (${sma_50:,.2f}) - Bearish long-term trend")
        
        if sma_20_vs_50:
            bullish_signals += 1
            signals['explanation'].append("Short-term trend is stronger than long-term trend - Positive momentum")
        else:
            bearish_signals += 1
            signals['explanation'].append("Short-term trend is weaker than long-term trend - Negative momentum")
        
        # RSI calculation
        rsi = self._calculate_rsi(data['close'])
        if not pd.isna(rsi):
            if rsi > 70:
                bearish_signals += 1
                signals['explanation'].append(f"RSI is {rsi:.1f} (overbought) - Consider taking profits")
            elif rsi < 30:
                bullish_signals += 1
                signals['explanation'].append(f"RSI is {rsi:.1f} (oversold) - Potential buying opportunity")
            else:
                signals['explanation'].append(f"RSI is {rsi:.1f} (neutral) - No extreme conditions")
        
        # Volume analysis
        if len(data) >= 10:
            avg_volume = data['volume'].rolling(window=10).mean().iloc[-1]
            current_volume = data['volume'].iloc[-1]
            if current_volume > avg_volume * 1.5:
                signals['explanation'].append("High trading volume - Strong market interest")
                bullish_signals += 0.5
            elif current_volume < avg_volume * 0.5:
                signals['explanation'].append("Low trading volume - Weak market interest")
                bearish_signals += 0.5
        
        # Determine overall signal
        if bullish_signals > bearish_signals + 1:
            signals['overall_signal'] = 'BULLISH'
            signals['strength'] = 'high' if bullish_signals > bearish_signals + 2 else 'medium'
        elif bearish_signals > bullish_signals + 1:
            signals['overall_signal'] = 'BEARISH'
            signals['strength'] = 'high' if bearish_signals > bullish_signals + 2 else 'medium'
        else:
            signals['overall_signal'] = 'NEUTRAL'
            signals['strength'] = 'medium'
        
        signals['indicators'] = {
            'rsi': rsi if not pd.isna(rsi) else None,
            'price_vs_sma_20': above_sma_20,
            'price_vs_sma_50': above_sma_50,
            'sma_20_vs_50': sma_20_vs_50,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals
        }
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return np.nan
        
        delta = prices.diff()
        gains = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        losses = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
    
    def _analyze_model_prediction(self, model_proba: float, model_name: str) -> Dict:
        """Analyze ML model prediction and provide explanation"""
        confidence = self._get_confidence_level(model_proba)
        
        if model_proba >= 0.7:
            signal = 'STRONG_BUY'
            explanation = f"AI model ({model_name}) shows {model_proba:.1%} confidence for price increase - Strong buy signal"
        elif model_proba >= 0.6:
            signal = 'BUY'
            explanation = f"AI model ({model_name}) shows {model_proba:.1%} confidence for price increase - Moderate buy signal"
        elif model_proba >= 0.45:
            signal = 'HOLD'
            explanation = f"AI model ({model_name}) shows {model_proba:.1%} confidence - Neutral signal, consider holding"
        elif model_proba >= 0.35:
            signal = 'SELL'
            explanation = f"AI model ({model_name}) shows {model_proba:.1%} confidence for price decrease - Moderate sell signal"
        else:
            signal = 'STRONG_SELL'
            explanation = f"AI model ({model_name}) shows {model_proba:.1%} confidence for price decrease - Strong sell signal"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'probability': model_proba,
            'explanation': explanation,
            'model_name': model_name
        }
    
    def _analyze_market_context(self, data: pd.DataFrame) -> Dict:
        """Analyze broader market context"""
        context = {
            'trend': 'Unknown',
            'volatility': 'Medium',
            'explanation': []
        }
        
        if len(data) < 10:
            context['explanation'].append("Limited historical data for market context analysis")
            return context
        
        # Volatility calculation
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        if volatility > 0.4:
            context['volatility'] = 'High'
            context['explanation'].append(f"High volatility ({volatility:.1%} annually) - Expect significant price swings")
        elif volatility > 0.2:
            context['volatility'] = 'Medium'
            context['explanation'].append(f"Moderate volatility ({volatility:.1%} annually) - Normal market conditions")
        else:
            context['volatility'] = 'Low'
            context['explanation'].append(f"Low volatility ({volatility:.1%} annually) - Stable market conditions")
        
        # Trend analysis
        if len(data) >= 30:
            recent_trend = (data['close'].iloc[-1] / data['close'].iloc[-30] - 1) * 100
            if recent_trend > 10:
                context['trend'] = 'Strong Uptrend'
                context['explanation'].append(f"Strong upward trend (+{recent_trend:.1f}% over 30 days)")
            elif recent_trend > 2:
                context['trend'] = 'Uptrend'
                context['explanation'].append(f"Upward trend (+{recent_trend:.1f}% over 30 days)")
            elif recent_trend < -10:
                context['trend'] = 'Strong Downtrend'
                context['explanation'].append(f"Strong downward trend ({recent_trend:.1f}% over 30 days)")
            elif recent_trend < -2:
                context['trend'] = 'Downtrend'
                context['explanation'].append(f"Downward trend ({recent_trend:.1f}% over 30 days)")
            else:
                context['trend'] = 'Sideways'
                context['explanation'].append(f"Sideways movement ({recent_trend:+.1f}% over 30 days)")
        
        return context
    
    def _generate_recommendation(self, technical_signals: Dict, model_signal: Optional[Dict], 
                               market_context: Dict, current_price: float, price_change_24h: float) -> Dict:
        """Generate final trading recommendation with detailed explanation"""
        
        # Scoring system
        score = 0
        explanations = []
        risk_level = "Medium"
        
        # Technical analysis contribution
        if technical_signals['overall_signal'] == 'BULLISH':
            score += 2 if technical_signals['strength'] == 'high' else 1
            explanations.append(f"Technical indicators are {technical_signals['strength']} bullish")
        elif technical_signals['overall_signal'] == 'BEARISH':
            score -= 2 if technical_signals['strength'] == 'high' else 1
            explanations.append(f"Technical indicators are {technical_signals['strength']} bearish")
        else:
            explanations.append("Technical indicators are neutral")
        
        # Model signal contribution
        if model_signal:
            if model_signal['signal'] in ['STRONG_BUY', 'BUY']:
                model_score = 2 if 'STRONG' in model_signal['signal'] else 1
                score += model_score
                explanations.append(f"AI model predicts price increase with {model_signal['confidence']} confidence")
            elif model_signal['signal'] in ['STRONG_SELL', 'SELL']:
                model_score = 2 if 'STRONG' in model_signal['signal'] else 1
                score -= model_score
                explanations.append(f"AI model predicts price decrease with {model_signal['confidence']} confidence")
            else:
                explanations.append("AI model suggests neutral position")
        
        # Market context adjustments
        if market_context['volatility'] == 'High':
            risk_level = "High"
            explanations.append("High market volatility increases investment risk")
        elif market_context['volatility'] == 'Low':
            risk_level = "Low"
            explanations.append("Low market volatility suggests stable conditions")
        
        # Recent price movement
        if abs(price_change_24h) > 10:
            explanations.append(f"Recent large price movement ({price_change_24h:+.1f}%) suggests high volatility")
            risk_level = "High"
        
        # Generate final recommendation
        if score >= 3:
            decision = "BUY"
            action_explanation = f"Strong indicators suggest buying at current price (${current_price:,.2f})"
            confidence_level = "High"
        elif score >= 1:
            decision = "DCA_IN"
            action_explanation = f"Positive signals suggest gradual accumulation (dollar-cost averaging in)"
            confidence_level = "Medium"
        elif score <= -3:
            decision = "SELL"
            action_explanation = f"Strong indicators suggest selling at current price (${current_price:,.2f})"
            confidence_level = "High"
        elif score <= -1:
            decision = "DCA_OUT"
            action_explanation = f"Negative signals suggest gradual position reduction (dollar-cost averaging out)"
            confidence_level = "Medium"
        else:
            decision = "HOLD"
            action_explanation = f"Mixed signals suggest holding current position"
            confidence_level = "Low"
        
        return {
            'decision': decision,
            'confidence_level': confidence_level,
            'risk_level': risk_level,
            'score': score,
            'action_explanation': action_explanation,
            'detailed_explanation': explanations,
            'suggested_actions': self._get_suggested_actions(decision, current_price, risk_level)
        }
    
    def _get_suggested_actions(self, decision: str, current_price: float, risk_level: str) -> List[str]:
        """Get specific actionable recommendations"""
        actions = []
        
        if decision == "BUY":
            actions.append(f"Consider buying at current price: ${current_price:,.2f}")
            actions.append("Set stop-loss at 5-10% below purchase price")
            actions.append("Take partial profits at 15-20% gains")
            if risk_level == "High":
                actions.append("Start with smaller position size due to high volatility")
        
        elif decision == "DCA_IN":
            actions.append("Begin dollar-cost averaging strategy")
            actions.append("Split your investment into 4-6 equal parts")
            actions.append("Buy one part every few days/weeks")
            actions.append("This reduces timing risk in volatile markets")
        
        elif decision == "SELL":
            actions.append(f"Consider selling at current price: ${current_price:,.2f}")
            actions.append("Take profits while price is favorable")
            actions.append("Consider tax implications of selling")
            actions.append("Keep some cash ready for future opportunities")
        
        elif decision == "DCA_OUT":
            actions.append("Gradually reduce position size")
            actions.append("Sell 20-25% of holdings every few days/weeks")
            actions.append("This allows you to benefit if price continues up")
            actions.append("Preserves capital if price continues down")
        
        else:  # HOLD
            actions.append("Maintain current position")
            actions.append("Monitor for clear trend signals")
            actions.append("Be patient - avoid emotional decisions")
            actions.append("Review position if new information emerges")
        
        return actions
    
    def _get_confidence_level(self, probability: float) -> str:
        """Convert probability to confidence level"""
        if probability >= 0.85:
            return "Very High"
        elif probability >= 0.75:
            return "High"
        elif probability >= 0.65:
            return "Medium"
        elif probability >= 0.55:
            return "Low"
        else:
            return "Very Low"
    
    def _create_no_data_response(self, symbol: str) -> Dict:
        """Create response when no data is available"""
        return {
            'symbol': symbol,
            'current_price': 0,
            'price_change_24h': 0,
            'recommendation': {
                'decision': 'HOLD',
                'confidence_level': 'Very Low',
                'risk_level': 'Unknown',
                'score': 0,
                'action_explanation': 'Insufficient data for analysis',
                'detailed_explanation': ['No price data available for analysis'],
                'suggested_actions': ['Wait for more data before making trading decisions']
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'technical_analysis': {'overall_signal': 'UNKNOWN'},
            'model_analysis': None,
            'market_context': {'trend': 'Unknown'}
        }

    def format_analysis_for_display(self, analysis: Dict) -> str:
        """Format analysis results for user-friendly display"""
        symbol = analysis['symbol']
        rec = analysis['recommendation']
        
        # Header
        output = f"## 🎯 Trading Analysis for {symbol}\n\n"
        
        # Key recommendation
        decision_emoji = {
            'BUY': '🟢',
            'DCA_IN': '🔵', 
            'SELL': '🔴',
            'DCA_OUT': '🟠',
            'HOLD': '⚪'
        }
        
        output += f"### {decision_emoji.get(rec['decision'], '⚪')} **Recommendation: {rec['decision']}**\n"
        output += f"**Confidence:** {rec['confidence_level']} | **Risk Level:** {rec['risk_level']}\n\n"
        
        # Action explanation
        output += f"**What to do:** {rec['action_explanation']}\n\n"
        
        # Detailed explanation
        output += "### 📊 Analysis Summary\n"
        for i, explanation in enumerate(rec['detailed_explanation'], 1):
            output += f"{i}. {explanation}\n"
        
        output += "\n### 📋 Suggested Actions\n"
        for i, action in enumerate(rec['suggested_actions'], 1):
            output += f"{i}. {action}\n"
        
        # Technical details
        if analysis.get('technical_analysis'):
            tech = analysis['technical_analysis']
            output += f"\n### 📈 Technical Analysis\n"
            output += f"- **Overall Signal:** {tech['overall_signal']}\n"
            output += f"- **Signal Strength:** {tech.get('strength', 'Unknown')}\n"
            
            if tech.get('explanation'):
                output += "\n**Technical Details:**\n"
                for explanation in tech['explanation']:
                    output += f"- {explanation}\n"
        
        # Model analysis
        if analysis.get('model_analysis'):
            model = analysis['model_analysis']
            output += f"\n### 🤖 AI Model Prediction\n"
            output += f"- **Model Signal:** {model['signal']}\n"
            output += f"- **Confidence:** {model['confidence']}\n"
            output += f"- **Explanation:** {model['explanation']}\n"
        
        return output