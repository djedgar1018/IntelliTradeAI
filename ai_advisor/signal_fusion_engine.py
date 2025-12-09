"""
Signal Fusion Engine
Intelligently combines ML predictions, chart pattern signals, and news sentiment
Resolves conflicts and provides unified trading recommendations with full explainability
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from ai_advisor.price_level_analyzer import PriceLevelAnalyzer


@dataclass
class SignalPayload:
    """Unified signal structure for all AI systems"""
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    source: str  # 'ML_MODEL', 'PATTERN_RECOGNITION', or 'NEWS_INTELLIGENCE'
    reasoning: str  # Explanation for the signal
    technical_data: Dict  # Additional technical details


class SignalFusionEngine:
    """
    Combines three AI signal sources into a unified recommendation:
    1. ML Model predictions (technical analysis)
    2. Chart Pattern Recognition (visual patterns)
    3. News Intelligence (market catalysts and sentiment)
    
    Handles conflicts and provides transparent reasoning
    """
    
    def __init__(self):
        # Weighting factors for different signal sources (total = 1.0)
        self.ml_base_weight = 0.45  # ML model gets 45% base weight
        self.pattern_base_weight = 0.30  # Pattern recognition gets 30% base weight
        self.news_base_weight = 0.25  # News intelligence gets 25% base weight
        
        # Conflict thresholds
        self.high_confidence_threshold = 0.65
        self.conflict_gap_threshold = 0.15
        
        # Signal value mapping for scoring
        self.signal_values = {'BUY': 1, 'HOLD': 0, 'SELL': -1}
        
        # Price level analyzer for HOLD signals
        self.price_analyzer = PriceLevelAnalyzer()
    
    def fuse_signals(self, ml_prediction: Dict, pattern_signals: List[Dict], 
                    symbol: str, historical_data=None, news_data: Dict = None) -> Dict:
        """
        Combine ML prediction, pattern signals, and news sentiment into unified recommendation
        
        Args:
            ml_prediction: Output from MLPredictor
            pattern_signals: List of patterns from ChartPatternRecognizer
            symbol: Asset symbol
            historical_data: Historical price data for price level analysis
            news_data: News intelligence data with sentiment and recommendation
        
        Returns:
            Unified signal with tri-signal conflict resolution
        """
        # Convert inputs to SignalPayload format
        ml_signal = self._convert_ml_to_payload(ml_prediction)
        pattern_signal = self._convert_patterns_to_payload(pattern_signals)
        news_signal = self._convert_news_to_payload(news_data)
        
        # Get all three signals for tri-signal fusion
        signals = [ml_signal, pattern_signal, news_signal]
        
        # Calculate weighted scores for each signal
        weights = self._calculate_adjusted_weights(ml_signal, pattern_signal, news_signal, ml_prediction)
        
        # Perform tri-signal fusion
        unified_signal = self._fuse_tri_signals(
            ml_signal, pattern_signal, news_signal,
            weights, ml_prediction, symbol, historical_data
        )
        
        # Add all perspectives for transparency
        unified_signal['ml_insight'] = {
            'signal': ml_signal.signal,
            'confidence': ml_signal.confidence,
            'reasoning': ml_signal.reasoning,
            'weight': weights['ml']
        }
        
        unified_signal['pattern_insight'] = {
            'signal': pattern_signal.signal,
            'confidence': pattern_signal.confidence,
            'reasoning': pattern_signal.reasoning,
            'weight': weights['pattern']
        }
        
        unified_signal['news_insight'] = {
            'signal': news_signal.signal,
            'confidence': news_signal.confidence,
            'reasoning': news_signal.reasoning,
            'weight': weights['news']
        }
        
        # Determine conflict type
        unified_signal['has_conflict'] = self._detect_tri_conflict(ml_signal, pattern_signal, news_signal)
        unified_signal['conflict_type'] = self._classify_conflict(ml_signal, pattern_signal, news_signal)
        
        return unified_signal
    
    def _convert_news_to_payload(self, news_data: Dict) -> SignalPayload:
        """Convert news sentiment data to SignalPayload format"""
        if not news_data:
            return SignalPayload(
                signal='HOLD',
                confidence=0.0,
                source='NEWS_INTELLIGENCE',
                reasoning='No news data available',
                technical_data={}
            )
        
        recommendation = news_data.get('recommendation', {})
        signal = recommendation.get('recommendation', 'HOLD')
        confidence = recommendation.get('confidence', 0.5)
        rationale = recommendation.get('rationale', 'News sentiment analysis')
        
        # Get catalyst info for context
        articles = news_data.get('articles', [])
        high_impact_count = sum(1 for a in articles if a.get('catalyst', {}).get('is_high_impact', False))
        
        reasoning = rationale
        if high_impact_count > 0:
            reasoning += f" ({high_impact_count} high-impact catalyst{'s' if high_impact_count > 1 else ''} detected)"
        
        return SignalPayload(
            signal=signal,
            confidence=confidence,
            source='NEWS_INTELLIGENCE',
            reasoning=reasoning,
            technical_data={
                'article_count': len(articles),
                'high_impact_count': high_impact_count,
                'sentiment_breakdown': news_data.get('sentiment_breakdown', {})
            }
        )
    
    def _calculate_adjusted_weights(self, ml_signal: SignalPayload, pattern_signal: SignalPayload,
                                    news_signal: SignalPayload, ml_prediction: Dict) -> Dict:
        """Calculate dynamically adjusted weights based on signal quality"""
        # Start with base weights
        ml_weight = self.ml_base_weight
        pattern_weight = self.pattern_base_weight
        news_weight = self.news_base_weight
        
        # Adjust ML weight by model accuracy
        ml_metrics = ml_prediction.get('model_metrics', {})
        ml_accuracy = ml_metrics.get('accuracy', 0.5)
        ml_weight *= (ml_accuracy / 0.5)  # Scale by accuracy relative to 50%
        
        # Adjust pattern weight by confidence
        if pattern_signal.confidence < 0.3:
            pattern_weight *= 0.7  # Reduce weight for low confidence patterns
        elif pattern_signal.confidence > 0.7:
            pattern_weight *= 1.2  # Boost weight for high confidence patterns
        
        # Adjust news weight by number of high-impact catalysts
        high_impact = news_signal.technical_data.get('high_impact_count', 0)
        if high_impact > 0:
            news_weight *= (1 + 0.15 * min(high_impact, 3))  # Up to 45% boost for catalysts
        
        # Normalize weights to sum to 1.0
        total = ml_weight + pattern_weight + news_weight
        return {
            'ml': ml_weight / total,
            'pattern': pattern_weight / total,
            'news': news_weight / total
        }
    
    def _detect_tri_conflict(self, ml: SignalPayload, pattern: SignalPayload, news: SignalPayload) -> bool:
        """Detect if any of the three signals conflict"""
        signals = [s.signal for s in [ml, pattern, news] if s.signal != 'HOLD']
        if len(signals) < 2:
            return False
        return len(set(signals)) > 1  # Conflict if not all the same
    
    def _classify_conflict(self, ml: SignalPayload, pattern: SignalPayload, news: SignalPayload) -> str:
        """Classify the type of conflict"""
        active_signals = [(s.signal, s.source) for s in [ml, pattern, news] if s.signal != 'HOLD']
        
        if len(active_signals) < 2:
            return 'none'
        
        signal_set = set(s[0] for s in active_signals)
        if len(signal_set) == 1:
            return 'consensus'  # All agree
        elif len(active_signals) == 3 and len(signal_set) == 2:
            return 'majority'  # 2 vs 1
        elif len(active_signals) == 2 and len(signal_set) == 2:
            return 'split'  # 1 vs 1 (third is HOLD)
        else:
            return 'three_way'  # All three disagree (BUY vs SELL vs HOLD active)
    
    def _fuse_tri_signals(self, ml: SignalPayload, pattern: SignalPayload, news: SignalPayload,
                         weights: Dict, ml_prediction: Dict, symbol: str, historical_data) -> Dict:
        """Fuse three signals using weighted scoring and consensus logic"""
        
        # Calculate weighted score (-1 to 1 scale)
        ml_value = self.signal_values.get(ml.signal, 0)
        pattern_value = self.signal_values.get(pattern.signal, 0)
        news_value = self.signal_values.get(news.signal, 0)
        
        weighted_score = (
            ml_value * ml.confidence * weights['ml'] +
            pattern_value * pattern.confidence * weights['pattern'] +
            news_value * news.confidence * weights['news']
        )
        
        # Count votes
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        for s in [ml, pattern, news]:
            if s.confidence > 0.3:  # Only count signals with meaningful confidence
                votes[s.signal] += 1
        
        # Determine final signal based on weighted score and consensus
        conflict_type = self._classify_conflict(ml, pattern, news)
        
        if conflict_type == 'consensus':
            # All active signals agree - boost confidence
            agreed_signal = ml.signal if ml.signal != 'HOLD' else (pattern.signal if pattern.signal != 'HOLD' else news.signal)
            avg_conf = np.mean([s.confidence for s in [ml, pattern, news] if s.signal == agreed_signal])
            final_confidence = min(0.95, avg_conf * 1.15)  # 15% confidence boost
            final_signal = agreed_signal
            
            explanation = (
                f"✅ STRONG CONSENSUS: All three AI systems agree on {final_signal}. "
                f"ML Model: {ml.confidence:.0%}, Pattern Recognition: {pattern.confidence:.0%}, "
                f"News Intelligence: {news.confidence:.0%}. High conviction signal."
            )
            risk_level = 'Low'
            
        elif conflict_type == 'majority':
            # 2 vs 1 - go with majority but note dissent
            majority_signal = 'BUY' if votes['BUY'] >= 2 else ('SELL' if votes['SELL'] >= 2 else 'HOLD')
            dissenters = [s for s in [ml, pattern, news] if s.signal != majority_signal and s.signal != 'HOLD']
            
            # Average confidence of majority
            majority_confs = [s.confidence for s in [ml, pattern, news] if s.signal == majority_signal]
            final_confidence = np.mean(majority_confs) * 0.95  # Slight penalty for dissent
            final_signal = majority_signal
            
            dissent_info = ""
            if dissenters:
                d = dissenters[0]
                dissent_info = f"However, {d.source.replace('_', ' ').title()} disagrees with {d.signal} ({d.confidence:.0%}). "
            
            explanation = (
                f"⚖️ MAJORITY DECISION: 2 of 3 systems recommend {final_signal}. {dissent_info}"
                f"Proceeding with majority view but exercise caution."
            )
            risk_level = 'Medium'
            
        elif conflict_type == 'split' or conflict_type == 'three_way':
            # Strong disagreement - use weighted score but be cautious
            if weighted_score > 0.15:
                final_signal = 'BUY'
            elif weighted_score < -0.15:
                final_signal = 'SELL'
            else:
                final_signal = 'HOLD'
            
            final_confidence = min(0.7, abs(weighted_score) + 0.3)  # Cap confidence due to disagreement
            
            # Build detailed explanation
            signal_details = []
            if ml.signal != 'HOLD':
                signal_details.append(f"ML: {ml.signal} ({ml.confidence:.0%})")
            if pattern.signal != 'HOLD':
                signal_details.append(f"Pattern: {pattern.signal} ({pattern.confidence:.0%})")
            if news.signal != 'HOLD':
                signal_details.append(f"News: {news.signal} ({news.confidence:.0%})")
            
            explanation = (
                f"⚠️ CONFLICTING SIGNALS: {' vs '.join(signal_details)}. "
                f"Weighted analysis favors {final_signal} (score: {weighted_score:.2f}). "
                f"This is a high-risk scenario - consider waiting for clearer signals."
            )
            risk_level = 'High'
            
        else:
            # No active signals or all HOLD
            final_signal = 'HOLD'
            final_confidence = 0.6
            explanation = "All systems suggest holding. No clear directional signals detected."
            risk_level = 'Low'
        
        # Determine confidence level label
        if final_confidence >= 0.75:
            confidence_level = 'High'
        elif final_confidence >= 0.55:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'
        
        result = {
            'symbol': symbol,
            'signal': final_signal,
            'confidence': final_confidence,
            'weighted_score': weighted_score,
            'recommendation': {
                'decision': final_signal,
                'confidence_level': confidence_level,
                'risk_level': risk_level,
                'action_explanation': explanation
            },
            'vote_breakdown': votes,
            'current_price': ml_prediction.get('current_price', 0),
            'price_change_24h': ml_prediction.get('price_change_24h', 0),
            'technical_indicators': ml_prediction.get('technical_indicators', {})
        }
        
        # Add price levels for HOLD signals
        if final_signal == 'HOLD' and historical_data is not None:
            try:
                price_levels = self.price_analyzer.analyze_key_levels(
                    historical_data,
                    result['current_price'],
                    result['technical_indicators']
                )
                result['price_levels'] = price_levels
            except:
                pass
        
        return result
    
    def _convert_ml_to_payload(self, ml_pred: Dict) -> SignalPayload:
        """Convert ML prediction to SignalPayload format"""
        return SignalPayload(
            signal=ml_pred.get('signal', 'HOLD'),
            confidence=ml_pred.get('confidence', 0.5),
            source='ML_MODEL',
            reasoning=ml_pred.get('recommendation', {}).get('action_explanation', 'ML-based prediction'),
            technical_data=ml_pred.get('technical_indicators', {})
        )
    
    def _convert_patterns_to_payload(self, patterns: List[Dict]) -> SignalPayload:
        """Convert pattern signals to SignalPayload format"""
        if not patterns:
            return SignalPayload(
                signal='HOLD',
                confidence=0.0,
                source='PATTERN_RECOGNITION',
                reasoning='No chart patterns detected',
                technical_data={}
            )
        
        # Use the strongest pattern signal
        strongest = max(patterns, key=lambda p: p.get('confidence', 0))
        
        return SignalPayload(
            signal=strongest.get('signal', 'HOLD'),
            confidence=strongest.get('confidence', 0.5),
            source='PATTERN_RECOGNITION',
            reasoning=f"{strongest.get('pattern_type', 'Pattern')}: {strongest.get('description', 'Detected pattern')}",
            technical_data=strongest
        )
    
    def _detect_conflict(self, ml_signal: SignalPayload, pattern_signal: SignalPayload) -> bool:
        """Detect if ML and pattern signals conflict"""
        # HOLD doesn't conflict with anything
        if ml_signal.signal == 'HOLD' or pattern_signal.signal == 'HOLD':
            return False
        
        # BUY vs SELL is a conflict
        if ml_signal.signal != pattern_signal.signal:
            return True
        
        return False
    
    def _resolve_conflict(self, ml_signal: SignalPayload, pattern_signal: SignalPayload,
                         ml_prediction: Dict, symbol: str, historical_data=None) -> Dict:
        """
        Resolve conflicting signals with intelligent logic
        
        Rules:
        1. If both have high confidence (>65%) but gap <15% → HOLD (too risky)
        2. If gap ≥15% → Choose higher confidence signal
        3. Weight ML model by its actual accuracy for this symbol
        """
        ml_conf = ml_signal.confidence
        pattern_conf = pattern_signal.confidence
        
        # Get ML model's actual accuracy for weighting
        ml_metrics = ml_prediction.get('model_metrics', {})
        ml_accuracy = ml_metrics.get('accuracy', 0.5)  # Default to 50% if unknown
        
        # Adjust weights based on actual model performance
        ml_weight = self.ml_base_weight * (ml_accuracy / 0.5)  # Scale by accuracy
        pattern_weight = self.pattern_base_weight
        
        # Normalize weights
        total_weight = ml_weight + pattern_weight
        ml_weight /= total_weight
        pattern_weight /= total_weight
        
        # Calculate weighted scores
        ml_score = ml_conf * ml_weight
        pattern_score = pattern_conf * pattern_weight
        
        # Check confidence gap
        confidence_gap = abs(ml_conf - pattern_conf)
        
        # Rule 1: Both high confidence but close gap → HOLD (avoid risky conflict)
        if (ml_conf >= self.high_confidence_threshold and 
            pattern_conf >= self.high_confidence_threshold and
            confidence_gap < self.conflict_gap_threshold):
            
            hold_signal = {
                'symbol': symbol,
                'signal': 'HOLD',
                'confidence': max(ml_conf, pattern_conf),
                'recommendation': {
                    'decision': 'HOLD',
                    'confidence_level': 'High',
                    'risk_level': 'High',
                    'action_explanation': (
                        f"⚠️ CONFLICTING SIGNALS DETECTED: ML Model says {ml_signal.signal} "
                        f"({ml_conf:.1%} confidence) but Chart Pattern shows {pattern_signal.signal} "
                        f"({pattern_conf:.1%} confidence). Both signals are strong but contradictory. "
                        f"Recommendation: HOLD until signals align. This protects you from risky trades "
                        f"when AI systems disagree."
                    )
                },
                'conflict_reason': f'High-confidence disagreement: {ml_signal.signal} vs {pattern_signal.signal}',
                'current_price': ml_prediction.get('current_price', 0),
                'price_change_24h': ml_prediction.get('price_change_24h', 0),
                'technical_indicators': ml_prediction.get('technical_indicators', {})
            }
            
            # Add price levels for HOLD signals
            if historical_data is not None:
                price_levels = self.price_analyzer.analyze_key_levels(
                    historical_data,
                    hold_signal['current_price'],
                    hold_signal['technical_indicators']
                )
                hold_signal['price_levels'] = price_levels
            
            return hold_signal
        
        # Rule 2: Choose higher weighted score
        if ml_score > pattern_score:
            chosen_signal = ml_signal.signal
            chosen_conf = ml_conf
            reason = (
                f"ML Model prediction ({ml_signal.signal} at {ml_conf:.1%}) has higher weighted "
                f"confidence than Pattern Recognition ({pattern_signal.signal} at {pattern_conf:.1%}). "
                f"However, note that Chart Pattern detected: {pattern_signal.reasoning}. "
                f"Exercise caution due to conflicting signals."
            )
        else:
            chosen_signal = pattern_signal.signal
            chosen_conf = pattern_conf
            reason = (
                f"Chart Pattern ({pattern_signal.signal} at {pattern_conf:.1%}) has higher confidence "
                f"than ML Model ({ml_signal.signal} at {ml_conf:.1%}). "
                f"However, ML analysis suggests: {ml_signal.reasoning}. "
                f"Exercise caution due to conflicting signals."
            )
        
        # Determine confidence and risk levels
        if chosen_conf >= 0.75:
            confidence_level = 'High'
        elif chosen_conf >= 0.60:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'
        
        return {
            'symbol': symbol,
            'signal': chosen_signal,
            'confidence': chosen_conf,
            'recommendation': {
                'decision': chosen_signal,
                'confidence_level': confidence_level,
                'risk_level': 'High',  # Always high risk when conflicting
                'action_explanation': f"⚠️ {reason}"
            },
            'conflict_reason': f'Resolved: Chose {chosen_signal} (ML: {ml_signal.signal}, Pattern: {pattern_signal.signal})',
            'current_price': ml_prediction.get('current_price', 0),
            'price_change_24h': ml_prediction.get('price_change_24h', 0),
            'technical_indicators': ml_prediction.get('technical_indicators', {})
        }
    
    def _combine_aligned_signals(self, ml_signal: SignalPayload, 
                                pattern_signal: SignalPayload, ml_prediction: Dict, historical_data=None) -> Dict:
        """
        Combine signals when they agree (both BUY, both SELL, or one is HOLD)
        """
        # If signals agree, boost confidence
        if ml_signal.signal == pattern_signal.signal and ml_signal.signal != 'HOLD':
            # Both agree on BUY or SELL - boost confidence
            combined_confidence = min(0.95, (ml_signal.confidence + pattern_signal.confidence) / 2 * 1.1)
            signal = ml_signal.signal
            
            explanation = (
                f"✅ ALIGNED SIGNALS: Both ML Model and Chart Pattern agree on {signal}. "
                f"ML confidence: {ml_signal.confidence:.1%}, Pattern confidence: {pattern_signal.confidence:.1%}. "
                f"Combined confidence: {combined_confidence:.1%}. Strong agreement across both AI systems."
            )
            
        elif ml_signal.signal == 'HOLD' or pattern_signal.signal == 'HOLD':
            # One says HOLD - use the other signal but don't boost
            if ml_signal.signal != 'HOLD':
                signal = ml_signal.signal
                combined_confidence = ml_signal.confidence * 0.9  # Slight penalty for no pattern confirmation
                explanation = f"ML Model suggests {signal} ({ml_signal.confidence:.1%}). Chart patterns are neutral."
            else:
                signal = pattern_signal.signal
                combined_confidence = pattern_signal.confidence * 0.9
                explanation = f"Chart Pattern suggests {signal} ({pattern_signal.confidence:.1%}). ML model is neutral."
        else:
            # Both HOLD
            signal = 'HOLD'
            combined_confidence = max(ml_signal.confidence, pattern_signal.confidence)
            explanation = "Both ML Model and Chart Pattern suggest waiting. No clear signals detected. Watch the key levels below for actionable opportunities."
        
        # Determine levels
        if combined_confidence >= 0.75:
            confidence_level = 'High'
        elif combined_confidence >= 0.60:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'
        
        volatility = ml_prediction.get('technical_indicators', {}).get('volatility', 0.03)
        if volatility > 0.05:
            risk_level = 'High'
        elif volatility > 0.03:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        result = {
            'symbol': ml_prediction.get('symbol', 'UNKNOWN'),
            'signal': signal,
            'confidence': combined_confidence,
            'recommendation': {
                'decision': signal,
                'confidence_level': confidence_level,
                'risk_level': risk_level,
                'action_explanation': explanation
            },
            'conflict_reason': None,
            'current_price': ml_prediction.get('current_price', 0),
            'price_change_24h': ml_prediction.get('price_change_24h', 0),
            'technical_indicators': ml_prediction.get('technical_indicators', {})
        }
        
        # Add price levels for HOLD signals
        if signal == 'HOLD' and historical_data is not None:
            price_levels = self.price_analyzer.analyze_key_levels(
                historical_data,
                result['current_price'],
                result['technical_indicators']
            )
            result['price_levels'] = price_levels
        
        return result
