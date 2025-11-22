"""
Quick test to demonstrate Signal Fusion Engine
handling the XRP BUY vs Bearish Flag conflict
"""

from ai_advisor.signal_fusion_engine import SignalFusionEngine

# Simulate the XRP conflict case you experienced
ml_prediction = {
    'symbol': 'XRP',
    'signal': 'BUY',
    'confidence': 0.69,  # 69% confidence
    'probability_up': 0.69,
    'probability_down': 0.31,
    'current_price': 2.45,
    'price_change_24h': -3.2,  # Price dropped 3.2%
    'recommendation': {
        'decision': 'BUY',
        'confidence_level': 'Medium',
        'risk_level': 'Medium',
        'action_explanation': 'AI model predicts upward price movement for XRP. RSI shows room for upward movement. Model confidence: 69.0%.'
    },
    'technical_indicators': {
        'rsi': 45.2,
        'macd': 0.012,
        'ma_5': 2.42,
        'ma_10': 2.48,
        'ma_20': 2.55,
        'volatility': 0.045
    },
    'model_metrics': {
        'accuracy': 0.3636,  # XRP's actual model accuracy is 36.36%
        'precision': 0.30,
        'recall': 0.4615,
        'f1_score': 0.3636
    }
}

# Pattern recognizer detected bearish flag (SELL signal)
pattern_signals = [
    {
        'symbol': 'XRP',
        'pattern_type': 'Bearish Flag Pattern',
        'signal': 'SELL',
        'confidence': 0.75,  # 75% confidence in bearish flag
        'success_rate': 0.83,
        'signal_strength': 0.6225,
        'entry_price': 2.45,
        'target_price': 2.25,  # 8% down
        'stop_loss': 2.55,
        'risk_reward_ratio': 2.0,
        'description': 'Brief consolidation after strong downward move',
        'detected_at': '2025-11-22T20:55:00',
        'timeframe': '1D',
        'expires_at': '2025-11-29T20:55:00'
    }
]

# Create fusion engine
fusion_engine = SignalFusionEngine()

# Fuse the conflicting signals
print("="*80)
print("🔬 SIGNAL FUSION ENGINE TEST - XRP CONFLICT CASE")
print("="*80)
print()
print("📊 INPUT SIGNALS:")
print(f"  • ML Model: {ml_prediction['signal']} at {ml_prediction['confidence']:.1%} confidence")
print(f"  • Pattern: {pattern_signals[0]['signal']} ({pattern_signals[0]['pattern_type']}) at {pattern_signals[0]['confidence']:.1%} confidence")
print()
print("⚠️ CONFLICT DETECTED: ML says BUY but Pattern says SELL")
print()
print("-"*80)

# Fuse signals
unified_signal = fusion_engine.fuse_signals(
    ml_prediction=ml_prediction,
    pattern_signals=pattern_signals,
    symbol='XRP'
)

print()
print("🎯 UNIFIED SIGNAL OUTPUT:")
print("="*80)
print(f"Final Decision: {unified_signal['signal']}")
print(f"Confidence: {unified_signal['confidence']:.1%}")
print(f"Has Conflict: {unified_signal['has_conflict']}")
print(f"Conflict Reason: {unified_signal.get('conflict_reason', 'None')}")
print()
print("📝 RECOMMENDATION:")
print(f"  Decision: {unified_signal['recommendation']['decision']}")
print(f"  Confidence Level: {unified_signal['recommendation']['confidence_level']}")
print(f"  Risk Level: {unified_signal['recommendation']['risk_level']}")
print()
print("💬 EXPLANATION:")
print(f"  {unified_signal['recommendation']['action_explanation']}")
print()
print("-"*80)
print()
print("🔍 DETAILED PERSPECTIVES:")
print()
print("🤖 ML Model Insight:")
print(f"  Signal: {unified_signal['ml_insight']['signal']}")
print(f"  Confidence: {unified_signal['ml_insight']['confidence']:.1%}")
print(f"  Reasoning: {unified_signal['ml_insight']['reasoning']}")
print()
print("📊 Pattern Insight:")
print(f"  Signal: {unified_signal['pattern_insight']['signal']}")
print(f"  Confidence: {unified_signal['pattern_insight']['confidence']:.1%}")
print(f"  Reasoning: {unified_signal['pattern_insight']['reasoning']}")
print()
print("="*80)
print("✅ TEST COMPLETE")
print("="*80)
