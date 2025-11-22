"""
Test Signal Fusion Engine with REAL MLPredictor output
Validates that the fusion engine correctly handles actual ML predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ai_advisor.ml_predictor import MLPredictor
from ai_vision.chart_pattern_recognition import ChartPatternRecognizer
from ai_advisor.signal_fusion_engine import SignalFusionEngine

# Create synthetic XRP data for testing
dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
np.random.seed(42)
prices = 2.5 + np.cumsum(np.random.randn(200) * 0.05)

xrp_data = pd.DataFrame({
    'open': prices + np.random.randn(200) * 0.02,
    'high': prices + abs(np.random.randn(200) * 0.03),
    'low': prices - abs(np.random.randn(200) * 0.03),
    'close': prices,
    'volume': np.random.randint(10000000, 50000000, 200)
}, index=dates)

print("="*80)
print("🧪 REAL ML PREDICTOR + FUSION ENGINE TEST")
print("="*80)
print()

# Initialize real components
ml_predictor = MLPredictor()
pattern_recognizer = ChartPatternRecognizer()
fusion_engine = SignalFusionEngine()

print("📊 Step 1: Get REAL ML Prediction for XRP")
print("-"*80)
ml_analysis = ml_predictor.analyze_asset('XRP', xrp_data)

print(f"✅ ML Prediction received:")
print(f"  • Symbol: {ml_analysis.get('symbol', 'N/A')}")
print(f"  • Signal: {ml_analysis.get('signal', 'N/A')}")
print(f"  • Confidence: {ml_analysis.get('confidence', 0):.1%}")
print(f"  • Current Price: ${ml_analysis.get('current_price', 0):.2f}")
print(f"  • Price Change 24h: {ml_analysis.get('price_change_24h', 0):+.1f}%")
print(f"  • Decision: {ml_analysis.get('recommendation', {}).get('decision', 'N/A')}")
print(f"  • Confidence Level: {ml_analysis.get('recommendation', {}).get('confidence_level', 'N/A')}")
print()

print("📈 Step 2: Get Chart Pattern Detection for XRP")
print("-"*80)
patterns = pattern_recognizer.detect_patterns_from_data(xrp_data, 'XRP')

print(f"✅ Patterns detected: {len(patterns)}")
if patterns:
    for pattern in patterns:
        print(f"  • {pattern['pattern_type']}: {pattern['signal']} ({pattern['confidence']:.1%})")
print()

print("🔄 Step 3: Fuse Signals with Signal Fusion Engine")
print("-"*80)
unified_signal = fusion_engine.fuse_signals(
    ml_prediction=ml_analysis,
    pattern_signals=patterns,
    symbol='XRP'
)

print(f"✅ Unified Signal Generated:")
print(f"  • Final Signal: {unified_signal['signal']}")
print(f"  • Confidence: {unified_signal['confidence']:.1%}")
print(f"  • Has Conflict: {unified_signal['has_conflict']}")
print(f"  • Current Price: ${unified_signal['current_price']:.2f}")
print(f"  • Price Change: {unified_signal['price_change_24h']:+.1f}%")
print()

print("🎯 Step 4: Verify All Required Fields Present")
print("-"*80)
required_fields = [
    'symbol', 'signal', 'confidence', 'recommendation',
    'current_price', 'price_change_24h', 'technical_indicators',
    'ml_insight', 'pattern_insight', 'has_conflict'
]

all_present = True
for field in required_fields:
    present = field in unified_signal
    status = "✅" if present else "❌"
    print(f"{status} {field}: {'Present' if present else 'MISSING'}")
    if not present:
        all_present = False

print()
print("="*80)
if all_present:
    print("✅ TEST PASSED: All fields present and fusion engine working correctly!")
else:
    print("❌ TEST FAILED: Some fields missing!")
print("="*80)
print()

print("📝 Unified Recommendation:")
print(f"  {unified_signal['recommendation']['action_explanation']}")
print()
print("="*80)
