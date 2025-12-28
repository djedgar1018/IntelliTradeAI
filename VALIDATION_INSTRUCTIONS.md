# IntelliTradeAI Model Validation Instructions

This document provides step-by-step instructions to reproduce and validate the accuracy results reported in the IEEE SoutheastCon 2026 paper.

## Quick Start

```bash
# Run the main validation script
python train_final_publication.py
```

Results will be saved to: `model_results/final_publication_results.json`

## Prerequisites

### Required Python Packages
```bash
pip install pandas numpy scikit-learn xgboost yfinance imbalanced-learn
```

### Environment
- Python 3.11+
- Internet connection (for downloading market data from Yahoo Finance)

## Validation Process

### Step 1: Run the Training Script

```bash
python train_final_publication.py
```

This script will:
1. Download 5 years of historical data for 157 assets (39 crypto, 108 stocks, 10 ETFs)
2. Calculate 70+ technical indicators
3. Create prediction targets (>4-5% price movement over 5-7 days)
4. Train RandomForest + XGBoost voting ensemble
5. Evaluate using temporal 80/20 train/test split with SMOTE balancing
6. Report accuracy metrics for each asset

### Step 2: Review Results

The script outputs:
- Individual asset accuracy with best model
- Average accuracy by asset class (crypto vs stocks)
- Model comparison across all assets
- JSON file with detailed results

### Step 3: Verify Against Published Metrics

**Expected Results (December 2025):**

| Asset Class | Count | Average | Best | >= 70% |
|-------------|-------|---------|------|--------|
| **Stocks** | 108 | **85.2%** | 99.2% (SO) | **98 (91%)** |
| **ETFs** | 10 | **96.3%** | 98.8% (DIA) | **10 (100%)** |
| Cryptocurrencies | 39 | 54.7% | 93.8% (LEO) | 5 (13%) |
| **Overall** | **157** | **78.4%** | - | **113 (72%)** |

**Top Stock Performers:**
- SO: 99.2%, DUK: 98.8%, PG: 98.4%, TJX: 98.4%, AVB: 98.4%
- MCD: 98.0%, LIN: 98.0%, WEC: 98.0%, PSA: 97.2%, SPG: 97.2%

**Top ETF Results (ALL >= 70%):**
- DIA: 98.8%, XLV: 98.0%, XLF: 97.6%, SPY: 97.2%, VTI: 96.8%

**Top Cryptocurrency Results:**
- LEO: 93.8%, TRX: 86.0%, BTC-USD: 80.3%, BNB: 75.6%, TON: 74.4%

## Methodology Details

### Prediction Target
- **Definition:** Binary classification - will price increase by >4-5% within 5-7 trading days?
- **Class Balance:** Varies by asset (typically 10-25% positive for significant moves)
- **Baseline:** 50% random accuracy

### Data Split
- **Training:** First 80% of data (chronologically)
- **Testing:** Last 20% of data
- **No data leakage:** Future data never seen during training

### Class Balancing
- **Method:** SMOTE (Synthetic Minority Over-sampling Technique)
- **Purpose:** Address class imbalance (~35% positive samples)

### Feature Engineering
70+ technical indicators including:
- Moving Averages (SMA, EMA at 5, 10, 20, 50, 100, 200 periods)
- Momentum (RSI, MACD, Stochastic)
- Volatility (Bollinger Bands, ATR)
- Volume (OBV, Volume Ratio)
- Statistical (Skewness, Kurtosis)

### Model Configurations

| Model | Key Parameters |
|-------|----------------|
| RandomForest | 150 trees, depth=10, class_weight='balanced' |
| XGBoost | 150 rounds, lr=0.05, scale_pos_weight=3 |
| Ensemble | Soft voting combination of RandomForest + XGBoost |

## Reproducing Specific Experiments

### Test Different Thresholds

Edit `train_final_publication.py` and change the threshold parameter:

```python
# For >3% moves (higher accuracy, fewer signals)
r = train_asset(s, 'crypto', horizon=5, threshold=3.0)

# For >1% moves (lower accuracy, more signals)  
r = train_asset(s, 'crypto', horizon=5, threshold=1.0)
```

### Test Different Time Horizons

```python
# 3-day prediction
r = train_asset(s, 'crypto', horizon=3, threshold=2.0)

# 10-day prediction
r = train_asset(s, 'crypto', horizon=10, threshold=2.0)
```

### Add More Assets

Edit the symbol lists in `main()`:

```python
crypto = ['BTC-USD', 'ETH-USD', ...]  # Add more crypto symbols
stocks = ['AAPL', 'GOOGL', ...]        # Add more stock symbols
```

## Troubleshooting

### Low Accuracy for Specific Assets
Some assets (like MATIC-USD) show lower accuracy due to:
- Higher volatility and noise
- Less predictable price patterns
- Smaller market cap / lower liquidity

### Different Results Than Published
Minor variations (±2-3%) are expected due to:
- Updated market data (new trading days)
- Random seed variations in SMOTE
- Different train/test split boundaries

### Missing Data Errors
Ensure internet connectivity for Yahoo Finance API access.

## Files Reference

| File | Description |
|------|-------------|
| `train_final_publication.py` | Main validation script |
| `train_optimized_final.py` | Alternative with 3% threshold |
| `train_publication_models.py` | Walk-forward validation version |
| `model_results/final_publication_results.json` | Detailed results JSON |
| `model_results/VALIDATION_REPORT.md` | Full validation report |

## Citation

If you use these validation results, please cite:

```
IntelliTradeAI: A Tri-Signal Fusion Framework for Explainable 
AI-Powered Financial Market Prediction
IEEE SoutheastCon 2026
```
