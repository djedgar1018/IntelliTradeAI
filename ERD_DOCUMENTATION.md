# 📊 IntelliTradeAI - Entity Relationship Diagram (ERD)
## Complete Data Model Documentation

**Created:** November 19, 2025  
**Diagrams:** 2 ERD diagrams (Comprehensive + Simplified)

---

## 🎯 Overview

This document explains the data model for **IntelliTradeAI**, an AI-powered cryptocurrency and stock trading system. The ERD shows how different data entities relate to each other and support the trading system's operations.

**Key Points:**
- **Current Storage:** File-based (JSON files)
- **Future Migration:** PostgreSQL database (schema-ready)
- **Design Philosophy:** Normalized, scalable, and ML-friendly

---

## 📁 Generated Diagrams

### 1. Comprehensive ERD
**File:** `diagrams/erd_diagram.png`

Shows all 10 entities with complete attributes and relationships:
- Cryptocurrency
- OHLCV_Data
- Technical_Indicators
- ML_Models
- Training_Sessions
- Predictions
- Portfolio_Performance
- API_Cache
- Feature_Engineering
- Backtest_Results

### 2. Simplified ERD
**File:** `diagrams/erd_simplified.png`

Shows 5 core entities for quick understanding:
- Assets
- Price_Data
- Models
- Predictions
- Performance

---

## 🗂️ Entity Descriptions

### 1. **Cryptocurrency** (Master Table)

**Purpose:** Stores information about cryptocurrencies and stocks being tracked

**Attributes:**
| Column | Type | Description |
|--------|------|-------------|
| **symbol** (PK) | VARCHAR(10) | Crypto symbol (BTC, ETH, etc.) |
| name | VARCHAR(100) | Full name (Bitcoin, Ethereum) |
| cmc_id | INT | CoinMarketCap ID |
| rank | INT | Current market cap rank |
| yahoo_symbol | VARCHAR(20) | Yahoo Finance symbol (BTC-USD) |
| market_cap | DECIMAL(20,2) | Current market capitalization |
| last_updated | TIMESTAMP | Last data update time |

**Relationships:**
- One-to-Many with OHLCV_Data
- One-to-Many with ML_Models
- One-to-Many with Portfolio_Performance
- One-to-Many with API_Cache

**Current Implementation:**
```json
// data/top_coins_cache.json
{
  "timestamp": "2025-11-19T21:08:10.673Z",
  "coins": [
    {
      "id": 1,
      "symbol": "BTC",
      "name": "Bitcoin",
      "rank": 1,
      "yahoo_symbol": "BTC-USD",
      "market_cap": 1789234567890,
      "last_updated": "2025-11-19T21:08:10.673Z"
    }
  ]
}
```

---

### 2. **OHLCV_Data** (Price History)

**Purpose:** Stores historical price data (Open, High, Low, Close, Volume)

**Attributes:**
| Column | Type | Description |
|--------|------|-------------|
| **id** (PK) | INT | Unique record ID |
| **symbol** (FK) | VARCHAR(10) | References Cryptocurrency |
| date | DATE | Trading date |
| open | DECIMAL(20,8) | Opening price |
| high | DECIMAL(20,8) | Highest price |
| low | DECIMAL(20,8) | Lowest price |
| close | DECIMAL(20,8) | Closing price |
| volume | BIGINT | Trading volume |
| source | VARCHAR(50) | Data source (Yahoo, CMC) |

**Relationships:**
- Many-to-One with Cryptocurrency
- One-to-Many with Technical_Indicators
- One-to-Many with Feature_Engineering

**Current Implementation:**
```json
// data/crypto_top10_cache.json
{
  "BTC": {
    "data": [
      {
        "date": "2025-11-19",
        "open": 90234.50,
        "high": 91345.20,
        "low": 89901.10,
        "close": 90567.80,
        "volume": 45678901234
      }
    ]
  }
}
```

---

### 3. **Technical_Indicators** (Calculated Metrics)

**Purpose:** Stores technical analysis indicators calculated from OHLCV data

**Attributes:**
| Column | Type | Description |
|--------|------|-------------|
| **id** (PK) | INT | Unique indicator ID |
| **ohlcv_id** (FK) | INT | References OHLCV_Data |
| rsi | DECIMAL(10,4) | Relative Strength Index (0-100) |
| macd | DECIMAL(10,4) | MACD line |
| macd_signal | DECIMAL(10,4) | MACD signal line |
| bb_upper | DECIMAL(20,8) | Bollinger Band upper |
| bb_middle | DECIMAL(20,8) | Bollinger Band middle |
| bb_lower | DECIMAL(20,8) | Bollinger Band lower |
| ema_12 | DECIMAL(20,8) | 12-period EMA |
| ema_26 | DECIMAL(20,8) | 26-period EMA |
| volume_ratio | DECIMAL(10,4) | Volume vs average ratio |

**Relationships:**
- Many-to-One with OHLCV_Data

**Purpose in ML:**
These indicators are used as features in machine learning models for prediction.

---

### 4. **ML_Models** (Trained Models)

**Purpose:** Stores metadata about trained machine learning models

**Attributes:**
| Column | Type | Description |
|--------|------|-------------|
| **model_id** (PK) | INT | Unique model ID |
| **symbol** (FK) | VARCHAR(10) | Cryptocurrency this model predicts |
| model_type | VARCHAR(50) | RF, XGBoost, LSTM, Ensemble |
| version | VARCHAR(20) | Model version (v1.0, v2.1) |
| accuracy | DECIMAL(5,4) | Test set accuracy (0.0-1.0) |
| precision | DECIMAL(5,4) | Precision score |
| recall | DECIMAL(5,4) | Recall score |
| f1_score | DECIMAL(5,4) | F1 score |
| roc_auc | DECIMAL(5,4) | ROC-AUC score |
| trained_date | TIMESTAMP | When model was trained |
| model_path | VARCHAR(255) | File path to serialized model |

**Relationships:**
- Many-to-One with Cryptocurrency
- One-to-Many with Training_Sessions
- One-to-Many with Predictions
- One-to-Many with Backtest_Results

**Current Implementation:**
```python
# models/model_cache/BTC_random_forest.pkl (joblib serialized)
{
    "model_id": 1,
    "symbol": "BTC",
    "model_type": "Random Forest",
    "accuracy": 0.85,
    "precision": 0.88,
    "recall": 0.82,
    "f1_score": 0.85
}
```

---

### 5. **Training_Sessions** (Model Training History)

**Purpose:** Tracks each model training session with parameters and results

**Attributes:**
| Column | Type | Description |
|--------|------|-------------|
| **session_id** (PK) | INT | Unique session ID |
| **model_id** (FK) | INT | References ML_Models |
| train_start | TIMESTAMP | Training start time |
| train_end | TIMESTAMP | Training completion time |
| train_samples | INT | Number of training samples |
| test_samples | INT | Number of test samples |
| hyperparameters | JSON | Model hyperparameters |
| status | VARCHAR(20) | completed, failed, running |

**Relationships:**
- Many-to-One with ML_Models

**Hyperparameters Example:**
```json
{
  "n_estimators": 200,
  "max_depth": 20,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "random_state": 42
}
```

---

### 6. **Predictions** (Model Predictions)

**Purpose:** Stores predictions made by ML models

**Attributes:**
| Column | Type | Description |
|--------|------|-------------|
| **prediction_id** (PK) | INT | Unique prediction ID |
| **model_id** (FK) | INT | Which model made prediction |
| **symbol** (FK) | VARCHAR(10) | Cryptocurrency predicted |
| prediction_date | TIMESTAMP | When prediction was made |
| signal | VARCHAR(10) | BUY, SELL, HOLD |
| confidence | DECIMAL(5,4) | Confidence score (0.0-1.0) |
| predicted_direction | INT | 1 (up), 0 (down) |
| actual_direction | INT | Actual outcome (for evaluation) |
| target_price | DECIMAL(20,8) | Predicted price target |

**Relationships:**
- Many-to-One with ML_Models
- Many-to-One with Cryptocurrency

**Example:**
```json
{
  "prediction_id": 12345,
  "model_id": 1,
  "symbol": "BTC",
  "prediction_date": "2025-11-19T15:30:00Z",
  "signal": "BUY",
  "confidence": 0.853,
  "predicted_direction": 1,
  "target_price": 95000.00
}
```

---

### 7. **Portfolio_Performance** (Performance Metrics)

**Purpose:** Tracks portfolio performance metrics for each cryptocurrency

**Attributes:**
| Column | Type | Description |
|--------|------|-------------|
| **performance_id** (PK) | INT | Unique performance record ID |
| **symbol** (FK) | VARCHAR(10) | Cryptocurrency |
| period_start | DATE | Performance period start |
| period_end | DATE | Performance period end |
| total_return_pct | DECIMAL(10,4) | Total return % |
| volatility_pct | DECIMAL(10,4) | Daily volatility % |
| sharpe_ratio | DECIMAL(10,4) | Risk-adjusted return |
| max_drawdown | DECIMAL(10,4) | Maximum loss % |
| win_rate | DECIMAL(5,4) | % of profitable days |

**Relationships:**
- Many-to-One with Cryptocurrency

**Example (6-month performance):**
```json
{
  "symbol": "BNB",
  "period_start": "2025-05-19",
  "period_end": "2025-11-19",
  "total_return_pct": 37.17,
  "volatility_pct": 2.94,
  "sharpe_ratio": 1.85,
  "max_drawdown": -12.34,
  "win_rate": 0.573
}
```

---

### 8. **API_Cache** (API Response Cache)

**Purpose:** Caches API responses to minimize API calls and improve performance

**Attributes:**
| Column | Type | Description |
|--------|------|-------------|
| **cache_id** (PK) | INT | Unique cache ID |
| cache_key | VARCHAR(255) | Cache key (unique identifier) |
| data | JSON | Cached data (flexible structure) |
| created_at | TIMESTAMP | Cache creation time |
| expires_at | TIMESTAMP | Cache expiration time |
| source | VARCHAR(50) | API source (CoinMarketCap, Yahoo) |

**Current Implementation:**
```json
// data/top_coins_cache.json
{
  "timestamp": "2025-11-19T21:08:10.673Z",
  "coins": [...],
  "ttl": 3600
}
```

**Cache Strategy:**
- Top 10 coins: 1-hour TTL
- OHLCV data: 5-minute TTL
- Current prices: 1-minute TTL

---

### 9. **Feature_Engineering** (ML Features)

**Purpose:** Stores engineered features used for machine learning

**Attributes:**
| Column | Type | Description |
|--------|------|-------------|
| **feature_id** (PK) | INT | Unique feature record ID |
| **ohlcv_id** (FK) | INT | References source OHLCV data |
| momentum_features | JSON | ROC, momentum indicators |
| volatility_features | JSON | ATR, std dev, ranges |
| pattern_features | JSON | Candlestick patterns |
| lagged_features | JSON | Historical values (t-1, t-2, etc.) |
| target_variable | INT | Binary target (1=up, 0=down) |

**Feature Categories (70+ total):**
- **Momentum** (9 features): ROC_1, ROC_3, ROC_5, ROC_10, ROC_20, etc.
- **Volatility** (5 features): ATR, rolling_std_5, rolling_std_10, etc.
- **Patterns** (8 features): Doji, Hammer, Shooting Star, etc.
- **Lagged** (10 features): return_lag_1, return_lag_2, volume_lag_1, etc.

---

### 10. **Backtest_Results** (Strategy Testing)

**Purpose:** Stores backtesting results for trading strategies

**Attributes:**
| Column | Type | Description |
|--------|------|-------------|
| **backtest_id** (PK) | INT | Unique backtest ID |
| **model_id** (FK) | INT | Model being backtested |
| start_date | DATE | Backtest period start |
| end_date | DATE | Backtest period end |
| initial_capital | DECIMAL(20,2) | Starting capital ($) |
| final_capital | DECIMAL(20,2) | Ending capital ($) |
| total_trades | INT | Number of trades executed |
| winning_trades | INT | Profitable trades |
| losing_trades | INT | Losing trades |
| profit_factor | DECIMAL(10,4) | Gross profit / gross loss |

**Relationships:**
- Many-to-One with ML_Models

**Example:**
```json
{
  "backtest_id": 1,
  "model_id": 1,
  "start_date": "2025-05-19",
  "end_date": "2025-11-19",
  "initial_capital": 10000.00,
  "final_capital": 13517.00,
  "total_trades": 47,
  "winning_trades": 32,
  "losing_trades": 15,
  "profit_factor": 2.34
}
```

---

## 🔗 Key Relationships

### Primary Relationships

```
Cryptocurrency (1) ──→ (N) OHLCV_Data
    └─ One crypto has many price records

OHLCV_Data (1) ──→ (N) Technical_Indicators
    └─ Each price point has multiple indicators

OHLCV_Data (1) ──→ (N) Feature_Engineering
    └─ Each price point generates features

Cryptocurrency (1) ──→ (N) ML_Models
    └─ One crypto can have multiple models

ML_Models (1) ──→ (N) Training_Sessions
    └─ Each model has training history

ML_Models (1) ──→ (N) Predictions
    └─ Each model generates predictions

ML_Models (1) ──→ (N) Backtest_Results
    └─ Each model can be backtested

Cryptocurrency (1) ──→ (N) Portfolio_Performance
    └─ Track performance per crypto

Cryptocurrency (1) ──→ (N) API_Cache
    └─ Cache data per crypto
```

---

## 📊 Data Flow Through System

### 1. **Data Ingestion Flow**

```
External APIs
    ↓
API_Cache (check cache)
    ↓
Cryptocurrency (create/update)
    ↓
OHLCV_Data (store historical prices)
    ↓
Technical_Indicators (calculate)
    ↓
Feature_Engineering (create ML features)
```

### 2. **Model Training Flow**

```
Feature_Engineering (load features)
    ↓
ML_Models (create/update model)
    ↓
Training_Sessions (track training)
    ↓
Backtest_Results (evaluate strategy)
    ↓
Portfolio_Performance (calculate metrics)
```

### 3. **Prediction Flow**

```
OHLCV_Data (get latest data)
    ↓
Technical_Indicators (calculate indicators)
    ↓
Feature_Engineering (generate features)
    ↓
ML_Models (load trained model)
    ↓
Predictions (generate signal)
    ↓
Portfolio_Performance (track results)
```

---

## 💾 Current vs Future Storage

### Current Implementation (File-based)

**Advantages:**
- ✅ Simple and portable
- ✅ No database setup required
- ✅ Easy to inspect and debug
- ✅ Version control friendly (JSON)

**Limitations:**
- ⚠️ Limited query capabilities
- ⚠️ No ACID transactions
- ⚠️ Scaling challenges with large data
- ⚠️ No concurrent access control

**Files:**
```
data/
├── top_coins_cache.json           # Cryptocurrency table
├── crypto_top10_cache.json        # OHLCV_Data table
├── crypto_data.json               # API_Cache table
└── stock_data.json                # API_Cache table

models/
└── model_cache/
    ├── BTC_random_forest.pkl      # ML_Models + Training_Sessions
    ├── ETH_xgboost.pkl
    └── features/                  # Feature_Engineering
```

---

### Future Implementation (PostgreSQL)

**Migration Path:**

```sql
-- Create tables (already designed in ERD)
CREATE TABLE cryptocurrency (
    symbol VARCHAR(10) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    cmc_id INT,
    rank INT,
    yahoo_symbol VARCHAR(20),
    market_cap DECIMAL(20,2),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ohlcv_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) REFERENCES cryptocurrency(symbol),
    date DATE NOT NULL,
    open DECIMAL(20,8),
    high DECIMAL(20,8),
    low DECIMAL(20,8),
    close DECIMAL(20,8),
    volume BIGINT,
    source VARCHAR(50),
    UNIQUE(symbol, date)
);

-- ... (8 more tables)
```

**Advantages:**
- ✅ ACID transactions
- ✅ Complex queries with SQL
- ✅ Better performance at scale
- ✅ Concurrent access
- ✅ Data integrity constraints
- ✅ Backup and recovery tools

---

## 🎯 Design Principles

### 1. **Normalization**
- Minimize data redundancy
- Each entity has a clear purpose
- Relationships via foreign keys

### 2. **Scalability**
- Support multiple cryptocurrencies
- Handle millions of OHLCV records
- Store unlimited predictions

### 3. **ML-Friendly**
- Separate raw data from features
- Track model versions and performance
- Store training metadata

### 4. **Auditability**
- Timestamps on all records
- Track prediction accuracy
- Maintain training history

### 5. **Flexibility**
- JSON columns for variable data
- Support multiple model types
- Extensible architecture

---

## 📈 Typical Data Volumes

### Current System (10 Coins, 6 Months)

| Entity | Records | Storage |
|--------|---------|---------|
| Cryptocurrency | 10 | ~2 KB |
| OHLCV_Data | 1,850 | ~450 KB |
| Technical_Indicators | 1,850 | ~300 KB |
| Feature_Engineering | 1,850 | ~500 KB |
| ML_Models | 30 (3×10) | ~5 MB |
| Training_Sessions | 30 | ~10 KB |
| Predictions | ~500/day | ~50 KB/day |
| Portfolio_Performance | 10 | ~5 KB |
| API_Cache | 20 | ~100 KB |
| Backtest_Results | 30 | ~15 KB |

**Total:** ~6.5 MB (current)

### Projected (100 Coins, 2 Years)

| Entity | Records | Storage |
|--------|---------|---------|
| Cryptocurrency | 100 | ~20 KB |
| OHLCV_Data | 73,000 | ~20 MB |
| Technical_Indicators | 73,000 | ~15 MB |
| Feature_Engineering | 73,000 | ~25 MB |
| ML_Models | 300 | ~50 MB |
| Training_Sessions | 1,000 | ~500 KB |
| Predictions | ~365K/year | ~40 MB/year |
| Portfolio_Performance | 100 | ~50 KB |
| API_Cache | 200 | ~2 MB |
| Backtest_Results | 300 | ~150 KB |

**Total:** ~150 MB (2 years)

---

## 🔄 Migration Strategy

### Phase 1: Parallel Running (Current)
- Keep JSON files
- Test database schema
- Validate data consistency

### Phase 2: Dual Write
- Write to both JSON and database
- Read from JSON (primary)
- Verify database integrity

### Phase 3: Gradual Migration
- Migrate historical data
- Switch reads to database
- Keep JSON as backup

### Phase 4: Database Primary
- Database becomes source of truth
- Archive JSON files
- Remove dual-write logic

**Timeline:** Can be done incrementally over 2-4 weeks

---

## 📋 Summary

### ERD Highlights

✅ **10 Well-Designed Entities** covering all aspects of trading system  
✅ **Clear Relationships** with proper foreign keys  
✅ **Normalized Structure** minimizing redundancy  
✅ **ML-Optimized** with separate feature and model tracking  
✅ **Scalable Design** ready for growth  
✅ **Migration-Ready** schema for PostgreSQL  

### Key Entities

1. **Cryptocurrency** - Master data for assets
2. **OHLCV_Data** - Historical prices
3. **Technical_Indicators** - Calculated metrics
4. **ML_Models** - Trained models
5. **Predictions** - Trading signals
6. **Portfolio_Performance** - Results tracking

### Current Status

- **Storage:** File-based (JSON)
- **Data Volume:** ~6.5 MB (10 coins, 6 months)
- **Ready for:** PostgreSQL migration when needed

---

## 📞 Quick Reference

**View Diagrams:**
```bash
# Comprehensive ERD (10 entities)
diagrams/erd_diagram.png

# Simplified ERD (5 core entities)
diagrams/erd_simplified.png
```

**Regenerate Diagrams:**
```bash
python generate_erd_diagram.py
```

**Current Data Files:**
```bash
data/top_coins_cache.json          # Cryptocurrency data
data/crypto_top10_cache.json       # OHLCV data
models/model_cache/*.pkl           # ML models
```

---

**Created:** November 19, 2025  
**Status:** Production-ready schema  
**Migration:** Optional (file-based works well for current scale)
