# IntelliTradeAI System Architecture

## Class Diagram (Mermaid)

```mermaid
classDiagram
    %% Data Layer
    class DataIngestion {
        -coinmarketcap_api_key: str
        -coinmarketcap_base_url: str
        -yahoo_timeout: int
        +fetch_crypto_data(symbols, period, interval)
        +fetch_stock_data(symbols, period, interval)
        +fetch_mixed_data(crypto_symbols, stock_symbols, period, interval)
        -_fetch_crypto_current_price(symbol)
    }

    %% ML Models Layer
    class MLPredictor {
        -models: Dict
        -scalers: Dict
        +load_model(symbol)
        +predict(symbol, data)
        +get_signal(prediction, confidence)
        +calculate_confidence(probabilities)
    }

    class RandomForestModel {
        -n_estimators: int
        -max_depth: int
        +train(X, y)
        +predict(X)
        +predict_proba(X)
        +feature_importances_
    }

    class XGBoostModel {
        -n_estimators: int
        -learning_rate: float
        +train(X, y)
        +predict(X)
        +predict_proba(X)
    }

    %% AI Advisor Layer
    class SignalFusionEngine {
        -ml_weight: float
        -pattern_weight: float
        +fuse_signals(ml_signal, pattern_signal)
        +resolve_conflict(ml_confidence, pattern_confidence)
        +generate_unified_signal()
    }

    class ChartPatternRecognizer {
        +detect_patterns_from_data(df, symbol)
        +identify_support_resistance(df)
        +calculate_risk_reward(entry, target, stop_loss)
    }

    class PriceLevelAnalyzer {
        +calculate_levels(df)
        +get_support_levels(df)
        +get_resistance_levels(df)
    }

    %% Trading Layer
    class TradingModeManager {
        -current_mode: TradingMode
        -asset_modes: Dict
        -auto_trade_config: Dict
        +switch_mode(new_mode)
        +set_asset_mode(asset_type, mode)
        +get_asset_mode(asset_type)
        +should_execute_trade(signal)
    }

    class TradeExecutor {
        -db_manager: DBManager
        +execute_stock_trade(symbol, action, quantity, price)
        +execute_crypto_trade(symbol, action, quantity, price)
        +execute_option_trade(symbol, option_type, strike, premium)
    }

    %% Sentiment Layer
    class FearGreedIndexAnalyzer {
        +get_crypto_index()
        +get_stock_index()
        +get_options_index()
        +get_overall_sentiment()
        +classify_sentiment(value)
    }

    %% UI Layer
    class ChartToolbar {
        -chart_id: str
        -active_tool: str
        -drawings: List
        -indicators: List
        +render_toolbar(chart_key)
        +render_indicator_panel(chart_key)
    }

    %% Security Layer
    class SecureAuthManager {
        -secret_key: str
        +authenticate_user(username, password, totp)
        +register_user(username, email, password)
        +generate_2fa_secret()
    }

    class SecureWalletManager {
        +create_ethereum_wallet(password)
        +get_wallet_balance(address)
        +generate_wallet_qr_code(address)
    }

    %% Database Layer
    class DBManager {
        -connection_url: str
        +execute_query(sql, params)
        +insert_trade(trade_data)
        +get_positions()
        +get_portfolio()
    }

    %% Relationships
    DataIngestion --> MLPredictor : provides data
    MLPredictor --> RandomForestModel : uses
    MLPredictor --> XGBoostModel : uses
    MLPredictor --> SignalFusionEngine : feeds signals
    ChartPatternRecognizer --> SignalFusionEngine : feeds patterns
    SignalFusionEngine --> TradingModeManager : sends unified signal
    TradingModeManager --> TradeExecutor : triggers execution
    TradeExecutor --> DBManager : logs trades
    FearGreedIndexAnalyzer --> ChartToolbar : sentiment overlay
    SecureAuthManager --> DBManager : user management
    SecureWalletManager --> TradeExecutor : crypto execution
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐         ┌─────────────────────┐                    │
│  │   Yahoo Finance     │         │   CoinMarketCap     │                    │
│  │   (Free API)        │         │   (Paid API)        │                    │
│  ├─────────────────────┤         ├─────────────────────┤                    │
│  │ - Historical OHLCV  │         │ - Real-time prices  │                    │
│  │ - 5-10 years data   │         │ - Market cap        │                    │
│  │ - Stocks & Crypto   │         │ - Top 10 coins      │                    │
│  │ - Daily intervals   │         │ - 24h changes       │                    │
│  └──────────┬──────────┘         └──────────┬──────────┘                    │
│             │                               │                                │
│             └───────────────┬───────────────┘                                │
│                             ▼                                                │
│                  ┌─────────────────────┐                                    │
│                  │   DataIngestion     │                                    │
│                  │   (Hybrid Fetcher)  │                                    │
│                  └──────────┬──────────┘                                    │
└─────────────────────────────┼───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA PROCESSING                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Technical Indicators Engine                       │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  SMA (20, 50, 200)  │  EMA (12, 26)  │  Bollinger Bands  │  RSI    │    │
│  │  MACD               │  ATR           │  Fibonacci        │  Volume │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│              ┌───────────────┼───────────────┐                              │
│              ▼               ▼               ▼                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                   │
│  │ Random Forest │  │   XGBoost     │  │    LSTM       │                   │
│  │   Model       │  │   Model       │  │   (Optional)  │                   │
│  │ Acc: 47-79%   │  │ Acc: 50-82%   │  │ Acc: 55-75%   │                   │
│  └───────────────┘  └───────────────┘  └───────────────┘                   │
│              │               │               │                              │
│              └───────────────┼───────────────┘                              │
│                              ▼                                               │
│                  ┌─────────────────────┐                                    │
│                  │  Signal Fusion      │                                    │
│                  │  Engine             │                                    │
│                  │  (Conflict Resolver)│                                    │
│                  └──────────┬──────────┘                                    │
└─────────────────────────────┼───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRADING EXECUTION                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐         ┌─────────────────────┐                    │
│  │  Trading Mode       │         │  Trade Executor     │                    │
│  │  Manager            │────────▶│                     │                    │
│  ├─────────────────────┤         ├─────────────────────┤                    │
│  │ Manual: User decide │         │ Stocks: Market/Limit│                    │
│  │ Auto: AI executes   │         │ Crypto: Blockchain  │                    │
│  │ Per-asset toggles   │         │ Options: Chain API  │                    │
│  └─────────────────────┘         └─────────┬───────────┘                    │
│                                            │                                │
│                                            ▼                                │
│                              ┌─────────────────────┐                        │
│                              │  PostgreSQL Database │                        │
│                              │  (Trade Log & P&L)   │                        │
│                              └─────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Sources Summary

| Source | Type | Data Provided | Period Available | Cost |
|--------|------|---------------|------------------|------|
| Yahoo Finance | REST API | Historical OHLCV, Volume | Up to 10+ years | Free |
| CoinMarketCap | REST API | Real-time prices, Market Cap, 24h Change | Current | Paid (API Key Required) |

### Yahoo Finance Symbols
- **Stocks**: Direct ticker (e.g., AAPL, MSFT, GOOGL)
- **Crypto**: Ticker-USD format (e.g., BTC-USD, ETH-USD)

### Supported Assets (38 Total)

**Cryptocurrencies (20):**
BTC, ETH, USDT, XRP, BNB, SOL, USDC, TRX, DOGE, ADA, AVAX, SHIB, TON, DOT, LINK, BCH, LTC, XLM, WTRX, STETH

**Stocks (18):**
AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, WMT, JNJ, V, BAC, DIS, NFLX, INTC, AMD, CRM, ORCL

## Model Performance Metrics

### Current Model Accuracy (Random Forest - 1 Year Data)

| Asset Type | Min Accuracy | Max Accuracy | Average |
|------------|-------------|--------------|---------|
| Cryptocurrencies | 47% | 72% | 58% |
| Stocks | 52% | 79% | 64% |

### Expected Improvement with Extended Data (5-10 Years)

| Metric | 1 Year Data | 5 Year Data | 10 Year Data |
|--------|-------------|-------------|--------------|
| Training Samples | ~250 | ~1,250 | ~2,500 |
| Pattern Recognition | Limited | Better seasonality | Full cycle coverage |
| Model Stability | Moderate | High | Very High |
| Expected Accuracy Gain | Baseline | +8-12% | +12-18% |

### Model Ensemble Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENSEMBLE PREDICTION                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Random Forest (40%)  +  XGBoost (40%)  +  Pattern AI (20%)   │
│         │                    │                   │              │
│         ▼                    ▼                   ▼              │
│   ┌───────────┐        ┌───────────┐      ┌───────────┐        │
│   │ Tree-based│        │ Gradient  │      │ Technical │        │
│   │ Voting    │        │ Boosting  │      │ Patterns  │        │
│   └───────────┘        └───────────┘      └───────────┘        │
│         │                    │                   │              │
│         └────────────────────┼───────────────────┘              │
│                              ▼                                   │
│                    ┌─────────────────┐                          │
│                    │ Signal Fusion   │                          │
│                    │ Engine          │                          │
│                    ├─────────────────┤                          │
│                    │ BUY | SELL | HOLD                          │
│                    │ Confidence: 0-100%                         │
│                    └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

### Feature Engineering (70+ Features)

| Category | Features | Description |
|----------|----------|-------------|
| Price | Open, High, Low, Close | Raw OHLC data |
| Volume | Volume, Volume MA | Trading volume indicators |
| Moving Averages | SMA 20/50/200, EMA 12/26 | Trend indicators |
| Momentum | RSI, MACD, Stochastic | Overbought/oversold |
| Volatility | Bollinger Bands, ATR | Price volatility |
| Patterns | Head & Shoulders, Wedges | Chart patterns |
| Time | Day of week, Month | Seasonal features |

## TradingView-Style Chart Toolbar

### Available Tools

| Category | Tools | Description |
|----------|-------|-------------|
| Drawing | Trendline, H-Line, V-Line, Ray, Channel | Manual drawing tools |
| Fibonacci | Retracement, Extension, Fan | Fibonacci analysis |
| Shapes | Rectangle, Ellipse, Triangle, Arrow | Shape annotations |
| Indicators | SMA, EMA, Bollinger, RSI, MACD, Volume | Technical overlays |

### Time Ranges Available

- 1 Day, 1 Week, 1 Month, 3 Months, 6 Months
- 1 Year, 2 Years, **5 Years**, 10 Years, Max

---

*Last Updated: December 2025*
*Data Period: Extended to 5-10 years for improved model training*
