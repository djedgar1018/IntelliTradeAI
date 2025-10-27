# IntelliTradeAI Use Case Diagrams

This folder contains professional UML use case diagrams showing how different users interact with the IntelliTradeAI trading bot system.

## 📊 Diagram Overview

### 1. Core Trading Operations (`01_core_trading_use_case.png`)
**Purpose**: Shows how traders get predictions and trading signals

**Actors**:
- Day Trader (red) - Fast-paced trading decisions
- Swing Trader (green) - Multi-day trend analysis
- Long-term Investor (blue) - Validation and decision support

**Key Use Cases**:
- Get Instant Prediction
- View BUY/SELL Signal
- View Confidence Score
- Monitor Watchlist
- Check Signal History

**Relationships**:
- `<<include>>` - Getting predictions requires selecting an asset and viewing confidence scores

---

### 2. Model Management (`02_model_management_use_case.png`)
**Purpose**: Shows how users train and manage AI models

**Actors**:
- Data Scientist (purple) - Advanced model development
- Advanced User (yellow) - Model training and optimization
- System Scheduler (gray) - Automated retraining

**Key Use Cases**:
- Train New Model
- Select Algorithm (Random Forest, XGBoost, LSTM)
- Compare Models
- View Accuracy Metrics
- Manage Model Cache
- Auto-Retrain Models

**Relationships**:
- `<<include>>` - Training requires selecting algorithm and fetching data
- `<<extend>>` - Auto-retraining extends the core training functionality

---

### 3. Analytics & Risk Management (`03_analytics_risk_use_case.png`)
**Purpose**: Shows professional analytics and risk control features

**Actors**:
- Portfolio Manager (red) - Performance tracking and reporting
- Financial Advisor (green) - Client recommendation support
- Risk Analyst (orange) - Risk metrics and controls

**Key Use Cases**:
- Run Backtest
- Calculate Risk Metrics
- Set Stop-Loss Levels
- Set Take-Profit Levels
- Track Portfolio P&L
- View SHAP Analysis (explainability)

**Relationships**:
- `<<include>>` - Backtesting includes performance metrics
- `<<extend>>` - Take-profit extends stop-loss functionality

---

### 4. API & Automation (`04_api_automation_use_case.png`)
**Purpose**: Shows how developers integrate the system via API

**Actors**:
- Algorithm Developer (purple) - Building custom trading algorithms
- Trading Bot (gray) - Automated trading systems
- External System (yellow) - Third-party integrations

**Key Use Cases**:
- Access REST API
- Authenticate API Key
- Get Predictions via API
- Trigger Model Retrain
- Receive Webhooks
- Execute Automated Trades

**Relationships**:
- `<<include>>` - API access requires authentication
- `<<extend>>` - Predictions can extend to fetch fresh market data

---

### 5. System Overview (`05_system_overview_use_case.png`)
**Purpose**: Complete bird's-eye view of the entire platform

**Actors**: All user types from previous diagrams
- Day Trader, Portfolio Manager, Data Scientist (left side)
- Financial Advisor, Trading Bot, External System (right side)
- Algorithm Developer (left side)

**Key Features Shown**:
- Core trading functions (predictions, signals, watchlist)
- Model operations (training, comparison, cache management)
- Analytics (backtest, risk analysis, metrics)
- API services (REST access, webhooks, automation)
- Data & technical features (market data, indicators, SHAP, reports)

**Purpose**: Use this diagram for presentations, documentation, or to explain the full system capability to stakeholders.

---

## 🎨 UML Legend

### Visual Elements

**Stick Figures** = Actors (users or external systems)
- Different colors represent different user types
- Positioned outside the system boundary

**Ellipses** = Use Cases (system features)
- Yellow background indicates core functionality
- Positioned inside the system boundary

**Solid Lines** = Associations
- Connect actors to use cases they interact with

**Dashed Arrows with `<<include>>`** = Include Relationships
- The base use case always includes this functionality
- Arrow points from base use case to included use case

**Dashed Arrows with `<<extend>>`** = Extend Relationships
- Optional or conditional functionality
- Arrow points from extension to base use case

**Rectangle Border** = System Boundary
- Defines what's inside IntelliTradeAI vs. external actors

---

## 📖 How to Use These Diagrams

### For Business Presentations
- Use **Diagram 5 (System Overview)** for executive summaries
- Use **Diagram 1 (Core Trading)** to explain value to traders
- Use **Diagram 3 (Analytics)** to show risk management features

### For Technical Documentation
- Use **Diagram 2 (Model Management)** for ML architecture discussions
- Use **Diagram 4 (API & Automation)** for developer onboarding
- Reference specific use cases when writing user stories

### For User Training
- Use **Diagram 1** to teach basic trading features
- Use **Diagram 2** for advanced model training tutorials
- Use **Diagram 3** for risk management workshops

### For Development Planning
- Map sprint tasks to specific use cases in diagrams
- Verify all use cases have corresponding features implemented
- Track which actors can access which functionality

---

## 🔗 Related Documentation

- **USER_MANUAL.md** - Detailed instructions for each use case
- **DEVELOPMENT_ROADMAP.md** - Sprint planning aligned with these diagrams
- **PROJECT_CHANGES_ANALYSIS.md** - Technical implementation details
- **replit.md** - System architecture and technical specifications

---

## 📝 Diagram Maintenance

These diagrams were generated using Python with matplotlib. To regenerate or modify:

```bash
python generate_use_case_diagrams.py
```

**When to Update Diagrams**:
- New user types are identified
- Major features are added to the system
- System boundaries change (new integrations)
- Use case relationships change

**Generator Script**: `generate_use_case_diagrams.py` in the root directory

---

## 💡 Key Insights from Diagrams

1. **Multi-Actor System**: IntelliTradeAI serves 7+ different user types with distinct needs
2. **Layered Functionality**: Core trading → Model management → Analytics → Automation
3. **Include Dependencies**: Many advanced features depend on core prediction capability
4. **External Integration**: Strong API layer enables bot and system integrations
5. **Comprehensive Coverage**: 30+ use cases covering entire trading workflow

---

**Generated**: 2025-10-27  
**Tool**: IntelliTradeAI Use Case Diagram Generator  
**Format**: UML 2.5 Use Case Diagrams  
**Resolution**: 300 DPI PNG images
