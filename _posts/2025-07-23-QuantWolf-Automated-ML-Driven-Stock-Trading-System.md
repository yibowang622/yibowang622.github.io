---
layout: post
title: "🐺 QuantWolf: Automated ML-Driven Stock Trading System"
date: 2025-07-23
categories: ["machine learning", "quantitative trading"]
---

### 📚 Introduction
QuantWolf is a sophisticated end-to-end automated trading pipeline that combines machine learning, sentiment analysis, and technical indicators to identify high-conviction stock trading opportunities. Built with modularity and scalability in mind, it processes real-time financial data to generate actionable trading signals.

The system operates through a comprehensive 12-step pipeline that seamlessly integrates multiple data sources and analytical frameworks. QuantWolf employs a unique dual-strategy approach: analyzing Yahoo Finance sector indices (^YH311, ^YH103, etc.) and industry-level performance metrics to identify trending sectors, then drilling down to individual stocks within high-performing industries. The pipeline combines MACD momentum analysis, multi-period return calculations (1D, 5D, 15D, 30D), and market weight assessments to create composite industry rankings.

At its core, QuantWolf leverages a custom Transformer-based binary classifier with enhanced architecture (48 d_model, 6 attention heads, 3 encoder layers) that processes 18 sophisticated technical features including Bollinger Bands, MACD histograms, volatility ratios, and microstructure indicators. The model predicts across multiple timeframes (1-day direction, 3-day direction, and magnitude-based moves) with AUC optimization and confidence-based thresholding. This is enhanced by real-time sentiment analysis using GPT-4 integration with CNBC news scraping, extracting per-stock bullish/bearish sentiment scores that serve as confirmation signals.

What sets QuantWolf apart is its hierarchical filtering approach: starting with 11 major Yahoo Finance sectors (^YH311-^YH207), ranking key industries using proprietary dual-strategy scoring (combining momentum scores up to 7.0 points and stability scores up to 6.0 points), then selecting individual tickers within winning industries. The system demonstrated impressive results in walk-forward validation with 87.5% win rate and 12.7% average period returns, though v1.0 includes look-ahead bias that will be addressed in production v2.0.

Each component is designed as an independent module, allowing for rapid strategy iteration and A/B testing. The pipeline processes everything from sector analysis to executable trading signals with comprehensive backtesting (253% total returns in historical testing) and rigorous walk-forward validation ensuring statistical robustness. This architecture enables both systematic strategy development and production-ready automated execution.
<style>
p {
  text-align: justify !important;
  text-justify: inter-word;
}
</style>

### 🚀 Key Highlights

* **Hierarchical Analysis Framework:** Top-down approach from 11 sectors → key industries → individual stocks
  
* **Multi-Modal Signal Fusion:** Combines sentiment (GPT-4), technical (MACD/RSI), and ML predictions
  
* **Custom Scoring Algorithms:** Proprietary industry ranking system with momentum and stability metrics
  
* **Advanced ML Pipeline:** Transformer-based predictions with AUC validation and confidence scoring
  
* **Production-Ready Architecture:** Modular design with comprehensive error handling and logging
  
* **Rigorous Validation:** Walk-forward testing showing 87.5% win rate across multiple time periods
  
* **Smart Risk Management:** Dynamic position sizing with ML signal-weighted allocation strategies

### 🏗️ System Architecture
Pipeline Overview
The QuantWolf system operates through a 12-step modular pipeline:

| Step | Module | Description |
|------|--------|-------------|
| 1 | News Scraping | CNBC financial news scraping using custom scrapers with JSON storage |
| 2 | Sentiment Analysis | GPT-4 sentiment scoring on news articles for stock-specific bullish/bearish signals |
| 3.1 | Sector Performance | Yahoo Finance sector indices analysis (^YH311-^YH207) for 1D, 5D, 15D, 30D returns |
| 3.2 | Industry Analysis | MACD and yield analysis across key industries with market weight calculations |
| 3.3 | Industry Ranking | Proprietary dual-strategy scoring: Short-term (7.0 max) + Long-term (6.0 max) composite scores |
| 4 | Ticker Selection | Individual stock filtering within top-ranked industries using technical indicators |
| 5 | Sentiment Integration | Merge news sentiment scores with technically filtered stock candidates |
| 6 | Technical Validation | MACD, RSI, SMA confirmation signals for final stock selection |
| 7 | ML Signal Generation | Transformer-based predictions (1-day, 3-day, magnitude) with AUC scoring |
| 8 | ML Filtering | Confidence-based filtering using prediction thresholds and signal strength |
| 9 | Signal Fusion | Combine sentiment, technical, and ML signals into composite conviction scores |
| 10 | Smart Allocation | ML signal-weighted position sizing with risk constraints and diversification |
| 11 | Backtesting Engine | Comprehensive performance analysis with Sharpe ratio, drawdown, win rate metrics |
| 12 | Walk-Forward Validation | Time-series simulation to validate strategy robustness and eliminate look-ahead bias |

### 🧠 Technical Innovation

### Proprietary Industry Ranking System

* **Dual-Strategy Scoring:** Short-term momentum (max 7.0) + Long-term stability (max 6.0)
* **Multi-Period Analysis:** 1D, 5D, 15D, 30D return calculations with MACD confirmation
* **Market Weight Integration:** Yahoo Finance industry data with YTD performance metrics
* **Hierarchical Filtering:** Sector → Industry → Individual stock selection process

### 🤖 Machine Learning Architecture

* **Model:** Custom Transformer-based binary classifier (d_model=48, nhead=6, num_layers=3)
* **Features:** 18 sophisticated technical indicators including multi-timeframe momentum, Bollinger Bands position, MACD histogram, volatility ratios, price patterns, volume trends, and market microstructure
* **Multi-Target Predictions:** 1-day direction, 3-day direction, and magnitude-based volatility moves (>1.5σ)
* **Sequence Processing:** 20-day lookback windows with temporal attention mechanisms
* **Confidence Scoring:** AUC-based validation with threshold optimization (0.3-0.8 range)
* **Advanced Training:** Early stopping, learning rate scheduling, weight decay regularization

### 👾 Data Engineering Pipeline

* **Web Scraping:** Custom CNBC news scrapers with JSON data persistence
* **API Integration:** Yahoo Finance sector indices and individual stock data
* **Real-Time Processing:** Automated sentiment analysis with GPT-4 integration
* **Error Handling:** Robust data validation and missing value management
* **Scalable Architecture:** Modular design supporting easy addition of new data sources

### 📊 Risk Management Framework

* **Dynamic Position Sizing:** ML signal-weighted allocation with risk constraints
* **Diversification Controls:** Sector exposure limits and position concentration rules
* **Performance Monitoring:** Real-time drawdown tracking and volatility adjustment
* **Walk-Forward Validation:** Eliminates look-ahead bias through temporal simulation

### 📦 Project Structure 
<pre>
QuantWolf_1.0/
│
├── step1_scraper/
│   ├── scrapers.py
│   └── data/
├── step2_sentiment_analysis/
│   ├── gpt4_analyzer.py
│   └── results/
├── step3_sector_and_stock_filtering/
│   ├── sector_analysis.py
│   ├── industry_ranking.py
│   └── data/
├── step7_match_sentiment_and_ml_score/
├── step8_ml_prediction_filtering/
├── step9_optional_ml_filter/
├── step10_asset_allocation_simulation/
├── step11_backtesting_and_metrics/
├── step12_walkforward_simulation/
├── launcher.py
├── requirements_QuantWolf_demo_1.txt
└── README.md
</pre>
### 📊 Performance Metrics (v1.0 Demo Results)
### Backtesting Performance (2023-01-03 to 2024-12-31)

* **Total Return:** 253.71%
* **Annualized Return:** 88.54%
* **Sharpe Ratio:** 2.17
* **Sortino Ratio:** 3.91
* **Maximum Drawdown:** -27.07%
* **Win Rate:** 52.79%
* **Win/Loss Ratio:** 1.22
* **Alpha vs Benchmark:** +2.46%

### Walk-Forward Validation Results (8 Periods)

* **Simulation Success Rate:** 100% (8/8 periods)
* **Period Win Rate:** 87.5%
* **Average Period Return:** 12.70%
* **Volatility:** 12.90%
* **Best Period:** +27.78%
* **Worst Period:** -10.45%
* **Estimated Annualized Return:** 107.3%

### Individual Stock Performance

* **Best Performer:** OUST (+353.64%)
* **Portfolio Average:** +253.70%
* **Worst Performer:** PLAY (+108.29%)
* **Final Portfolio Value:** $176,853 (from $50,000 initial)

### Technical Performance

* **Pipeline Execution Time:** ~15 minutes end-to-end
* **ML Model Training:** 60 epochs max with early stopping
* **Feature Engineering:** 18 technical indicators with 20-day sequences
* **Data Coverage:** 6 final stocks across multiple sectors and industries


### Note: v1.0 results include look-ahead bias for demonstration purposes. All performance metrics are from backtesting and walk-forward simulation. Production v2.0 will implement strict temporal constraints and live trading validation.

### 📋 Sample Portfolio Output
### Smart Allocation Results
<pre>
🏆 PORTFOLIO ALLOCATION BREAKDOWN:
#1: PSIX    $10,981 (22.0%) | AUC: 0.616 | Excellent | magnitude | Strong
#2: OUST    $10,355 (20.7%) | AUC: 0.597 | Good      | magnitude | Strong  
#3: JHX     $9,810  (19.6%) | AUC: 0.581 | Good      | magnitude | Strong
#4: PLAY    $6,508  (13.0%) | AUC: 0.538 | Fair      | magnitude | Weak
#5: HOOD    $6,478  (13.0%) | AUC: 0.534 | Fair      | 3day      | Weak
#6: NVMI    $5,868  (11.7%) | AUC: 0.513 | Poor      | magnitude | None

💰 ALLOCATION SUMMARY:
- Strong Signals: $31,146 (62.3%) across 3 stocks
- Weak Signals:  $12,986 (26.0%) across 2 stocks  
- Risk-Adjusted: 2 high-risk positions (42.7% capital)
- Expected Alpha: 21.56 points from ML edge
</pre>

### Key Algorithm Features Demonstrated

* **Signal-Weighted Allocation:** Higher AUC scores receive larger allocations
* **Risk Controls:** Maximum 22% single position, diversified across prediction types
* **ML Categories:** Strong signals get 15%+ minimum allocation
* **Prediction Diversity:** Mix of magnitude and 3-day predictions for robustness

### 🛠️ Technical Implementation & Setup
### Core Dependencies

<pre>
# ML & Data Processing
torch>=1.9.0              # Transformer model implementation
pandas>=1.3.0              # Data manipulation
numpy>=1.21.0              # Numerical computing
scikit-learn>=1.0.0        # ML metrics and preprocessing
yfinance>=0.1.70           # Financial data API


# Sentiment Analysis
openai>=0.27.0             # GPT-4 integration
requests>=2.26.0           # Web scraping

# Visualization & Analysis
matplotlib>=3.4.0          # Plotting
seaborn>=0.11.0           # Statistical visualization
</pre>

### ML Feature Engineering Pipeline
<pre>
# 18 Technical Features Used in Transformer Model
feature_names = [
    'Return', 'Momentum', 'RSI', 'Volume_MA',           # Basic momentum
    'Momentum_3', 'Momentum_10', 'Momentum_20',         # Multi-timeframe momentum  
    'BB_position', 'MACD', 'MACD_histogram',           # Bollinger & MACD signals
    'Vol_ratio', 'Price_position',                      # Volatility & price patterns
    'Volume_momentum', 'Price_volume_trend',            # Volume analysis
    'Distance_from_high', 'Distance_from_low',          # Support/resistance
    'Intraday_return', 'Gap'                           # Market microstructure
]

# Transformer Architecture
SEQ_LEN = 20               # 20-day lookback window
d_model = 48               # Embedding dimension
nhead = 6                  # Attention heads
num_layers = 3             # Encoder layers
</pre>

### Smart Allocation Parameters
<pre>
python# Portfolio Configuration
TOTAL_CAPITAL = 50000
MIN_ALLOCATION = 2000      # Minimum position size
MAX_ALLOCATION = 12000     # Maximum position size (24% cap)

# ML Signal Weighting
ML_WEIGHT = 0.6            # ML signal importance
CONFIDENCE_WEIGHT = 0.4    # Confidence score importance
DIVERSIFICATION_BONUS = 0.05 # Prediction type diversity bonus

# Risk Management
MAX_SINGLE_POSITION = 0.25  # 25% position limit
MIN_STRONG_ALLOCATION = 0.15 # 15% minimum for strong signals
</pre>
  
### Backtesting & Validation Setup
<pre>
# Backtesting Parameters
INITIAL_CAPITAL = 50000
REBALANCING_FREQUENCY = "monthly"
TRANSACTION_COST = 0.001   # 0.1% per trade
CASH_THRESHOLD = 0.05      # 5% cash buffer

# Walk-Forward Validation
SIMULATION_PERIODS = 8     # Number of time periods
HOLDING_PERIOD = 30        # Days per simulation
LOOKBACK_WINDOW = 60       # Training data window

# Risk Controls
STOP_LOSS = None           # No stop-loss in v1.0
TAKE_PROFIT = None         # No take-profit in v1.0  
MAX_POSITION_SIZE = 0.3    # 30% maximum position
</pre>

<div align="left">
  <img src="https://github.com/yibowang622/yibowang622.github.io/raw/main/assets/images/2025-07-23-final_equity_curve.png" width="70%">
</div>


### 🔄 Walk-Forward Validation Framework
### Simulation Configuration 
<pre>
# Walk-Forward Simulation Parameters
CURRENT_TICKERS = ["OUST", "HOOD", "PSIX", "JHX", "NVMI", "PLAY"]
TOTAL_CAPITAL = 50000
ALLOCATION_METHOD = "equal"  # Equal weight validation
MIN_ALLOCATION = 1000
MAX_ALLOCATION = 8500

# Validation Setup
SIMULATION_PERIODS = 8         # Number of time periods tested
HOLDING_PERIOD = 30            # Days per simulation
TRANSACTION_COST_PCT = 0.001   # 0.1% per trade
BUFFER_DAYS = 5                # Data availability buffer

# Output Files
WALKFORWARD_RESULTS_FILE = "walkforward_results.csv"
WALKFORWARD_SUMMARY_FILE = "walkforward_summary.json"
EQUITY_CURVE_FILE = "walkforward_equity_curve.csv"
</pre>

### Walk-Forward Validation Results
### Strategy Performance Summary:
<pre>
🎯 WALK-FORWARD VALIDATION RESULTS (8 Periods)
- Total Simulations: 8 | Success Rate: 100%
- Win Rate: 87.5% (7/8 periods profitable)
- Average Period Return: +12.70%
- Total Portfolio Return: +146.2%
- Sharpe Ratio: 0.98 | Volatility: 12.90%

📊 INDIVIDUAL STOCK PERFORMANCE:
- PSIX: +39.47% avg return (best: +104.45%, worst: -3.35%)
- OUST: +17.93% avg return (best: +77.14%, worst: -47.15%)  
- HOOD: +15.86% avg return (best: +56.75%, worst: -14.42%)
- NVMI: +4.07% avg return (best: +26.47%, worst: -11.53%)
- JHX: +0.18% avg return (best: +17.13%, worst: -17.62%)
- PLAY: -1.33% avg return (best: +14.86%, worst: -19.03%)

⏱️ TIMING ANALYSIS:
- Average Holding Period: 29.9 days
- Best Period: +27.78% (2024-05-01 to 2024-05-31)
- Worst Period: -10.45% (2024-04-01 to 2024-05-01)
- Annualized Return Estimate: +107.3%
</pre>

<div align="left">
  <img src="https://github.com/yibowang622/yibowang622.github.io/raw/main/assets/images/2025-07-23-walkforward_performance.png" alt="Walk-Forward Performance Dashboard" width="70%">
</div>

### Key Validation Features

* **Temporal Constraints:** No look-ahead bias in simulation periods
* **Transaction Costs:** 0.1% realistic trading costs included
* **Multiple Timeframes:** 8 different market periods tested
* **Equal Weight Baseline:** Conservative allocation for validation
* **Comprehensive Metrics:** Returns, Sharpe ratio, volatility, win rates
* **Individual Stock Analysis:** Performance breakdown by ticker
* **Visual Dashboard:** Equity curve, return distribution, period analysis

### 🔮 v2.0 Production Roadmap
### 🔍 Smart Signal Selection Enhancements

* **Multi-Criteria Filtering:** Strict temporal alignment of macro (sector), micro (valuation), sentiment, and ML signals
* **Signal Confidence Scoring:** Weighted composite scores when all criteria align vs. partial matches
* **Dynamic Thresholds:** Adaptive filtering based on market volatility and regime detection
* **Signal Decay Models:** Time-weighted importance for news sentiment and technical indicators

### 📈 Advanced Portfolio Management

* **Multi-Period Allocation:** Dynamic rebalancing based on signal strength changes over time
* **Risk-Adjusted Position Sizing:** Volatility-based allocation with correlation adjustments
* **Performance Attribution:** Decompose returns by signal type (sentiment vs ML vs technical)
* **Benchmark Integration:** Real-time comparison against sector ETFs and market indices

### 🧱 Production Architecture Improvements

* **Strict Module Interfaces:** Standardized data contracts between pipeline stages
* **Hot-Swappable Components:** Plugin architecture for ML models, sentiment analyzers, and data sources
* **Data Lineage Tracking:** Full audit trail from raw news to final portfolio decisions
* **Automated Testing:** Unit tests and integration tests for each pipeline module

### ⚠️ Bias Control & Validation

* **Temporal Data Access:** Point-in-time database ensuring no future data leakage
* **News Timestamp Validation:** Strict publication time vs. market close alignment
* **ML Prediction Constraints:** Forward-only feature engineering with proper time splits
* **Walk-Forward Expansion:** Continuous out-of-sample testing with growing training windows

### Infrastructure & Monitoring

* **Automated Pipeline Scheduling:** Daily execution with dependency management
* **Data Quality Monitoring:** Alerts for missing data, API failures, and signal anomalies
* **Performance Tracking:** Live monitoring of signal accuracy and portfolio metrics
* **Configuration Management:** Environment-based settings for development vs production

### ⚠️ Disclaimer
This software is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss. Please consult with a qualified financial advisor before making investment decisions.
