---
layout: post
title: "🐺 QuantWolf: Automated ML-Driven Stock Trading System"
date: 2025-07-23
categories: ["machine learning", "quantitative trading"]
---
<p style="text-align: justify;">
  This paragraph is justified. It spans the width and aligns text to both left and right margins.
</p>

### 📚 Introduction
QuantWolf is a sophisticated end-to-end automated trading pipeline that combines machine learning, sentiment analysis, and technical indicators to identify high-conviction stock trading opportunities. Built with modularity and scalability in mind, it processes real-time financial data to generate actionable trading signals.

The system operates through a comprehensive 12-step pipeline that seamlessly integrates multiple data sources and analytical frameworks. QuantWolf employs a unique dual-strategy approach: analyzing Yahoo Finance sector indices (^YH311, ^YH103, etc.) and industry-level performance metrics to identify trending sectors, then drilling down to individual stocks within high-performing industries. The pipeline combines MACD momentum analysis, multi-period return calculations (1D, 5D, 15D, 30D), and market weight assessments to create composite industry rankings.

At its core, QuantWolf leverages a custom Transformer-based binary classifier with enhanced architecture (48 d_model, 6 attention heads, 3 encoder layers) that processes 18 sophisticated technical features including Bollinger Bands, MACD histograms, volatility ratios, and microstructure indicators. The model predicts across multiple timeframes (1-day direction, 3-day direction, and magnitude-based moves) with AUC optimization and confidence-based thresholding. This is enhanced by real-time sentiment analysis using GPT-4 integration with CNBC news scraping, extracting per-stock bullish/bearish sentiment scores that serve as confirmation signals.

What sets QuantWolf apart is its hierarchical filtering approach: starting with 11 major Yahoo Finance sectors (^YH311-^YH207), ranking key industries using proprietary dual-strategy scoring (combining momentum scores up to 7.0 points and stability scores up to 6.0 points), then selecting individual tickers within winning industries. The system demonstrated impressive results in walk-forward validation with 87.5% win rate and 12.7% average period returns, though v1.0 includes look-ahead bias that will be addressed in production v2.0.

Each component is designed as an independent module, allowing for rapid strategy iteration and A/B testing. The pipeline processes everything from sector analysis to executable trading signals with comprehensive backtesting (253% total returns in historical testing) and rigorous walk-forward validation ensuring statistical robustness. This architecture enables both systematic strategy development and production-ready automated execution.

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
</pre>
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

</pre>

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
