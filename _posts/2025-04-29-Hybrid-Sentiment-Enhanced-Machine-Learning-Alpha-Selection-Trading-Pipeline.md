---
layout: post
title: "Hybrid Sentiment-Enhanced Machine Learning + Alpha Selection Trading Pipeline"
date: 2025-04-29
categories: ["machine learning", "quantitative trading"]
colab_notebook: "https://colab.research.google.com/drive/1D42gO8AoCPPcdY56My-kk5NNuD38S23q?usp=sharing"
---
### Introduction
This project presents a complete end-to-end trading system that integrates financial news sentiment analysis, technical indicator-based stock screening, and machine learning probability forecasts.

The goal is to construct a multi-layered decision pipeline that replicates realistic quantitative trading workflows:

* 📚 Scraping financial news articles with Scrapy

* 💬 Analyzing sentiment using FinBERT transformer model

* 📈 Screening stocks using technical indicators (RSI, MACD)

* 🤖 Forecasting market movement probabilities using a Transformer-based model

* 🎯 Generating final trade signals through a double-threshold logic combining Sentiment + Alpha + ML prediction

* 🔄 Backtesting and evaluating strategy performance using Sharpe Ratio, Max Drawdown, and Total Returns

This approach simulates institutional research methodology, enhancing the robustness and interpretability of trading decisions.

### 1. Sentiment Analysis and Aggregation
We scrape financial news articles using Scrapy and apply the FinBERT model to generate sentiment scores for each article.

The complete scraping results can be viewed in [this gist](https://gist.github.com/yibowang622/5f109ca7a5cf6f5a3e2b69e7dc66946f).

To align sentiment with stock trading decisions, we aggregate sentiment scores for each stock over the past 7 days by calculating the average confidence score for positive sentiment. Only stocks with a 7-day average sentiment score greater than 0.5 are considered for further selection.

**Implementation Files:**
- [Download Web Scraper (cnbc_spider.py)](/src/cnbc_spider.py) or can be viewed in [this gist](https://gist.github.com/yibowang622/f5a8cc1752f62e6293810b31d0f891b2).

- [Download Sentiment Aggregation (sentiment_aggregation.py)](/src/sentiment_aggregation.py) or can be viewed in [this gist](https://gist.github.com/yibowang622/735012a9938b7fc1597d120f0ea7acef).


**Sample Results:**

| Stock | avg_sentiment_score | article_count |
|-------|---------------------|---------------|
| BA    | 1.0                 | 1            |
| BRK   | 1.0                 | 1             |
| AMZN  | 0.959               | 1            |
| DB    | 0.667               | 2            |
| META  | 0.667               | 1            |
| NVS   | 0.5                 | 2            |

[Download Sentiment Results](/data/sentiment_scores.csv)

### 2. Technical Indicator Screening
After identifying stocks with positive sentiment, we apply a multi-factor technical screening process to filter for promising trading opportunities. This stage combines sentiment analysis with traditional technical indicators to generate a comprehensive trading signal.

The screening process integrates:

* RSI (Relative Strength Index) to identify oversold or overbought conditions
* MACD (Moving Average Convergence Divergence) to detect momentum shifts
* SMA50 (50-day Simple Moving Average) as a trend filter
* Volume analysis for confirmation of price movements

Our implementation includes robust error handling and rate limiting protection to ensure reliable data collection from financial APIs.

**Implementation Files:**
- [Download Screening (screening.py)](/src/screening.py) or can be viewed in [this gist](https://gist.github.com/yibowang622/29a6a206c5b61db8e7aceeccd7abc91d).

**Sample Results:**

| Ticker | Sentiment | Articles | RSI | MACD_Bullish | Above_SMA50 | Avg_Volume |
|--------|-----------|----------|-----|-------------|-------------|------------|
| BA     | 1.00      | 1        | 48.3 | Yes         | Yes         | 8792622    |
| BRK    | 1.00      | 1        | 52.7 | Yes         | Yes         | 5077701    |
| AMZN   | 0.96      | 1        | 56.8 | Yes         | No          | 49614533   |
| DB     | 0.67      | 2        | 45.2 | Yes         | Yes         | 4239758    |
| META   | 0.67      | 1        | 62.1 | Yes         | No          | 18266843   |
| NVS    | 0.50      | 2        | 41.9 | Yes         | Yes         | 2484374    |

[Download Screening Results](/data/screening_results.csv)

### 3. Alpha Model Construction
The core of our trading strategy is a weighted alpha model that combines sentiment signals with technical indicators. This approach creates a robust scoring system that prioritizes stocks with the highest probability of positive performance.

Our alpha model assigns weights to different components:

* Sentiment score (40%): News sentiment from our FinBERT analysis
* RSI score (20%): Normalized inverse RSI for oversold conditions
* MACD score (20%): Binary signal for bullish crossovers
* Volume score (20%): Relative volume compared to 30-day average

The mathematical formulation of our alpha score is:

```
score = (
    sentiment_score * 0.4 +
    volume_score * 0.2 +
    rsi_score * 0.2 +
    macd_score * 0.2
)
```python

This weighted approach allows us to rank stocks by their combined technical and sentiment signals, creating a daily watchlist of high-potential trading candidates.

**Implementation Files:**
- [Download Alpha Score (alpha_score.py)](/src/alpha_score.py) or can be viewed in [this gist](https://gist.github.com/yibowang622/fd541f3f9359d74626c586276c449621).

**Sample Results:**

| Ticker | alpha_score |
|--------|-------------|
| BA     | 0.800       |
| BRK    | 0.800       |
| DB     | 0.668       |
| NVS    | 0.600       |
| AMZN   | 0.584       |
| META   | 0.468       |

[Download Alpha Score](/data/top_signals.csv)

### 4. Machine Learning Probability Forecasts

To enhance our decision-making process, we implement a Transformer-based machine learning model that generates probability forecasts for directional price movements. This model is trained on historical price patterns and calculates the probability of upward price movement.

Our ML model architecture includes:
* A Transformer encoder with multi-head attention
* Sequence length of 20 days
* Two primary features: price returns and momentum
* Binary classification output (probability of price increase)

**Implementation Files:**
- [Transformer ML Model (transform_model.py)](https://colab.research.google.com/drive/15M9wl1AWaBM-SZkpoTxyILCabBoD5CoO?usp=sharing)

**Sample Results:**

| Ticker | P_up    | Best_Threshold | F1_Score | AUC      | Data_Source |
|--------|---------|----------------|----------|----------|-------------|
| BA     | 0.517137| 0.5            | 0        | 0.485358 | Historic    |
| BRK    | 0.533452| 0.5            | 0        | 0.477690 | Historic    |
| DB     | 0.519094| 0.5            | 0        | 0.489465 | Historic    |
| NVS    | 0.480948| 0.5            | 0        | 0.517180 | Historic    |
| AMZN   | 0.470642| 0.5            | 0        | 0.512692 | Historic    |
| META   | 0.484280| 0.5            | 0        | 0.461475 | Historic    |

[Download ML Predictions](/data/ml_predictions.csv)

The model provides probability values for upward movement (P_up) and model performance metrics (AUC). These outputs serve as the final validation layer in our multi-factor decision pipeline.

