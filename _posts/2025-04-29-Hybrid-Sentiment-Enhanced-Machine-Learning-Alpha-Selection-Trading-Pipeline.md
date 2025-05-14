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
