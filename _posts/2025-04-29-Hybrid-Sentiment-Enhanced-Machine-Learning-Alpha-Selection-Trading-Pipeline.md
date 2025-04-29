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
To align sentiment with stock trading decisions, we aggregate sentiment scores for each stock over the past 7 days by calculating the average confidence score for positive sentiment.

Only stocks with a 7-day average sentiment score greater than 0.6 are considered for further selection.
