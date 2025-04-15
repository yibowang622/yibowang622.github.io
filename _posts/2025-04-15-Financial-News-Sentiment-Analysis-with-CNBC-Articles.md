---
layout: post
title: "Financial News Sentiment Analysis with CNBC Articles"
date: 2025-04-15
categories: ["machine learning", "quantitative trading"]
colab_notebook: "https://colab.research.google.com/drive/1D42gO8AoCPPcdY56My-kk5NNuD38S23q?usp=sharing"
---
### Introduction
In this post, I'll explore how to analyze the sentiment of financial news articles from CNBC using Python, web scraping, and a specialized financial language model. Financial sentiment analysis provides valuable insights for investors, analysts, and researchers who want to understand the emotional tone behind market news.
The approach demonstrated here uses a combination of web scraping techniques with the FinBERT model, a financial sentiment analysis tool trained specifically for financial text. Unlike a traditional Python environment setup, this implementation runs entirely in Google Colab, which presents unique challenges and solutions that I'll explain throughout.

### Analysis Approach
### Data Collection
I scraped CNBC's website using a hybrid approach that combines elements of both Scrapy and Beautiful Soup. While a traditional Python environment would typically use Scrapy with a proper project structure, command-line interface, and persistent storage, Google Colab required adaptations to work within its notebook environment.
The main challenges faced in the Colab environment included:

**1. Reactor issues:** Scrapy's underlying Twisted framework can only be started once per Colab session<br>
**2. Non-persistent file system:** Files are lost between runtime restarts<br>
**3. Integration limitations:** Combining interactive notebook cells with asynchronous web scraping frameworks<br>

To overcome these limitations, I created a solution that:

* Uses BeautifulSoup for HTML parsing and requests for HTTP calls<br>
* Implements token-aware chunking to handle long articles<br>
* Processes articles sequentially with appropriate delays<br>
* Provides real-time feedback within the notebook<br>

### Sentiment Analysis
For the sentiment analysis, I used the FinBERT model from HuggingFace, which is specifically trained on financial texts. The model classifies text into three categories:

* Positive
* Negative
* Neutral

A key technical challenge was handling the token limit of the FinBERT model (512 tokens). To solve this, I implemented a chunking approach that:

1. Tokenizes the full article text<br>
2. Splits it into chunks of 450 tokens (safely below the limit)<br>
3. Analyzes each chunk separately<br>
4. Aggregates the sentiment scores across all chunks<br>

### Results
The analysis of five recent CNBC articles revealed:

**1. Stock market update:** Positive sentiment (50.5% confidence)<br>
**2. Abrego Garcia/El Salvador article:** Neutral sentiment (75.0% confidence)<br>
**3. Sports tech for blind fans:** Neutral sentiment (85.4% confidence)<br>
**4. Adobe/Synthesia AI video platform:** Neutral sentiment (65.3% confidence)<br>
**5. Trump tariff confusion article:** Neutral sentiment (74.9% confidence)<br>

The predominance of neutral sentiment (4 out of 5 articles) suggests that CNBC maintains a relatively balanced tone in their financial reporting. The stock market article showing positive sentiment aligns with the recent gains in the S&P 500 mentioned in the title.

### Conclusion
This project demonstrates how to combine web scraping and AI-powered sentiment analysis to gain insights from financial news, even within the constraints of a notebook environment. The approach shown here differs from a traditional Python environment setup in several key ways:

* It uses a more interactive, cell-based workflow instead of modular script files
* It handles environment-specific limitations like reactor constraints
* It provides real-time visual feedback during the scraping process
* It integrates data collection, processing, and visualization in a single notebook

For those looking to implement this in a traditional Python environment, I recommend creating a proper Scrapy project structure with separate modules for the spider, sentiment analysis, and data processing. This would improve maintainability and allow for more robust error handling and scheduling.
The combination of web scraping and sentiment analysis opens up numerous possibilities for financial market research, trend analysis, and even algorithmic trading strategies based on news sentiment signals.
