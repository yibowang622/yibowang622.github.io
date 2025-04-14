---
layout: post
title: "FinBERT News Sentiment Analysis for Stock Tickers"
date: 2025-04-15
categories: ["machine learning", "quantitative trading"]
colab_notebook: "https://colab.research.google.com/drive/1fUFtMN3o8G3P6XRxl2UsITsrYuoa568L?usp=sharing"
---
### Introduction
This project demonstrates a simple yet effective way to analyze sentiment in financial news headlines using FinBERT, a BERT model fine-tuned for financial text. By combining the NewsAPI for data collection with the power of FinBERT for sentiment analysis, we can gauge market sentiment about specific stocks in near real-time.

### The Problem
Investors and traders are constantly bombarded with news that could impact stock prices. Manually tracking and analyzing this information is time-consuming and subject to human bias. An automated sentiment analysis tool can help by:

1.Aggregating relevant news from multiple sources<br>
2.Providing an objective sentiment classification (positive, negative, or neutral)<br>
3.Quantifying the confidence of these classifications<br>

### Implementation
### Data Collection
We use NewsAPI to gather recent headlines related to a specific ticker symbol:

```
url = f'https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&apiKey={api_key}'
```python

One challenge we encountered was ensuring headlines were in English. Some non-English content was being returned despite the language=en parameter. We solved this by adding language detection:
pythonfrom langdetect import detect

```
english_headlines = []
for article in articles:
    title = article['title']
    try:
        if detect(title) == 'en':
            english_headlines.append(title)
    except:
        continue
```python

### Sentiment Analysis with FinBERT
For the sentiment classification, we use the yiyanghkust/finbert-tone model from Hugging Face:

```
pythonmodel_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
```python

FinBERT classifies text into three categories:

*Positive: Indicates optimistic sentiment
*Negative: Indicates pessimistic sentiment
*Neutral: Indicates factual or balanced reporting

Each classification comes with a confidence score between 0 and 1.

### Results
In our initial test with Apple (AAPL), we found:

*All headlines were classified as "Neutral" with very high confidence scores (average 0.9775)
*The headlines were predominantly about stock transactions rather than substantive news about Apple's performance or products

Sample output:
```
📰 FinBERT Sentiment Analysis on 10 News Headlines for AAPL:

                                                                                                 Headline Sentiment_Label  Confidence
                                              Quilter Plc Sells 20,975 Shares of Apple Inc. (NASDAQ:AAPL)         Neutral    0.999971
                              Fonville Wealth Management LLC Sells 584 Shares of Apple Inc. (NASDAQ:AAPL)         Neutral    0.999983
             Apple Inc. (NASDAQ:AAPL) Stock Holdings Lessened by Ferguson Wellman Capital Management Inc.         Neutral    0.792049
                           Apple Inc. (NASDAQ:AAPL) Holdings Decreased by Silicon Valley Capital Partners         Neutral    0.983713
                          8,173 Shares in Apple Inc. (NASDAQ:AAPL) Bought by Wingate Wealth Advisors Inc.         Neutral    0.999999

--- Sentiment Summary ---
Neutral: 10 headlines (100.0%)
Average confidence: 0.9775
```python

### Observations and Limitations

**1.News Selection: The headlines returned were primarily about institutional buying/selling of shares rather than substantive news that might actually affect stock prices.
**2.Sentiment Distribution: All headlines were classified as neutral, which is expected for transaction reports but limits the insights we can gain from the analysis.
**3.Language Filtering: The default API filters weren't sufficient, requiring additional language detection.
**4.Limited Context: Analyzing only headlines misses the context and details in the full articles that might contain more nuanced sentiment.

### Future Improvements

**1.Enhanced News Query: Modify the NewsAPI query to target more substantial news about company performance, products, or market conditions.
**2.Historical Analysis: Add functionality to track sentiment over time and correlate with stock price movements.
**3.Content Expansion: Analyze full article content rather than just headlines for more comprehensive sentiment assessment.
**4.Multi-ticker Comparison: Compare sentiment across multiple related stocks or an entire sector.
**5.Trading Signals: Develop a system to generate potential trading signals based on significant sentiment shifts.
**6.Dashboard Visualization: Create a dashboard with sentiment trends, key statistics, and real-time updates.

### Conclusion
Even this simple implementation demonstrates the potential of combining API-based news collection with specialized NLP models for financial analysis. While our initial results were limited by the nature of the headlines we received, the framework provides a foundation for more sophisticated sentiment analysis applications.
The high confidence scores from FinBERT (average 0.9775) suggest the model is well-suited for financial text classification, but a more diverse set of headlines would better showcase its ability to distinguish between positive, negative, and neutral sentiment.
By refining our approach to news collection and expanding the analysis to full articles and historical data, this tool could become a valuable component in a quantitative investor's toolkit.

### Resources

FinBERT on Hugging Face
NewsAPI Documentation
