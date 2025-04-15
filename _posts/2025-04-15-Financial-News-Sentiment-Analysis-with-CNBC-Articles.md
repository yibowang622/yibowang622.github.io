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

```python
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import time
import random
```

```python
# Load FinBERT model and tokenizer
print("Loading FinBERT model...")
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create a sentiment pipeline
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# User agent to mimic a browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
}

# Function to get article content
def get_article_content(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch {url}: {response.status_code}")
            return None, None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to extract title using different possible selectors
        title = None
        for selector in ['h1', 'h1.ArticleHeader-headline', '.headline']:
            if not title:
                title_element = soup.select_one(selector)
                if title_element:
                    title = title_element.get_text().strip()
        
        # Extract all paragraph text
        paragraphs = [p.get_text().strip() for p in soup.select('p')]
        content = ' '.join(paragraphs)
        
        return title, content
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None, None

# Function to analyze sentiment with chunking
def analyze_sentiment(text):
    if not text:
        return None, None
    
    try:
        # Process text in small chunks that won't exceed token limit
        # Use the tokenizer directly to control chunk size by tokens
        tokens = tokenizer.tokenize(text)
        max_tokens = 450  # Well below the 512 limit
        
        # Split into chunks by tokens
        token_chunks = []
        for i in range(0, len(tokens), max_tokens):
            token_chunks.append(tokens[i:i + max_tokens])
        
        # Convert token chunks back to text
        text_chunks = []
        for chunk in token_chunks:
            chunk_text = tokenizer.convert_tokens_to_string(chunk)
            text_chunks.append(chunk_text)
            
        print(f"  Split into {len(text_chunks)} chunks for analysis")
        
        # Process each chunk separately
        sentiments = []
        for i, chunk in enumerate(text_chunks):
            if chunk.strip():
                try:
                    # Add a 5% margin of safety by truncating very slightly
                    safe_chunk = ' '.join(chunk.split()[:int(max_tokens * 0.95)])
                    chunk_sentiment = finbert(safe_chunk)[0]
                    sentiments.append(chunk_sentiment)
                    print(f"  Analyzed chunk {i+1}/{len(text_chunks)}")
                except Exception as e:
                    print(f"  Error analyzing chunk {i+1}: {e}")
        
        if not sentiments:
            return None, None
            
        # Average the confidence scores for each sentiment class
        sentiment_scores = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        for s in sentiments:
            sentiment_scores[s['label']] += s['score'] / len(sentiments)
        
        # Find the sentiment with highest average score
        final_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
        return final_sentiment[0], final_sentiment[1]
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return None, None

# Fetch CNBC homepage
print("Fetching CNBC homepage...")
response = requests.get('https://www.cnbc.com/', headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# Find article links
links = []
for a in soup.select('a[href]'):
    href = a.get('href')
    if href and "cnbc.com" in href and href.startswith('http') and "/2025/" in href:
        links.append(href)

# Limit to unique links
links = list(set(links))[:5]
print(f"Found {len(links)} article links")

# Process each article
results = []
for i, url in enumerate(links):
    print(f"Processing article {i+1}/{len(links)}: {url}")
    
    title, content = get_article_content(url)
    if title and content:
        print(f"  Title: {title}")
        print(f"  Content length: {len(content)} characters")
        
        sentiment, confidence = analyze_sentiment(content)
        if sentiment:
            print(f"  Sentiment: {sentiment} (Confidence: {confidence:.4f})")
            
            results.append({
                'title': title,
                'sentiment_label': sentiment,
                'confidence': confidence,
                'url': url
            })
    
    # Be nice to the server
    time.sleep(2 + random.random())

# Create DataFrame and display results
df = pd.DataFrame(results)
print("\nScraping Results:")
display(df)

# Save results to JSON
df.to_json('/content/cnbc_results.json', orient='records')
print(f"Results saved to /content/cnbc_results.json")
```

### Results
```python
Scraping Results:
                      title	                          sentiment_label  confidence	url
0	U.S. stock futures slide after S&P 500 posts b...	Positive        0.505471	https://www.cnbc.com/2025/04/14/stock-market-t...
1	Trump administration says it lacks authority t...	Neutral	        0.749853	https://www.cnbc.com/2025/04/14/abrego-garcia-...
2	Sports teams adopt tactile tech for blind and ...	Neutral	        0.854449	https://www.cnbc.com/2025/04/13/sports-teams-t...
3	Adobe takes stake in Synthesia, startup behind...	Neutral	        0.653253	https://www.cnbc.com/2025/04/15/adobe-invests-...
4	Trump, top aides fuel tariff confusion by ques...	Neutral	        0.748785	https://www.cnbc.com/2025/04/13/trump-commerce...
```

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
