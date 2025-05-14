import pandas as pd
import json
import os
from datetime import datetime
import re


# Enhanced JSON line-by-line loader with more robust error handling
def load_ndjson(path):
    """
    Load NDJSON (newline-delimited JSON) file with robust error handling.
    Returns a DataFrame of successfully parsed records.
    """
    data = []
    errors = 0
    total_lines = 0

    try:
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                total_lines += 1
                line = line.strip()
                if not line:
                    continue

                # Try to clean and parse the JSON
                try:
                    # Remove any trailing commas that might cause issues
                    if line.endswith(','):
                        line = line[:-1]

                    # Handle case where line might be wrapped in array brackets
                    if line.startswith('[') and line.endswith(']'):
                        try:
                            items = json.loads(line)
                            if isinstance(items, list):
                                for item in items:
                                    data.append(item)
                                continue
                        except:
                            pass  # Fall through to regular parsing

                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        data.append(obj)
                except json.JSONDecodeError as e:
                    errors += 1
                    if errors < 50:  # Limit error messages to avoid flooding output
                        print(f"⚠️ Skipped invalid JSON on line {i}: {str(e)[:50]}...")
                    elif errors == 50:
                        print("⚠️ Too many errors, suppressing further error messages...")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return pd.DataFrame()

    print(f"📊 Processed {total_lines} lines with {errors} errors. Successfully parsed {len(data)} records.")
    return pd.DataFrame(data) if data else pd.DataFrame()


# Enhanced stock detection with more comprehensive mapping and partial matching
def detect_stocks(title, description=None):
    """
    Detect stock tickers from news title and description with improved matching.
    """
    if not isinstance(title, str):
        return None

    # Combine title and description if available
    text = title.lower()
    if isinstance(description, str):
        text += " " + description.lower()

    # Expanded stock map with more companies and variations
    stock_map = {
        'apple': 'AAPL', 'iphone': 'AAPL', 'ipad': 'AAPL', 'macbook': 'AAPL', 'tim cook': 'AAPL',
        'tesla': 'TSLA', 'elon musk': 'TSLA', 'spacex': 'TSLA', 'tsla': 'TSLA',
        'microsoft': 'MSFT', 'azure': 'MSFT', 'msft': 'MSFT', 'satya nadella': 'MSFT',
        'amazon': 'AMZN', 'aws': 'AMZN', 'bezos': 'AMZN', 'andy jassy': 'AMZN',
        'meta': 'META', 'facebook': 'META', 'instagram': 'META', 'zuckerberg': 'META',
        'nvidia': 'NVDA', 'nvda': 'NVDA', 'jensen huang': 'NVDA',
        'google': 'GOOGL', 'alphabet': 'GOOGL', 'youtube': 'GOOGL', 'pichai': 'GOOGL',
        'netflix': 'NFLX', 'nflx': 'NFLX',
        'goldman sachs': 'GS', 'morgan stanley': 'MS', 'jpmorgan': 'JPM', 'jp morgan': 'JPM',
        'boeing': 'BA', 'chipotle': 'CMG', 'hsbc': 'HSBC',
        'deutsche bank': 'DB', 'bp': 'BP', 'oracle': 'ORCL',
        'novartis': 'NVS', 'porsche': 'POAHY', 'volvo': 'VLVLY',
        'berkshire': 'BRK', 'buffett': 'BRK',
        'uber': 'UBER', 'domino': 'DPZ',
        'temu': 'PDD', 'nxp': 'NXPI', 'semiconductor': 'SMH',
        'deliveroo': 'ROO.L',
        'barclays': 'BCS'
    }

    matched_stocks = []

    # Check for matches in our map
    for keyword, ticker in stock_map.items():
        if keyword in text:
            matched_stocks.append(ticker)

    # Also look for explicit ticker mentions like $AAPL
    ticker_pattern = r'\$([A-Z]{1,5})'
    explicit_tickers = re.findall(ticker_pattern, text)
    matched_stocks.extend(explicit_tickers)

    # Return unique matches, prioritizing the first match
    return matched_stocks[0] if matched_stocks else None


# Main sentiment aggregation function with enhanced flexibility
def aggregate_sentiment(
        input_path=None,
        output_path=None,
        lookback_days=7,
        confidence_threshold=0.5
):
    """
    Aggregate sentiment data from news articles and identify positive sentiment for stocks.

    Parameters:
    - input_path: Path to the scraped news JSON file
    - output_path: Path to save the sentiment scores CSV
    - lookback_days: Number of days to look back for sentiment analysis
    - confidence_threshold: Minimum confidence score to consider (0.0-1.0)
    """
    # Default paths if not provided
    if input_path is None:
        input_path = os.path.join('data', 'scraped_news.json')

    if output_path is None:
        output_path = os.path.join('data', 'sentiment_scores.csv')

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load data
    print(f"📂 Loading data from {input_path}...")
    df = load_ndjson(input_path)

    if df.empty:
        print("❌ No valid data found in the input file.")
        return

    # Print column names for debugging
    print(f"\n📋 Available columns: {', '.join(df.columns)}")

    # Check if required columns exist
    required_cols = ['title', 'url']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Required column '{col}' not found in the data.")
            return

    # Print sample data
    print("\n📌 Sample data:")
    print(df[['title', 'sentiment_label', 'confidence']].head(3))

    # Extract date from URL or fallback to today
    date_pattern = r'(\d{4}/\d{2}/\d{2})'

    # Try to extract dates from URLs
    print("\n📅 Extracting dates from URLs...")
    df['date_str'] = df['url'].str.extract(date_pattern)[0]

    # Convert to datetime
    df['date'] = pd.to_datetime(df['date_str'], errors='coerce', format='%Y/%m/%d')

    # Fill missing dates with today's date
    missing_dates = df['date'].isna().sum()
    if missing_dates > 0:
        print(f"ℹ️ Filling {missing_dates} missing dates with today's date")
        df['date'] = df['date'].fillna(pd.Timestamp.today())

    # Detect stocks from title
    print("\n🔍 Detecting stocks from news titles...")
    df['stock'] = df['title'].apply(detect_stocks)

    # Count stock mentions
    stock_counts = df['stock'].value_counts()
    print(f"\n📊 Detected {len(stock_counts)} unique stocks")
    if not stock_counts.empty:
        print("Top 5 mentioned stocks:")
        print(stock_counts.head(5))

    # Drop rows without stock associations
    original_count = len(df)
    df = df.dropna(subset=['stock'])
    print(f"\nℹ️ Filtered from {original_count} to {len(df)} items with stock associations")

    if df.empty:
        print("❌ No news items with stock associations found.")
        return

    # Ensure sentiment_label column exists or fallback
    if 'sentiment_label' not in df.columns:
        print("⚠️ 'sentiment_label' column not found. Creating a placeholder.")
        df['sentiment_label'] = 'Unknown'

    # Ensure confidence column exists or fallback
    if 'confidence' not in df.columns:
        print("⚠️ 'confidence' column not found. Creating a placeholder.")
        df['confidence'] = 0.5

    # Filter to positive sentiment with confidence above threshold
    positive_news = df[
        (df['sentiment_label'] == 'Positive') &
        (df['confidence'] >= confidence_threshold)
        ]

    print(f"\n✨ Found {len(positive_news)} positive sentiment news items")

    if positive_news.empty:
        print("⚠️ No positive sentiment news found after filtering.")
        # Look for any positive sentiment regardless of confidence
        any_positive = df[df['sentiment_label'] == 'Positive']
        if not any_positive.empty:
            print(
                f"ℹ️ There are {len(any_positive)} positive items, but none meet the confidence threshold of {confidence_threshold}")
        return

    # Print top positive news
    print("\n📰 Sample positive news:")
    sample_cols = ['title', 'stock', 'confidence', 'date']
    print(positive_news[sample_cols].head(3))

    # Filter by date window
    recent_date = positive_news['date'].max()
    window_start = recent_date - pd.Timedelta(days=lookback_days)
    print(f"\n🕒 Aggregating sentiment between {window_start.date()} and {recent_date.date()}")

    recent_positive_news = positive_news[
        (positive_news['date'] >= window_start) &
        (positive_news['date'] <= recent_date)
        ]

    if recent_positive_news.empty:
        print("⚠️ No recent positive news in the selected date window.")
        return

    # Group and calculate mean confidence
    sentiment_scores = recent_positive_news.groupby('stock')['confidence'].agg(
        avg_sentiment_score=lambda x: round(x.mean(), 3),
        article_count='count'
    ).reset_index()

    # Sort by sentiment score (highest first)
    sentiment_scores = sentiment_scores.sort_values('avg_sentiment_score', ascending=False)

    # Save to CSV
    sentiment_scores.to_csv(output_path, index=False)
    print(f"\n✅ Sentiment scores saved to {output_path}")
    print("\n📊 Final sentiment scores:")
    print(sentiment_scores.to_string(index=False))

    return sentiment_scores


# Run when script is executed directly
if __name__ == "__main__":
    aggregate_sentiment()
