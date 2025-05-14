import pandas as pd
import os

def compute_alpha_score(row):
    # Invert RSI: lower RSI = better (oversold)
    rsi_score = max(0, (50 - row['RSI']) / 50)

    macd_score = 1 if row['MACD_Bullish'] == "Yes" else 0
    sma_score = 1 if row['Above_SMA50'] == "Yes" else 0

    sentiment = row['Sentiment']
    score = (
        sentiment * 0.4 +
        rsi_score * 0.2 +
        macd_score * 0.2 +
        sma_score * 0.2
    )
    return score

def main():
    input_path = 'data/screening_results.csv'
    output_path = 'data/top_signals.csv'

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input: {input_path}")

    df = pd.read_csv(input_path)

    print("📌 Columns found:", df.columns.tolist())

    df['alpha_score'] = df.apply(compute_alpha_score, axis=1)
    df_sorted = df.sort_values(by='alpha_score', ascending=False)

    df_sorted.to_csv(output_path, index=False)
    print(f"\n✅ Saved ranked signals to: {output_path}")
    print("\n🏆 Top Alpha Scores:")
    print(df_sorted[['Ticker', 'alpha_score']].head())

if __name__ == "__main__":
    main()

