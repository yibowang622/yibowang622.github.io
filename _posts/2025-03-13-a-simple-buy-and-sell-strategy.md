---
layout: post
title: "A simple buy and sell strategy of transformer performance on simulation data "
date: 2025-03-13
categories: ["machine learning", "quantitative trading"]
images:
 - "/assets/images/portfoliogrowth.jpg"
 - "/assets/images/drawndown.jpg"
 - "/assets/images/actual and predicted price.jpg"
colab_notebook: "https://colab.research.google.com/drive/1-T4U5QOx9p6SAVT0hZij7uyzBJvDAvJM?usp=sharing"
---
### 🚀 Overview
In this post, we explore how transformer-based models can predict simulated stock price movements using historical data.

### 📊 Model Setup
To evaluate the trading performance of a transformer-based model, we simulated 50 days of stock price data with a gradual uptrend and random noise. The model generates buy/sell signals based on predicted price movements, applying the following strategy:

✅ BUY → When the predicted price is higher than the current price by a set threshold.<br>
✅ SELL → When the predicted price is lower than the current price by a set threshold.<br>
✅ HOLD → If the difference is within the threshold range.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

# Step 1: Create Sample Data (Replace this with real predictions)
```python
data = {
    "date": pd.date_range(start="2025-01-01", periods=50, freq="D"),
    "current_price": np.linspace(100, 150, 50) + np.random.randn(50) * 2,  # Simulated price
}
df = pd.DataFrame(data)
```

# Simulated Transformer predictions (Add small noise)

```python
df["predicted_price"] = df["current_price"] * (1 + np.random.randn(50) * 0.02)
```

# Step 2: Generate Trading Signals

```python
def generate_trading_signals(df, threshold=0.01):
    """
    Generates Buy, Sell, or Hold signals based on Transformer predictions.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'current_price' and 'predicted_price'.
    threshold (float): Percentage threshold for triggering Buy/Sell (default: 1%).

    Returns:
    pd.DataFrame: Updated DataFrame with 'signal' column.
    """
    df = df.copy()
    df["pct_change"] = (df["predicted_price"] - df["current_price"]) / df["current_price"]
    df["signal"] = "HOLD"
    df.loc[df["pct_change"] > threshold, "signal"] = "BUY"
    df.loc[df["pct_change"] < -threshold, "signal"] = "SELL"
    return df

df = generate_trading_signals(df)
```

# Step 3: Backtest Strategy

```python
def backtest_strategy(df, initial_capital=10000, position_size=0.1):
    """
    Backtests a trading strategy using Buy/Sell signals.

    Parameters:
    df (pd.DataFrame): DataFrame with 'date', 'current_price', and 'signal'.
    initial_capital (float): Starting capital ($10,000 default).
    position_size (float): Fraction of capital to invest per trade.

    Returns:
    pd.DataFrame: Backtest results including portfolio value.
    """
    df = df.copy()
    capital = initial_capital
    position = 0  # Number of shares held
    portfolio_values = []

    for i in range(len(df)):
        price = df.loc[i, "current_price"]
        signal = df.loc[i, "signal"]

        if signal == "BUY":
            invest_amount = capital * position_size
            num_shares = invest_amount / price
            position += num_shares
            capital -= invest_amount

        elif signal == "SELL" and position > 0:
            capital += position * price
            position = 0

        portfolio_value = capital + (position * price)
        portfolio_values.append(portfolio_value)

    df["portfolio_value"] = portfolio_values
    df["returns"] = df["portfolio_value"].pct_change().fillna(0)

    # Performance Metrics
    total_return = (df["portfolio_value"].iloc[-1] / initial_capital) - 1
    sharpe_ratio = np.mean(df["returns"]) / np.std(df["returns"]) * np.sqrt(252)  # Annualized
    max_drawdown = np.min(df["portfolio_value"]) / np.max(df["portfolio_value"]) - 1

    print(f"📊 Final Portfolio Value: ${df['portfolio_value'].iloc[-1]:.2f}")
    print(f"📈 Total Return: {total_return * 100:.2f}%")
    print(f"⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"📉 Max Drawdown: {max_drawdown * 100:.2f}%")

    return df

df = backtest_strategy(df)
```

# Step 4: Plot Portfolio Value

```python
plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["portfolio_value"], label="Portfolio Value", color="blue")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.title("Portfolio Growth Over Time")
plt.legend()
plt.grid()
plt.show()
```

# Step 5: Drawdown Analysis

```python
df["peak"] = df["portfolio_value"].cummax()
df["drawdown"] = (df["portfolio_value"] - df["peak"]) / df["peak"]

plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["drawdown"], label="Drawdown", color="red")
plt.xlabel("Date")
plt.ylabel("Drawdown (%)")
plt.title("Drawdown Over Time")
plt.legend()
plt.grid()
plt.show()
```

# Plot actual vs predicted prices

```python
plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["current_price"], label="Actual Price", color="blue", linewidth=2)
plt.plot(df["date"], df["predicted_price"], label="Predicted Price", color="orange", linestyle="dashed", linewidth=2)
plt.xlabel("Date")
plt.ylabel("Stock Price ($)")
plt.title("Actual vs. Predicted Price")
plt.legend()
plt.grid()
plt.show()
```
