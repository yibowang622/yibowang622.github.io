---
layout: post
title: "A simple buy and sell strategy of transformer performance on AAPL data "
date: 2025-03-28
categories: ["machine learning", "quantitative trading"]
images:
 - "/assets/images/3-28-portfoliogrowth.png"
 - "/assets/images/3-28-drawdown.png"
 - "/assets/images/3-28-actual-predicted.png"
colab_notebook: "https://colab.research.google.com/drive/1qVk9dfC2CtlZVn33QDadCvVw7BSzPPgJ?usp=sharing"
---
### 🚀 Overview
In this project, we developed a transformer-based model to predict Apple (AAPL) stock price movements using 9 years of historical data from 2015 to 2024.

### 📊 Model Setup
We implemented a transformer encoder architecture that processes 60-day sequences of stock prices to predict the next day's closing price. The model features multi-head attention mechanisms to capture complex temporal patterns in price movements.

### 📈 Trading Strategy
Our backtesting framework evaluates the model's predictive performance by simulating trades based on the following rules:

✅ BUY → When predicted price exceeds current price by more than 1%.<br>
✅ SELL → When predicted price falls below current price by more than 1%.<br>
✅ HOLD → When price difference is within the 1% threshold.

```python
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
```
# ✅ Step 1: Download stock data
```python
stock_symbol = "AAPL"
df = yf.download(stock_symbol, start="2015-01-01", end="2024-01-01")
```

# ✅ Fix DataFrame structure
```python
df = df.reset_index()  # Ensure 'Date' is a column
df = df[['Date', 'Close']].rename(columns={'Date': 'date', 'Close': 'current_price'})
```

# ✅ Step 2: Normalize the data
```python
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df[['current_price']].values)
```

# Split into train and test sets (80% train, 20% test)
```python
train_size = int(len(data_scaled) * 0.8)
train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]
```

# ✅ Step 3: Function to create sequences
```python
def create_sequences(dataset, seq_length=60):
    X, y = [], []
    for i in range(len(dataset) - seq_length):
        X.append(dataset[i:i + seq_length])
        y.append(dataset[i + seq_length])
    return np.array(X), np.array(y)
```

# Create train and test sequences
```python
seq_length = 60
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)
```

# ✅ Step 4: Convert to PyTorch tensors
```python
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
y_train = torch.tensor(y_train, dtype=torch.float32).squeeze(-1)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).squeeze(-1)
```

# ✅ Step 5: DataLoader
```python
batch_size = 16
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
```

# ✅ Step 6: Define Transformer Model
```python
class StockPriceTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(StockPriceTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)  # Project input features
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(model_dim, output_dim)  # Predict output price

    def forward(self, src):
        src = src.permute(0, 2, 1)  # ✅ Fix: Swap dimensions to (batch_size, seq_length, input_dim)
        src = self.input_projection(src)  # ✅ Apply projection correctly (batch_size, seq_length, model_dim)
        output = self.transformer(src)  # Pass through transformer encoder
        output = self.output_projection(output[:, -1, :])  # Use last timestep output
        return output
```

# ✅ Step 7: Initialize Transformer model
```python
input_dim, model_dim, num_heads, num_layers, output_dim = 1, 32, 2, 1, 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StockPriceTransformer(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)
```

# ✅ Step 8: Define Loss and Optimizer
```python
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
```

# ✅ Step 9: Training loop
```python
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y.view_as(output))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")
```

# ✅ Step 10: Model Prediction
```python
with torch.no_grad():
    predictions = model(X_test.to(device)).cpu().numpy()
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
```

# ✅ Fix: Create a new DataFrame for backtesting with only the test data portion
# Calculate the exact indices for test data in the original dataframe
```python
test_start_idx = train_size + seq_length
test_end_idx = test_start_idx + len(predictions)
```

# Create a new DataFrame with only the relevant portion for backtesting
```python
backtest_df = df[test_start_idx:test_end_idx].copy().reset_index(drop=True)
backtest_df["predicted_price"] = predictions
```

# ✅ Completely standalone function that doesn't try to create or return a DataFrame
```python
def run_standalone_backtest(backtest_df, initial_capital=10000, position_size=0.1, threshold=0.01):
    """
    Runs a complete standalone backtest without creating any new DataFrames.
    Just prints the results and returns the data for plotting.
    """
    # Get the data we need as plain Python lists - properly flatten everything
    # Convert dates - ensure they're datetime objects
    dates = []
    for date in backtest_df['date'].values:
        if isinstance(date, (int, float)):  # If it's a timestamp
            dates.append(pd.to_datetime(date))
        else:
            dates.append(date)

    # Convert prices and ensure they're scalar values
    prices = []
    for price in backtest_df['current_price'].values:
        if np.isscalar(price):
            prices.append(float(price))
        elif hasattr(price, '__len__') and len(price) > 0:
            prices.append(float(price[0]))
        else:
            # Something went wrong, use a default
            print(f"Warning: Unexpected price format: {price}")
            prices.append(100.0)

    # Handle predicted_prices correctly - flatten any nested lists
    predicted_prices = []
    for pred in backtest_df['predicted_price'].values:
        # Check if pred is already a scalar
        if np.isscalar(pred):
            predicted_prices.append(float(pred))
        # Check if pred is a list or array with one element
        elif hasattr(pred, '__len__') and len(pred) > 0:
            predicted_prices.append(float(pred[0]))
        else:
            # Just add as is and hope for the best
            predicted_prices.append(float(pred))

    print(f"Running backtest on {len(dates)} data points from {dates[0]} to {dates[-1]}")

    # Generate signals first
    signals = []
    for i in range(len(prices)):
        current = prices[i]
        predicted = predicted_prices[i]
        pct_change = (predicted - current) / current

        if pct_change > threshold:
            signals.append("BUY")
        elif pct_change < -threshold:
            signals.append("SELL")
        else:
            signals.append("HOLD")

    # Now run the backtest
    capital = initial_capital
    position = 0
    portfolio_values = []

    for i in range(len(signals)):
        price = prices[i]
        signal = signals[i]

        # Process trades
        if signal == "BUY":
            invest_amount = capital * position_size
            position += invest_amount / price
            capital -= invest_amount
        elif signal == "SELL" and position > 0:
            capital += position * price
            position = 0

        # Calculate portfolio value
        portfolio_value = capital + (position * price)
        portfolio_values.append(portfolio_value)

    # Calculate performance metrics
    final_value = portfolio_values[-1]
    total_return = (final_value / initial_capital - 1) * 100

    # Calculate peaks and drawdowns
    peaks = []
    drawdowns = []
    current_peak = portfolio_values[0]

    for value in portfolio_values:
        current_peak = max(current_peak, value)
        peaks.append(current_peak)
        drawdown = (value - current_peak) / current_peak
        drawdowns.append(drawdown)

    max_drawdown = min(drawdowns)

    # Calculate returns
    returns = [0]  # First position has no return
    for i in range(1, len(portfolio_values)):
        prev_value = portfolio_values[i-1]
        curr_value = portfolio_values[i]
        returns.append((curr_value / prev_value) - 1 if prev_value > 0 else 0)

    # Calculate Sharpe ratio
    risk_free_rate = 0.02 / 252
    returns_mean = sum(returns) / len(returns)
    returns_std = np.std(returns) or 1  # Avoid division by zero
    sharpe_ratio = (returns_mean - risk_free_rate) / returns_std * np.sqrt(252)

    # Print results
    print(f"\n===== BACKTEST RESULTS =====")
    print(f"📊 Final Portfolio Value: ${final_value:.2f}")
    print(f"📈 Total Return: {total_return:.2f}%")
    print(f"⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"📉 Max Drawdown: {max_drawdown * 100:.2f}%")

    # Return the data needed for plotting, without creating a DataFrame
    return {
        'dates': dates,
        'prices': prices,
        'predicted_prices': predicted_prices,
        'signals': signals,
        'portfolio_values': portfolio_values,
        'drawdowns': drawdowns
    }
```

# ✅ Run the standalone backtest
```python
backtest_data = run_standalone_backtest(backtest_df)
```

# ✅ Step 11: Plot Portfolio Value
```python
plt.figure(figsize=(12, 5))
plt.plot(backtest_data['dates'], backtest_data['portfolio_values'], label="Portfolio Value", color="blue")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.title("Portfolio Growth Over Time")
plt.legend()
plt.grid()
plt.show()
```

# ✅ Step 12: Drawdown Analysis
```python
plt.figure(figsize=(12, 5))
plt.plot(backtest_data['dates'], backtest_data['drawdowns'], label="Drawdown", color="red")
plt.xlabel("Date")
plt.ylabel("Drawdown (%)")
plt.title("Drawdown Over Time")
plt.legend()
plt.grid()
plt.show()
```

# ✅ Plot actual vs predicted prices
```python
plt.figure(figsize=(12, 5))
plt.plot(backtest_data['dates'], backtest_data['prices'], label="Actual Price", color="blue", linewidth=2)
plt.plot(backtest_data['dates'], backtest_data['predicted_prices'], label="Predicted Price", color="orange", linestyle="dashed", linewidth=2)
plt.xlabel("Date")
plt.ylabel("Stock Price ($)")
plt.title("Actual vs. Predicted Price")
plt.legend()
plt.grid()
plt.show()
```
# ✅ Training progress
```python
Epoch [5/50], Loss: 0.0012
Epoch [10/50], Loss: 0.0010
Epoch [15/50], Loss: 0.0006
Epoch [20/50], Loss: 0.0005
Epoch [25/50], Loss: 0.0005
Epoch [30/50], Loss: 0.0006
Epoch [35/50], Loss: 0.0004
Epoch [40/50], Loss: 0.0004
Epoch [45/50], Loss: 0.0004
Epoch [50/50], Loss: 0.0003
Running backtest on 393 data points from 2022-06-08T00:00:00.000000000 to 2023-12-29T00:00:00.000000000
```

# ✅ Backtest results
```python
===== BACKTEST RESULTS =====
📊 Final Portfolio Value: $10166.47
📈 Total Return: 1.66%
⚡ Sharpe Ratio: -0.49
📉 Max Drawdown: -1.34%
```
### Backtest Performance
Running backtest on 393 data points from 2022-06-08 to 2023-12-29

### 📊 Key Metrics

📈 Final Portfolio Value: $10,166.47<br>
💰 Total Return: 1.66%<br>
⚡ Sharpe Ratio: -0.49<br>
📉 Max Drawdown: -1.34%<br>

### Analysis
Our transformer model showed promising learning capacity, with training loss decreasing 75% over 50 epochs. However, the moderate 1.66% return in backtesting reveals the challenge of translating pattern recognition into profitable trading.
The negative Sharpe ratio suggests that while our strategy preserved capital (low drawdown), it didn't generate returns commensurate with the risk taken. This highlights the difficulty of consistently predicting market movements even with sophisticated deep learning architectures.
These results demonstrate both the potential and limitations of transformer models in financial forecasting. Future work should explore incorporating additional market signals and optimizing trading thresholds to enhance performance.

### Future Directions
To enhance this approach, we could:

Incorporate additional features beyond price data (trading volume, sentiment analysis, etc.)
Experiment with different sequence lengths and prediction horizons
Implement more sophisticated position sizing based on prediction confidence
Test different thresholds for generating trading signals
Explore ensemble methods combining transformer predictions with other model architectures

This project demonstrates both the potential and limitations of applying transformer architectures to financial time series forecasting. While perfect prediction remains elusive, transformer-based approaches show promise as components in a broader trading strategy framework.
