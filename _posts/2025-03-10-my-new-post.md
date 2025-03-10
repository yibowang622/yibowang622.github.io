---
layout: default
title: "My New Post"
date: 2025-03-15
permalink: /machine%20learning/quantitative%20trading/2025/03/15/my-new-post.html
---



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

# ✅ Step 1: Download stock data
stock_symbol = "AAPL"
df = yf.download(stock_symbol, start="2015-01-01", end="2024-01-01")

# Keep only the 'Close' price
data = df[['Close']].values  # Convert to NumPy array

# ✅ Step 2: Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))  # Scale between 0 and 1
data_scaled = scaler.fit_transform(data)

# Split into train and test sets (80% train, 20% test)
train_size = int(len(data_scaled) * 0.8)
train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]


# ✅ Step 3: Function to create sequences
def create_sequences(dataset, seq_length=60):
    X, y = [], []
    for i in range(len(dataset) - seq_length):
        X.append(dataset[i:i + seq_length])
        y.append(dataset[i + seq_length])
    return np.array(X), np.array(y)


# Create train and test sequences
seq_length = 60  # Use last 60 days for prediction
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# ✅ Step 4: Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ✅ Fix Shape Issues
y_train = y_train.squeeze(-1)  # Ensure y_train shape is (num_samples, 1)
y_test = y_test.squeeze(-1)

# ✅ Fix X_train Shape: Ensure batch-first format (batch_size, seq_length, features)
X_train = X_train.permute(0, 2, 1)  # (num_samples, features, seq_length)
X_test = X_test.permute(0, 2, 1)

# ✅ Step 5: Define DataLoader to train in batches
batch_size = 16
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ✅ Step 6: Define Transformer Model
class StockPriceTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(StockPriceTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True  # ✅ Fix: Set batch_first=True
        )
        self.output_projection = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        src = src.permute(0, 2, 1)  # ✅ Fix: Ensure shape (batch_size, features, seq_length)
        src = self.input_projection(src)  # Project input features to model dimensions
        output = self.transformer(src, src)
        output = self.output_projection(output[:, -1, :])  # Use last time step for prediction
        return output


# ✅ Step 7: Initialize Transformer model
input_dim = 1
model_dim = 32
num_heads = 2
num_layers = 1
output_dim = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StockPriceTransformer(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)

# ✅ Step 8: Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Step 9: Training loop
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:  # ✅ Train in mini-batches
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # Move to GPU if available
        optimizer.zero_grad()
        output = model(batch_X)
        batch_y = batch_y.view_as(output)  # ✅ Fix Shape Mismatch
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

# ✅ Step 10: Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test.to(device)).cpu().numpy()  # Move back to CPU

# ✅ Fix Shape Mismatch Before Inverse Scaling
y_test_actual = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

# ✅ Step 11: Plot Results
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label="Actual Prices", color="blue", linestyle="-")
plt.plot(predictions, label="Transformer Predicted Prices", color="red", linestyle="--")
plt.legend()
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.title(f"Transformer Prediction for {stock_symbol}")
plt.grid(True)  # Add grid for better visualization
plt.show()

# ✅ Step 12: Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test_actual, predictions)
print(f"Transformer Model Mean Squared Error: {mse:.4f}")
