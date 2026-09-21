"""
Hybrid 1D-CNN + BiLSTM Neural Architecture with Temporal Attention.
Combines:
1. 1D Convolutional filter bank to extract localized multi-pollutant emission bursts.
2. Bidirectional LSTM to model forward accumulation and backward contextual atmospheric state.
3. Self-Attention context pooling layer for dynamic temporal weighting.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Tuple

class TemporalAttention(nn.Module):
    """Computes learned attention weights across the temporal dimension."""
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        # x shape: [batch, seq_len, hidden_dim]
        scores = self.attn(x) # [batch, seq_len, 1]
        weights = F.softmax(scores, dim=1) # [batch, seq_len, 1]
        context = torch.sum(x * weights, dim=1) # [batch, hidden_dim]
        return context, weights

class PyTorchCNNBiLSTM(nn.Module):
    def __init__(self, input_dim: int, conv_filters: int = 64, kernel_size: int = 3, lstm_hidden: int = 64, dropout: float = 0.25):
        super(PyTorchCNNBiLSTM, self).__init__()
        
        # 1D-CNN Feature Extractor (Input: [batch, features, seq_len])
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.conv_act = nn.Mish()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        
        # Bidirectional LSTM Layer
        # BiLSTM input features: conv_filters, output features: lstm_hidden * 2
        self.bilstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Self-Attention Layer over 2 * lstm_hidden
        self.attention = TemporalAttention(hidden_dim=lstm_hidden * 2)
        
        # Dense Regression Projection Head
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        # Transpose for Conv1d: [batch, features, seq_len]
        x_trans = x.permute(0, 2, 1)
        conv_out = self.conv_act(self.conv1d(x_trans))
        conv_pooled = self.pool(conv_out)
        
        # Transpose back for LSTM: [batch, new_seq_len, conv_filters]
        lstm_in = conv_pooled.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(lstm_in)
        
        # Dynamic temporal attention pooling
        context, _ = self.attention(lstm_out)
        
        # Output prediction
        out = self.regressor(context)
        return out.squeeze(-1)

class CNNBiLSTMHybrid:
    def __init__(self, input_dim: int, conv_filters: int = 64, lstm_hidden: int = 64, lr: float = 1e-3, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PyTorchCNNBiLSTM(
            input_dim=input_dim,
            conv_filters=conv_filters,
            lstm_hidden=lstm_hidden
        ).to(self.device)
        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 30,
        batch_size: int = 64
    ):
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        has_val = X_val is not None and y_val is not None
        if has_val:
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(bx)
                loss = self.criterion(preds, by)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item() * len(bx)
                
            train_loss /= len(train_dataset)
            
            if has_val:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(self.device), by.to(self.device)
                        preds = self.model(bx)
                        loss = self.criterion(preds, by)
                        val_loss += loss.item() * len(bx)
                val_loss /= len(val_dataset)
                self.scheduler.step(val_loss)
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device)
                preds = self.model(bx)
                all_preds.append(preds.cpu().numpy())
        predictions = np.concatenate(all_preds, axis=0)
        return np.clip(predictions, 0.0, None)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath: str):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        return self

