"""
Vanilla LSTM Sequence Network (PyTorch).
Models multi-timestep continuous non-linear temporal dynamics.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Tuple

class PyTorchLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(PyTorchLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, sequence_length, features]
        lstm_out, _ = self.lstm(x)
        # Take the representation from the final timestep
        last_step_out = lstm_out[:, -1, :]
        out = self.fc(last_step_out)
        return out.squeeze(-1)

class LSTMForecaster:
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, lr: float = 1e-3, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PyTorchLSTM(input_dim, hidden_dim, num_layers).to(self.device)
        self.criterion = nn.HuberLoss() # Robust to extreme winter smog outliers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=4)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, epochs: int = 25, batch_size: int = 64):
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

        best_val_loss = float('inf')
        
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
            
            val_loss = 0.0
            if has_val:
                self.model.eval()
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

