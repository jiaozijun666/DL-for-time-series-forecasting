# === models/lstm_model.py ===
import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error

class SeqLSTM(nn.Module):
    def __init__(self, n_features, hidden, n_layers, horizon, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, horizon)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).unsqueeze(-1)

def run_lstm_forecast(X_train, y_train, X_test, y_test, params):
    DEVICE = params.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model = SeqLSTM(
        n_features=X_train.shape[2],
        hidden=params['HIDDEN_SIZE'],
        n_layers=params['N_LAYERS'],
        horizon=y_train.shape[1],
        dropout=params['DROPOUT']
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['LEARNING_RATE'])
    criterion = nn.MSELoss()

    best_val, wait = float("inf"), 0
    val_losses = []
    for epoch in range(1, params['EPOCHS'] + 1):
        model.train()
        xb = torch.from_numpy(X_train).to(DEVICE)
        yb = torch.from_numpy(y_train).unsqueeze(-1).to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(X_test).to(DEVICE)
            yv = torch.from_numpy(y_test).unsqueeze(-1).to(DEVICE)
            val_loss = criterion(model(xv), yv).item()
        val_losses.append(val_loss)
        print(f"Epoch {epoch:02d} | valMSE={val_loss:.4f}")

        if val_loss < best_val:
            best_val, wait = val_loss, 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            wait += 1
            if wait >= params['PATIENCE']:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test).to(DEVICE)).cpu().numpy().squeeze(-1)
    y_true = y_test
    mae = mean_absolute_error(y_true, preds)
    rmse = math.sqrt(mean_squared_error(y_true, preds))
    return y_true, preds, val_losses, mae, rmse
