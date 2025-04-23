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

def run_lstm_forecast(loaders, n_features, params, return_train_loss=False):
    train_loader, val_loader, test_loader = loaders
    DEVICE = params.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model = SeqLSTM(n_features, params['HIDDEN_SIZE'], params['N_LAYERS'], params['HORIZON'], params['DROPOUT']).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['LEARNING_RATE'])
    criterion = nn.MSELoss()

    best_val, wait = float("inf"), 0
    train_losses, val_losses = [], []

    for epoch in range(1, params['EPOCHS'] + 1):
        model.train(); train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval(); val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d} | trainMSE={train_loss:.4f} | valMSE={val_loss:.4f}")

        if val_loss < best_val:
            best_val, wait = val_loss, 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            wait += 1
            if wait >= params['PATIENCE']:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load('best_model.pth'))
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
            trues.append(yb.numpy())
    preds = np.concatenate(preds).squeeze(-1)
    trues = np.concatenate(trues).squeeze(-1)

    mae = mean_absolute_error(trues, preds)
    rmse = math.sqrt(mean_squared_error(trues, preds))

    if return_train_loss:
        return trues, preds, train_losses, val_losses, mae, rmse
    else:
        return trues, preds, val_losses, mae, rmse
