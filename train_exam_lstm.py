import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import Config, ensure_dirs
from models.HybridANNLSTM import HybridANNLSTM


# ==========================================
# 方向惩罚 + Huber 损失
# ==========================================
class DirectionalHuberLoss(nn.Module):
    def __init__(self, alpha=2.0, delta=0.05):
        super().__init__()
        self.alpha = alpha
        self.delta = delta

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        abs_diff = diff.abs()
        huber = torch.where(abs_diff <= self.delta, 0.5 * diff ** 2, self.delta * (abs_diff - 0.5 * self.delta))
        weights = torch.where((y_pred * y_true) < 0, self.alpha, 1.0)
        return torch.mean(weights * huber)


def split_by_date(X, Y, dates, config=Config):
    dates = pd.to_datetime(dates)
    train_mask = dates <= pd.Timestamp(config.train_end)
    val_mask = (dates >= pd.Timestamp(config.val_start)) & (dates <= pd.Timestamp(config.val_end))
    test_mask = (dates >= pd.Timestamp(config.test_start)) & (dates <= pd.Timestamp(config.test_end))
    return (X[train_mask], Y[train_mask], dates[train_mask]), (X[val_mask], Y[val_mask], dates[val_mask]), (X[test_mask], Y[test_mask], dates[test_mask])


def evaluate(model, loader, criterion, device):
    model.eval()
    preds, trues = [], []
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for batch_X, batch_Y in loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            pred = model(batch_X)
            loss = criterion(pred, batch_Y)
            total_loss += loss.item() * batch_X.size(0)
            total += batch_X.size(0)
            preds.append(pred.cpu().numpy())
            trues.append(batch_Y.cpu().numpy())
    pred_all = np.vstack(preds).flatten() if preds else np.array([])
    true_all = np.vstack(trues).flatten() if trues else np.array([])
    return {
        "loss": total_loss / max(total, 1),
        "rmse": float(np.sqrt(np.mean((pred_all - true_all) ** 2))) if len(pred_all) else np.nan,
        "mae": float(np.mean(np.abs(pred_all - true_all))) if len(pred_all) else np.nan,
        "direction_acc": float(np.mean(np.sign(pred_all) == np.sign(true_all))) if len(pred_all) else np.nan,
        "pred": pred_all,
        "true": true_all,
    }


# ==========================================
# 考试版 LSTM 训练
# ==========================================
def train_exam_lstm(X, Y, dates, n_features, config=Config):
    ensure_dirs(config)
    if n_features < 20:
        raise ValueError("特征维度不足 20，不满足题目要求。")

    train_data, val_data, test_data = split_by_date(X, Y, dates, config)
    X_train, Y_train, _ = train_data
    X_val, Y_val, _ = val_data
    X_test, Y_test, test_dates = test_data

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(Y_val)), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(Y_test)), batch_size=config.batch_size, shuffle=False)

    model = HybridANNLSTM(n_features=n_features, lstm_hidden_dim=config.lstm_hidden_dim, ann_hidden_dim=config.ann_hidden_dim, dropout=config.dropout).to(config.device)
    criterion = DirectionalHuberLoss(config.alpha, config.delta)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    best_val_loss = float("inf")
    early_stop_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    print(f"考试版 LSTM 输入特征数: {n_features}")
    print(f"训练集 {len(X_train)}，验证集 {len(X_val)}，测试集 {len(X_test)}")

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(config.device)
            batch_Y = batch_Y.to(config.device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= max(len(train_loader.dataset), 1)

        val_metrics = evaluate(model, val_loader, criterion, config.device)
        scheduler.step(val_metrics["loss"])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["direction_acc"])
        history["lr"].append(optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch:03d} | Train {train_loss:.6f} | Val {val_metrics['loss']:.6f} | Acc {val_metrics['direction_acc']:.2%}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({"epoch": epoch, "n_features": n_features, "model_state_dict": model.state_dict(), "val_loss": best_val_loss}, config.checkpoint_dir / "exam_lstm_best.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if early_stop_counter >= config.early_stop_patience:
            print("早停触发。")
            break

    torch.save({"epoch": epoch, "n_features": n_features, "model_state_dict": model.state_dict()}, config.checkpoint_dir / "exam_lstm_last.pth")
    with open(config.checkpoint_dir / "exam_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    ckpt = torch.load(config.checkpoint_dir / "exam_lstm_best.pth", map_location=config.device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, config.device)
    pred_df = pd.DataFrame({"Date": test_dates, "true": test_metrics["true"], "lstm_pred": test_metrics["pred"]})
    pred_df.to_csv(config.table_dir / "L5_lstm_test_pred.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{k: v for k, v in test_metrics.items() if k not in ["pred", "true"]}]).to_csv(config.table_dir / "L5_lstm_test_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"测试集方向准确率: {test_metrics['direction_acc']:.2%}, RMSE: {test_metrics['rmse']:.6f}, MAE: {test_metrics['mae']:.6f}")
    return model, history, pred_df, test_metrics
