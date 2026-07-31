import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data import process_input, process_targets, process_test, load_cmapss, preprocess
from models.classical import Decoder, Encoder, Seq2Seq
from models.quantum import QEncoder

COLUMNS_TO_DROP = [0, 1, 2, 3, 4, 5, 9, 10, 14, 20, 22, 23]

def build_train_set(train_data, window_length, shift, max_rul):
    all_X, all_y = [], []
    for engine_id in np.sort(train_data[0].unique()):
        engine = train_data[train_data[0] == engine_id].drop(columns=[0]).values
        if len(engine) < window_length:
            raise AssertionError(f"train engine {engine_id}: fewer rows than window_length={window_length}")
        targets = process_targets(engine.shape[0], max_rul=max_rul)
        X, y = process_input(engine, targets, window_length, shift)
        all_X.append(X)
        all_y.append(y)
    return np.concatenate(all_X).astype(np.float32), np.concatenate(all_y).astype(np.float32)


def build_test_set(test_data, window_length, shift, num_test_windows):
    all_X, windows_per_engine = [], []
    for engine_id in np.sort(test_data[0].unique()):
        engine = test_data[test_data[0] == engine_id].drop(columns=[0]).values
        if len(engine) < window_length:
            raise AssertionError(f"test engine {engine_id}: fewer rows than window_length={window_length}")
        X, n = process_test(engine, window_length, shift, num_test_windows)
        all_X.append(X)
        windows_per_engine.append(n)
    return np.concatenate(all_X).astype(np.float32), windows_per_engine


def compute_s_score(rul_true, rul_pred):
    diff = rul_pred - rul_true
    return np.sum(np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1))


def train_one_epoch(net, loader, optimizer, loss_fn, device):
    net.train()
    total_loss, n_batches = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        dec_input = X[:, -1:, :]
        optimizer.zero_grad()
        y_hat, _ = net(X, dec_input)
        loss = loss_fn(y_hat.squeeze(-1), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def predict(net, X, device, batch_size=256):
    net.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[i:i + batch_size]).to(device)
        dec_input = batch[:, -1:, :]
        y_hat, _ = net(batch, dec_input)
        preds.append(y_hat.squeeze(-1).cpu().numpy())
    return np.concatenate(preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data", help="dir containing train_/test_/RUL_ txt files")
    p.add_argument("--subset", default="FD001")
    p.add_argument("--window-length", type=int, default=30)
    p.add_argument("--shift", type=int, default=1)
    p.add_argument("--max-rul", type=int, default=125)
    p.add_argument("--num-test-windows", type=int, default=5)
    p.add_argument("--num-hidden", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--attention-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", default="./saved_weights/seq2seq_rul.pt")
    p.add_argument("--model", default="classical", help="quantum or classical model")
    p.add_argument("--n_qubits", default=10)
    p.add_argument("--n_ql_layers", default=3)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_data, test_data, true_rul = load_cmapss(args.data_dir, args.subset)
    train_data, test_data, feature_scaler = preprocess(train_data, test_data)

    X_train, y_train = build_train_set(train_data, args.window_length, args.shift, args.max_rul)
    X_test, windows_per_engine = build_test_set(test_data, args.window_length, args.shift, args.num_test_windows)

    perm = np.random.permutation(len(y_train))
    X_train, y_train = X_train[perm], y_train[perm]

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1).astype(np.float32)

    print("train X", X_train.shape, "train y", y_train.shape)
    print("test X", X_test.shape, "engines", len(windows_per_engine))

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    num_features = X_train.shape[-1]
    if args.model == "classical":
        encoder = Encoder(args.num_hidden, args.num_layers, num_features, dropout=args.dropout)
    elif args.model == "quantum":
        encoder = QEncoder(args.num_hidden, args.num_layers, num_features, args.n_qubits, args.n_ql_layers, dropout=args.dropout)
    else:
        raise NotImplementedError("use either --model classical OR --model quantum")
    decoder = Decoder(num_features, args.num_hidden, args.num_layers, args.attention_size, dropout=args.dropout)
    net = Seq2Seq(encoder, decoder).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        loss = train_one_epoch(net, train_loader, optimizer, loss_fn, device)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {loss:.6f}")

    preds_scaled = predict(net, X_test, device)
    preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)

    split_points = np.cumsum(windows_per_engine)[:-1]
    preds_per_engine = np.split(preds, split_points)
    mean_preds = np.array([p.mean() for p in preds_per_engine])

    rmse = np.sqrt(mean_squared_error(true_rul, mean_preds))
    mae = mean_absolute_error(true_rul, mean_preds)
    print(f"[avg over {args.num_test_windows} windows] RMSE {rmse:.4f}  MAE {mae:.4f}")

    last_idx = np.cumsum(windows_per_engine) - 1
    preds_last = preds[last_idx]
    rmse_last = np.sqrt(mean_squared_error(true_rul, preds_last))
    mae_last = mean_absolute_error(true_rul, preds_last)
    print(f"[last window only]      RMSE {rmse_last:.4f}  MAE {mae_last:.4f}")

    s_score = compute_s_score(true_rul, preds_last)
    print(f"S-score: {s_score:.4f}")

    ckpt = {
        "model_state": net.state_dict(),
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "columns_to_drop": COLUMNS_TO_DROP,
        "window_length": args.window_length,
        "shift": args.shift,
        "max_rul": args.max_rul,
        "num_hidden": args.num_hidden,
        "num_layers": args.num_layers,
        "attention_size": args.attention_size,
        "num_features": num_features,
        "subset": args.subset,
        "n_qubits": args.n_qubits,
        "n_ql_layers": args.n_ql_layers
    }
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(ckpt, args.save_path)
    print(f"saved checkpoint -> {args.save_path}")


if __name__ == "__main__":
    main()