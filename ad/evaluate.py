import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ad.data import (
    COLUMNS_TO_DROP,
    add_rul,
    filter_eligible_engines,
    load_fd001,
    process_input,
    split_data,
    split_engines,
)
from ad.threshold import load_checkpoint, score_windows


def load_synthetic_dataset(synthetic_dir, subset="FD001"):
    """Load the CMAPSS-formatted synthetic set written by ad/datagen.py."""
    data_path = os.path.join(synthetic_dir, f"synthetic_{subset}.txt")
    labels_path = os.path.join(synthetic_dir, f"labels_{subset}.txt")

    raw = pd.read_csv(data_path, sep=r"\s+", header=None)
    sample_types = pd.read_csv(labels_path, header=None)[0].values

    num_features = raw.shape[1] - 2
    num_units = raw[0].nunique()
    window_length = len(raw) // num_units

    X = raw.drop(columns=[0, 1]).values.reshape(num_units, window_length, num_features)
    return X, sample_types


def load_unhealthy_windows(data_dir, subset, healthy_rul_threshold, window_length, shift, seed, scaler):
    """Score the unhealthy portion of the test-split engines (RUL < threshold),
    the rows select_healthy() normally discards. Returns windows plus the RUL
    at the last cycle of each window, so detection rate can be checked against
    proximity to failure."""
    data = load_fd001(data_dir, subset)
    data = add_rul(data)
    data = filter_eligible_engines(data, healthy_rul_threshold, window_length)
    train_ids, val_ids, test_ids = split_engines(data, seed=seed)
    _, _, test_data = split_data(data, train_ids, val_ids, test_ids)

    unhealthy = test_data[test_data["RUL"] < healthy_rul_threshold].copy()

    row_counts = unhealthy.groupby(0).size()
    eligible_ids = row_counts[row_counts >= window_length].index
    unhealthy = unhealthy[unhealthy[0].isin(eligible_ids)].copy()

    if len(unhealthy) == 0:
        return np.empty((0, window_length, scaler.n_features_in_)), np.empty(0)

    scaled = scaler.transform(unhealthy.drop(columns=COLUMNS_TO_DROP + ["RUL"]))
    scaled_df = pd.DataFrame(np.c_[unhealthy[0].values, unhealthy[1].values, scaled])

    windows = []
    window_ruls = []
    for engine_id in scaled_df[0].unique():
        engine_rows = scaled_df[scaled_df[0] == engine_id]
        sensor_data = engine_rows.drop(columns=[0, 1]).values
        engine_windows = process_input(sensor_data, None, window_length, shift)
        windows.append(engine_windows)

        engine_rul = unhealthy.loc[unhealthy[0] == engine_id, "RUL"].values
        num_batches = engine_windows.shape[0]
        window_ruls.extend(
            engine_rul[shift * b + (window_length - 1)] for b in range(num_batches)
        )

    X_unhealthy = np.concatenate(windows, axis=0)
    return X_unhealthy, np.array(window_ruls)


def evaluate_synthetic(model, device, threshold, synthetic_dir, subset):
    X, sample_types = load_synthetic_dataset(synthetic_dir, subset)
    y_true = (sample_types != "normal").astype(int)
    scores = score_windows(model, X, device)
    y_pred = (scores > threshold).astype(int)

    auroc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n=== Synthetic benchmark ===")
    print(f"windows: {len(X)} (normal={np.sum(y_true == 0)}, anomaly={np.sum(y_true == 1)})")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"At threshold={threshold:.6f}: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

    print("\nDetection rate by anomaly type (recall within that type):")
    for anomaly_type in sorted(set(sample_types) - {"normal"}):
        mask = sample_types == anomaly_type
        detected = np.mean(scores[mask] > threshold)
        mean_score = scores[mask].mean()
        print(f"  {anomaly_type:6s} n={mask.sum():4d}  detected={detected:.4f}  mean_mse={mean_score:.6f}")

    normal_mask = sample_types == "normal"
    false_positive_rate = np.mean(scores[normal_mask] > threshold)
    print(f"  {'normal':6s} n={normal_mask.sum():4d}  false_positive_rate={false_positive_rate:.4f}  "
          f"mean_mse={scores[normal_mask].mean():.6f}")


def evaluate_unhealthy(model, device, threshold, args, scaler):
    X_unhealthy, window_ruls = load_unhealthy_windows(
        args.data_dir, args.subset, args.healthy_rul_threshold,
        args.window_length, args.shift, args.seed, scaler,
    )

    print("\n=== Real-world sanity check (unhealthy test-engine windows) ===")
    if len(X_unhealthy) == 0:
        print("No eligible unhealthy windows found (engines too short past the healthy cutoff).")
        return

    scores = score_windows(model, X_unhealthy, device)
    flagged = scores > threshold
    print(f"windows: {len(X_unhealthy)}")
    print(f"overall flagged-as-anomalous rate: {flagged.mean():.4f}")
    print("(no ground-truth labels here -- this is a proxy check, not a benchmark)")

    bin_edges = np.arange(0, args.healthy_rul_threshold + 10, 10)
    bin_ids = np.digitize(window_ruls, bin_edges)
    print("\nDetection rate by remaining RUL at window end (expect it to rise near failure):")
    for b in sorted(set(bin_ids)):
        lo, hi = bin_edges[b - 1] if b > 0 else 0, bin_edges[b] if b < len(bin_edges) else bin_edges[-1]
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        print(f"  RUL [{lo:3d},{hi:3d}) n={mask.sum():4d}  flagged_rate={flagged[mask].mean():.4f}  "
              f"mean_mse={scores[mask].mean():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained VAE anomaly detector.")
    parser.add_argument("--checkpoint", default="vae_checkpoint.pt")
    parser.add_argument("--scaler", default="scaler.pkl")
    parser.add_argument("--threshold-file", default="ad_threshold.json")
    parser.add_argument("--threshold", type=float, default=None, help="Override the threshold from --threshold-file.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--synthetic-dir", default="data/synthetic")
    parser.add_argument("--healthy-rul-threshold", type=int, default=100)
    parser.add_argument("--window-length", type=int, default=30)
    parser.add_argument("--shift", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["synthetic", "unhealthy", "both"], default="both")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device)

    if args.threshold is not None:
        threshold = args.threshold
    else:
        with open(args.threshold_file) as f:
            threshold = json.load(f)["threshold"]
    print(f"Using threshold: {threshold:.6f}")

    if args.mode in ("synthetic", "both"):
        evaluate_synthetic(model, device, threshold, args.synthetic_dir, args.subset)

    if args.mode in ("unhealthy", "both"):
        scaler = joblib.load(args.scaler)
        evaluate_unhealthy(model, device, threshold, args, scaler)


if __name__ == "__main__":
    main()
