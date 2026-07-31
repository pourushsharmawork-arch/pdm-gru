import argparse
import json

import numpy as np
import torch

from ad.data import prepare_data
from ad.model import VAE


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = VAE(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def reconstruct_deterministic(model, x):
    """Reconstruct x using z = mu, bypassing the reparameterization sampling."""
    seq_len = x.shape[1]
    _, h_n = model.encoder(x)
    mu, _ = model.bottleneck(h_n)
    return model.decoder(mu, seq_len)


@torch.no_grad()
def score_windows(model, X, device, batch_size=256):
    """Per-window MSE reconstruction error, deterministic (z = mu, no sampling)."""
    model.eval()
    scores = []
    for start in range(0, len(X), batch_size):
        batch = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
        recon = reconstruct_deterministic(model, batch)
        mse = torch.mean((recon - batch) ** 2, dim=(1, 2))
        scores.append(mse.cpu().numpy())
    if not scores:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(scores)


def compute_threshold(scores, percentile):
    return float(np.percentile(scores, percentile))


def main():
    parser = argparse.ArgumentParser(description="Compute an anomaly threshold from healthy validation reconstruction error.")
    parser.add_argument("--checkpoint", default="vae_checkpoint.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--healthy-rul-threshold", type=int, default=100)
    parser.add_argument("--window-length", type=int, default=30)
    parser.add_argument("--shift", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--percentile", type=float, default=95, help="Percentile used as the selected threshold.")
    parser.add_argument("--output", default="ad_threshold.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device)

    _, X_val, _, _ = prepare_data(
        data_dir=args.data_dir,
        subset=args.subset,
        healthy_rul_threshold=args.healthy_rul_threshold,
        window_length=args.window_length,
        shift=args.shift,
        seed=args.seed,
    )

    val_scores = score_windows(model, X_val, device)
    threshold_p95 = compute_threshold(val_scores, 95)
    threshold_p99 = compute_threshold(val_scores, 99)
    selected = compute_threshold(val_scores, args.percentile)

    print(f"X_val windows: {len(X_val)}")
    print(f"val MSE: min={val_scores.min():.6f} mean={val_scores.mean():.6f} "
          f"std={val_scores.std():.6f} max={val_scores.max():.6f}")
    print(f"threshold @95th percentile: {threshold_p95:.6f}")
    print(f"threshold @99th percentile: {threshold_p99:.6f}")
    print(f"selected threshold (percentile={args.percentile}): {selected:.6f}")

    payload = {
        "percentile": args.percentile,
        "threshold": selected,
        "threshold_p95": threshold_p95,
        "threshold_p99": threshold_p99,
        "val_mse_mean": float(val_scores.mean()),
        "val_mse_std": float(val_scores.std()),
        "num_val_windows": len(X_val),
        "checkpoint": args.checkpoint,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved threshold to {args.output}")


if __name__ == "__main__":
    main()
