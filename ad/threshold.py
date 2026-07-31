import argparse
import json

import numpy as np
import torch

from ad.data import prepare_data
from ad.model import VAE

BASE_REDUCTIONS = ("mean", "max", "topk_time", "topk_cell")
_ALIASES = {"topk": "topk_time"}  # backward compat with earlier reduction name


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


def reduce_error(sq_err, reduction, topk=3):
    """Collapse (batch, seq_len, num_features) raw squared reconstruction
    error into one score per window.

    mean:       average over every cell.
    max:        single worst (timestep, sensor) cell.
    topk_time:  mean of the k worst timesteps, each pre-averaged over all
                sensors -- dilutes a single-sensor anomaly by the sensor
                count before ranking, but rewards anomalies spread across
                many sensors at once (e.g. a shared operating-condition
                shift).
    topk_cell:  mean of the k worst individual (timestep, sensor) cells,
                undiluted by averaging over sensors -- catches a
                single-sensor persistent anomaly (many elevated cells at
                full magnitude, not divided by feature count) and a
                single-timestep spike (one huge cell) without topk_time's
                dilution.
    """
    reduction = _ALIASES.get(reduction, reduction)
    batch_size = sq_err.shape[0]
    if reduction == "mean":
        return sq_err.mean(dim=(1, 2))
    elif reduction == "max":
        return sq_err.reshape(batch_size, -1).max(dim=1).values
    elif reduction == "topk_time":
        per_timestep = sq_err.mean(dim=2)
        k = min(topk, per_timestep.shape[1])
        return per_timestep.topk(k, dim=1).values.mean(dim=1)
    elif reduction == "topk_cell":
        flat = sq_err.reshape(batch_size, -1)
        k = min(topk, flat.shape[1])
        return flat.topk(k, dim=1).values.mean(dim=1)
    else:
        raise ValueError(f"Unknown reduction: {reduction!r} (expected {BASE_REDUCTIONS} or 'combined')")


@torch.no_grad()
def _reduced_scores(model, X, device, batch_size, reduction, topk):
    model.eval()
    scores = []
    for start in range(0, len(X), batch_size):
        batch = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
        recon = reconstruct_deterministic(model, batch)
        sq_err = (recon - batch) ** 2
        batch_scores = reduce_error(sq_err, reduction, topk=topk)
        scores.append(batch_scores.cpu().numpy())
    if not scores:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(scores)


def fit_ensemble_norm_stats(model, X_val, device, components, batch_size=256, topk=3):
    """Fit mean/std of each component reduction on healthy validation data,
    so differently-scaled signals become comparable z-scores before being
    combined. A raw max(topk_score, mean_score) would be meaningless since
    topk's raw value is always >= mean's (it's the average of a subset of
    the values that make up the mean)."""
    stats = {}
    for comp in components:
        scores = _reduced_scores(model, X_val, device, batch_size, comp, topk)
        stats[comp] = {"mu": float(scores.mean()), "std": float(scores.std())}
    return stats


def score_windows(model, X, device, batch_size=256, reduction="combined", topk=3,
                   norm_stats=None, components=("mean", "topk_time")):
    """Per-window reconstruction error, deterministic (z = mu, no sampling).

    reduction="combined" takes max(z_component_1, z_component_2, ...) over
    `components` -- an OR-of-detectors ensemble, so an anomaly only has to
    look anomalous under ONE reduction to be flagged, instead of picking a
    single reduction and trading its blind spots against its strengths.
    Requires norm_stats from fit_ensemble_norm_stats() with the same
    `components`.
    """
    if reduction == "combined":
        if norm_stats is None:
            raise ValueError("reduction='combined' requires norm_stats (see fit_ensemble_norm_stats).")
        z_scores = []
        for comp in components:
            raw = _reduced_scores(model, X, device, batch_size, comp, topk)
            stats = norm_stats[comp]
            z_scores.append((raw - stats["mu"]) / stats["std"])
        return np.maximum.reduce(z_scores)
    return _reduced_scores(model, X, device, batch_size, reduction, topk)


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
    parser.add_argument("--reduction", choices=list(BASE_REDUCTIONS) + ["combined"], default="combined",
                         help="How per-window reconstruction error is scored.")
    parser.add_argument("--topk", type=int, default=3, help="k for topk_time/topk_cell.")
    parser.add_argument("--components", default="mean,topk_time,topk_cell",
                         help="Comma-separated reductions to ensemble when --reduction=combined.")
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

    components = tuple(args.components.split(","))
    norm_stats = None
    if args.reduction == "combined":
        norm_stats = fit_ensemble_norm_stats(model, X_val, device, components, topk=args.topk)

    val_scores = score_windows(model, X_val, device, reduction=args.reduction, topk=args.topk,
                                norm_stats=norm_stats, components=components)
    threshold_p95 = compute_threshold(val_scores, 95)
    threshold_p99 = compute_threshold(val_scores, 99)
    selected = compute_threshold(val_scores, args.percentile)

    print(f"X_val windows: {len(X_val)}")
    print(f"val score: min={val_scores.min():.6f} mean={val_scores.mean():.6f} "
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
        "reduction": args.reduction,
        "topk": args.topk,
        "components": list(components) if args.reduction == "combined" else None,
    }
    if norm_stats is not None:
        payload["norm_stats"] = norm_stats
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved threshold to {args.output}")


if __name__ == "__main__":
    main()
