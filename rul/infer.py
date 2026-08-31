import argparse
import os

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from rul.data import process_input, process_test, load_cmapss
from rul.models.classical import Decoder, Encoder, Seq2Seq
from rul.models.quantum import QEncoder


def load_checkpoint(path, device, model):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if model == "classical":
        encoder = Encoder(ckpt["num_hidden"], ckpt["num_layers"], ckpt["num_features"])
    elif model == "quantum":
        encoder = QEncoder(ckpt["num_hidden"], ckpt["num_layers"], ckpt["num_features"], n_qubits=ckpt["n_qubits"], n_ql_layers=ckpt["n_ql_layers"])
    decoder = Decoder(ckpt["num_features"], ckpt["num_hidden"], ckpt["num_layers"], ckpt["attention_size"])
    net = Seq2Seq(encoder, decoder).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    return net, ckpt


def scale_test_data(test_data, feature_scaler, columns_to_drop):
    ids = test_data[0]
    scaled = feature_scaler.transform(test_data.drop(columns=columns_to_drop))
    return pd.DataFrame(np.c_[ids, scaled])


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


def predict_fleet_last_window(net, test_data, window_length, shift, num_test_windows, target_scaler, device):
    """One RUL number per engine: average prediction over the last `num_test_windows` windows."""
    engine_ids = np.sort(test_data[0].unique())
    preds_per_engine = []
    for engine_id in engine_ids:
        engine = test_data[test_data[0] == engine_id].drop(columns=[0]).values
        if len(engine) < window_length:
            raise ValueError(f"engine {engine_id}: fewer rows ({len(engine)}) than window_length={window_length}")
        X, _ = process_test(engine, window_length, shift, num_test_windows)
        preds_scaled = predict(net, X.astype(np.float32), device)
        preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
        preds_per_engine.append(preds.mean())
    return engine_ids, np.array(preds_per_engine)


def predict_engine_trajectory(net, test_data, engine_id, window_length, shift, true_rul, target_scaler, device):
    """Full RUL trajectory across one engine's recorded cycles, plus the true (unclipped) RUL line.

    True RUL at cycle c is derived from RUL_FD001.txt: the file gives the RUL at the *last*
    recorded cycle, so total_life = recorded_len + final_true_rul, and true_rul(c) = total_life - c.
    """
    engine_ids = np.sort(test_data[0].unique())
    if engine_id not in engine_ids:
        raise ValueError(f"engine_id {engine_id} not in test set (available: {engine_ids.min()}-{engine_ids.max()})")

    engine = test_data[test_data[0] == engine_id].drop(columns=[0]).values
    if len(engine) < window_length:
        raise ValueError(f"engine {engine_id}: fewer rows ({len(engine)}) than window_length={window_length}")

    X = process_input(engine, None, window_length, shift)
    num_windows = X.shape[0]
    end_cycles = window_length + shift * np.arange(num_windows)  # 1-indexed cycle each window predicts at

    preds_scaled = predict(net, X.astype(np.float32), device)
    preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)

    engine_pos = int(np.where(engine_ids == engine_id)[0][0])
    final_true_rul = true_rul[engine_pos]
    recorded_len = len(engine)
    total_life = recorded_len + final_true_rul
    all_cycles = np.arange(1, recorded_len + 1)
    true_traj = total_life - all_cycles

    return end_cycles, preds, all_cycles, true_traj


def visualize_rul(mode, net, test_data, true_rul, target_scaler, window_length, shift, device,
                   num_test_windows=5, engine_id=None, max_rul=None, output=None, show=False):
    fig, ax = plt.subplots(figsize=(10, 6))

    if mode == "overall":
        engine_ids, preds = predict_fleet_last_window(
            net, test_data, window_length, shift, num_test_windows, target_scaler, device
        )
        x = np.arange(len(engine_ids))
        ax.plot(x, true_rul, label="True RUL", color="red", marker="o", markersize=3)
        ax.plot(x, preds, label="Predicted RUL", color="blue", marker="o", markersize=3)
        ax.set_xlabel("Engine index (test set)")
        ax.set_ylabel("RUL (cycles)")
        rmse = np.sqrt(mean_squared_error(true_rul, preds))
        mae = mean_absolute_error(true_rul, preds)
        ax.set_title(f"RUL: actual vs predicted across test fleet (n={len(engine_ids)})  "
                     f"RMSE={rmse:.2f} MAE={mae:.2f}")

    elif mode == "engine":
        if engine_id is None:
            raise ValueError("engine_id is required for mode='engine'")
        end_cycles, preds, all_cycles, true_traj = predict_engine_trajectory(
            net, test_data, engine_id, window_length, shift, true_rul, target_scaler, device
        )
        ax.plot(all_cycles, true_traj, label="True RUL (raw)", color="red", linewidth=2)
        if max_rul is not None:
            clipped_true = np.minimum(true_traj, max_rul)
            ax.plot(all_cycles, clipped_true, label=f"True RUL (clipped @ {max_rul}, training target)",
                    color="darkorange", linewidth=1.5, linestyle=":")
        ax.plot(end_cycles, preds, label="Predicted RUL", color="blue", marker="o",
                markersize=4, linestyle="--")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("RUL (cycles)")
        ax.set_title(f"RUL trajectory — engine {engine_id}")

    else:
        raise ValueError(f"unknown mode '{mode}', expected 'overall' or 'engine'")

    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"saved plot -> {output}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--subset", default=None, help="defaults to subset stored in checkpoint")
    p.add_argument("--mode", choices=["overall", "engine"], default="overall")
    p.add_argument("--engine-id", type=int, default=None, help="required when --mode engine")
    p.add_argument("--num-test-windows", type=int, default=5, help="only used for --mode overall")
    p.add_argument("--traj-shift", type=int, default=None,
                    help="window shift for engine trajectory (finer than training shift = smoother curve); "
                         "defaults to the shift used at training time")
    p.add_argument("--output", default="rul_predictions.png")
    p.add_argument("--show", action="store_true")
    p.add_argument("--model", default="classical", help="quantum or classical model")
    args = p.parse_args()

    if args.mode == "engine" and args.engine_id is None:
        p.error("--engine-id is required when --mode engine")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, ckpt = load_checkpoint(args.checkpoint, device, model=args.model)
    subset = args.subset or ckpt["subset"]
    print(f"inferring using {args.model} model.")

    _, test_data, true_rul = load_cmapss(args.data_dir, subset)
    test_data = scale_test_data(test_data, ckpt["feature_scaler"], ckpt["columns_to_drop"])

    shift = args.traj_shift if (args.mode == "engine" and args.traj_shift is not None) else ckpt["shift"]

    visualize_rul(
        mode=args.mode,
        net=net,
        test_data=test_data,
        true_rul=true_rul,
        target_scaler=ckpt["target_scaler"],
        window_length=ckpt["window_length"],
        shift=shift,
        device=device,
        num_test_windows=args.num_test_windows,
        engine_id=args.engine_id,
        max_rul=ckpt.get("max_rul"),
        output=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()