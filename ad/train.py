import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from ad.data import prepare_data
from ad.model import Encoder, VAE, VAEBottleneck, VAEDecoder, reparam
import copy
import joblib

def vae_loss(recon, x, mu, logvar, beta=0.1):
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kl_loss = -0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())
    return recon_loss + beta*kl_loss, recon_loss, kl_loss

def get_beta(epoch, warmup_epochs=20, beta_max=1.0):
    return beta_max*min(1.0, epoch/warmup_epochs)

X_train, X_val, X_test_normal, scaler = prepare_data(
    data_dir="data", subset="FD001",
    healthy_rul_threshold=100, window_length=30, shift=1, seed=42
)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
    batch_size=64, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
    batch_size=64, shuffle=False
)

def train_epoch(model, loader, optimizer, device, beta):
    model.train()
    total_loss = total_recon = total_kl = 0.0
    for (x,) in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_recon += recon_loss.item() * x.size(0)
        total_kl += kl_loss.item() * x.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_recon / n, total_kl / n

@torch.no_grad()
def validate(model, loader, device, beta):
    model.eval()
    total_loss = total_recon = 0.0
    for (x,) in loader:
        x = x.to(device)
        recon, mu, logvar = model(x)
        loss, recon_loss, _ = vae_loss(recon, x, mu, logvar, beta)
        total_loss += loss.item() * x.size(0)
        total_recon += recon_loss.item() * x.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_recon / n

def main():
    parser = argparse.ArgumentParser(description="Train the VAE anomaly detector on healthy CMAPSS windows.")
    parser.add_argument("--beta-max", type=float, default=1.0,
                         help="KL weight reached at the end of warmup (see get_beta). Lower values "
                              "prioritize reconstruction fidelity over a tightly-regularized latent space.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_size=14, hidden_size=64, num_layers=1, latent_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    best_state = None
    patience, patience_counter = 15, 0
    num_epochs = 200

    for epoch in range(num_epochs):
        beta = get_beta(epoch, warmup_epochs=20, beta_max=args.beta_max)
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device, beta)
        val_loss, val_recon = validate(model, val_loader, device, beta)
        print(f"epoch {epoch:3d} | beta {beta:.2f} | "
            f"train {train_loss:.4f} (recon {train_recon:.4f}, kl {train_kl:.4f}) | "
            f"val {val_loss:.4f} (recon {val_recon:.4f})")

        if val_loss<best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"early stopping at epoch{epoch}")
                break
    model.load_state_dict(best_state)
    torch.save({"model_state_dict": best_state,"config": {"input_size": 14, "hidden_size": 64, "num_layers": 1, "latent_dim": 16},}, "vae_checkpoint.pt")
    joblib.dump(scaler, "scaler.pkl")

if __name__ == "__main__":
    main()
