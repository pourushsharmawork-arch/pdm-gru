import torch
import torch.nn as nn

class Encoder(nn.Module):
    """RNN Encoder for Seq2Seq"""
    def __init__(self, num_hidden:int, num_layers:int, input_size:int, dropout=0):
        super().__init__()
        self.rnn = nn.GRU(input_size, num_hidden, num_layers,dropout=dropout if num_layers > 1 else 0.0, batch_first=True)

    def forward(self, X):
        """X.shape=(batch_size, num_steps, num_features)"""
        output, h_n = self.rnn(X)
        return output, h_n

def reparam(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

class VAEBottleneck(nn.Module):
    def __init__(self, hidden_size, latent_dim):
        super().__init__()
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, h_n):
        h_last = h_n[-1]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_size, num_layers, output_size, dropout=0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.l2h = nn.Linear(latent_dim, num_layers*hidden_size)
        self.rnn = nn.GRU(latent_dim, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, z, seq_len):
        batch_size = z.shape[0]
        dec_input = z.unsqueeze(1).expand(-1, seq_len, -1)
        h0 = self.l2h(z)
        h0 = h0.view(batch_size, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        output, _ = self.rnn(dec_input, h0)
        return self.fc_out(output)

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dim, dropout=0):
        super().__init__()
        self.encoder = Encoder(hidden_size, num_layers, input_size, dropout)
        self.bottleneck = VAEBottleneck(hidden_size, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_size, num_layers, input_size, dropout)

    def forward(self, x):
        seq_len = x.shape[1]
        _, h_n = self.encoder(x)
        mu, logvar = self.bottleneck(h_n)
        z = reparam(mu, logvar)
        recon = self.decoder(z, seq_len)
        return recon, mu, logvar
            