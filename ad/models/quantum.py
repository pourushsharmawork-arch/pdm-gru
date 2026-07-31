import torch
import torch.nn as nn
import pennylane as qml

def make_qlayer(n_qubits, n_qlayers, dev_name="default.qubit"):
    dev = qml.device(dev_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_qlayers, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)

class QLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits=4, n_qlayers=2):
        super().__init__()
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size

        self.forget_in = nn.Linear(concat_size, n_qubits)
        self.input_in = nn.Linear(concat_size, n_qubits)
        self.update_in = nn.Linear(concat_size, n_qubits)
        self.output_in = nn.Linear(concat_size, n_qubits)

        self.VQC_forget = make_qlayer(n_qubits, n_qlayers)
        self.VQC_input = make_qlayer(n_qubits, n_qlayers)
        self.VQC_update = make_qlayer(n_qubits, n_qlayers)
        self.VQC_output = make_qlayer(n_qubits, n_qlayers)

        self.forget_out = nn.Linear(n_qubits, hidden_size)
        self.input_out = nn.Linear(n_qubits, hidden_size)
        self.update_out = nn.Linear(n_qubits, hidden_size)
        self.output_out = nn.Linear(n_qubits, hidden_size)

    def forward(self, x, h, c):
        v = torch.cat([x, h], dim=-1)
        f = torch.sigmoid(self.forget_out(self.VQC_forget(self.forget_in(v))))
        i = torch.sigmoid(self.input_out(self.VQC_input(self.input_in(v))))
        g = torch.tanh(self.update_out(self.VQC_update(self.update_in(v))))
        o = torch.sigmoid(self.output_out(self.VQC_output(self.output_in(v))))
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class QLSTM(nn.Module):
    """Drop-in for nn.GRU (single layer, batch_first). Returns (output, h_n)
    with h_n shaped (1, batch, hidden) to match the GRU convention used in
    the original VAEBottleneck (h_n[-1])."""

    def __init__(self, input_size, hidden_size, n_qubits=4, n_qlayers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = QLSTMCell(input_size, hidden_size, n_qubits, n_qlayers)

    def forward(self, x, h0=None):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x.device) if h0 is None else h0
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []
        for t in range(seq_len):
            h, c = self.cell(x[:, t, :], h, c)
            outputs.append(h.unsqueeze(1))
        output = torch.cat(outputs, dim=1)
        return output, h.unsqueeze(0)


def reparam(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# OPTION A — Full quantum: QLSTM replaces GRU in both encoder and decoder, bottleneck stays classical. 
class QEncoder(nn.Module):
    def __init__(self, hidden_size, input_size, n_qubits=4, n_qlayers=2):
        super().__init__()
        self.rnn = QLSTM(input_size, hidden_size, n_qubits, n_qlayers)

    def forward(self, X):
        return self.rnn(X)


class VAEBottleneck(nn.Module):
    """Unchanged classical bottleneck — reused as-is."""

    def __init__(self, hidden_size, latent_dim):
        super().__init__()
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, h_n):
        h_last = h_n[-1]
        return self.fc_mu(h_last), self.fc_logvar(h_last)


class QVAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_size, output_size, n_qubits=4, n_qlayers=2):
        super().__init__()
        self.l2h = nn.Linear(latent_dim, hidden_size)
        self.rnn = QLSTM(latent_dim, hidden_size, n_qubits, n_qlayers)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, z, seq_len):
        dec_input = z.unsqueeze(1).expand(-1, seq_len, -1)
        h0 = self.l2h(z)
        output, _ = self.rnn(dec_input, h0)
        return self.fc_out(output)


class QVAE_FullQuantum(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, n_qubits=4, n_qlayers=2):
        super().__init__()
        self.encoder = QEncoder(hidden_size, input_size, n_qubits, n_qlayers)
        self.bottleneck = VAEBottleneck(hidden_size, latent_dim)
        self.decoder = QVAEDecoder(latent_dim, hidden_size, input_size, n_qubits, n_qlayers)

    def forward(self, x):
        seq_len = x.shape[1]
        _, h_n = self.encoder(x)
        mu, logvar = self.bottleneck(h_n)
        z = reparam(mu, logvar)
        recon = self.decoder(z, seq_len)
        return recon, mu, logvar


# OPTION B — Quantum bottleneck only: encoder/decoder stay classical GRU, only the latent space is quantum.
class Encoder(nn.Module):
    def __init__(self, num_hidden, num_layers, input_size, dropout=0):
        super().__init__()
        self.rnn = nn.GRU(input_size, num_hidden, num_layers,
                           dropout=dropout if num_layers > 1 else 0.0, batch_first=True)

    def forward(self, X):
        return self.rnn(X)


class QVAEBottleneck(nn.Module):
    def __init__(self, hidden_size, latent_dim, n_qlayers=2):
        super().__init__()
        assert latent_dim >= 1, "latent_dim doubles as n_qubits here"
        self.pre = nn.Linear(hidden_size, latent_dim)
        self.VQC_mu = make_qlayer(latent_dim, n_qlayers)
        self.VQC_logvar = make_qlayer(latent_dim, n_qlayers)

    def forward(self, h_n):
        h_last = h_n[-1]
        angles = torch.tanh(self.pre(h_last)) * torch.pi  # -> valid rotation range
        mu = self.VQC_mu(angles)          # in [-1, 1], <Z> expectation
        logvar = self.VQC_logvar(angles)  # also in [-1, 1]
        return mu, logvar


class VAEDecoder(nn.Module):
    """Unchanged classical decoder — reused as-is."""

    def __init__(self, latent_dim, hidden_size, num_layers, output_size, dropout=0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.l2h = nn.Linear(latent_dim, num_layers * hidden_size)
        self.rnn = nn.GRU(latent_dim, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, z, seq_len):
        batch_size = z.shape[0]
        dec_input = z.unsqueeze(1).expand(-1, seq_len, -1)
        h0 = self.l2h(z)
        h0 = h0.view(batch_size, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        output, _ = self.rnn(dec_input, h0)
        return self.fc_out(output)


class QVAE_BottleneckOnly(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dim, dropout=0, n_qlayers=2):
        super().__init__()
        self.encoder = Encoder(hidden_size, num_layers, input_size, dropout)
        self.bottleneck = QVAEBottleneck(hidden_size, latent_dim, n_qlayers)
        self.decoder = VAEDecoder(latent_dim, hidden_size, num_layers, input_size, dropout)

    def forward(self, x):
        seq_len = x.shape[1]
        _, h_n = self.encoder(x)
        mu, logvar = self.bottleneck(h_n)
        z = reparam(mu, logvar)
        recon = self.decoder(z, seq_len)
        return recon, mu, logvar


if __name__ == "__main__":
    x = torch.randn(2, 10, 14)  # (batch, seq_len, n_sensors) — small for speed

    m_a = QVAE_FullQuantum(input_size=14, hidden_size=8, latent_dim=4, n_qubits=4, n_qlayers=1)
    recon, mu, logvar = m_a(x)
    print("A full-quantum:", recon.shape, mu.shape, logvar.shape)

    m_b = QVAE_BottleneckOnly(input_size=14, hidden_size=16, num_layers=1, latent_dim=4, n_qlayers=1)
    recon, mu, logvar = m_b(x)
    print("B bottleneck-only:", recon.shape, mu.shape, logvar.shape, "mu range:", mu.min().item(), mu.max().item())
