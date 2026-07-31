from __future__ import annotations
import pennylane as qml
import torch
import torch.nn as nn

def build_quantum_layer(n_qubits:int, n_ql_layers:int, shots:int|None=None):
    """Implements a simpler version of QAD's pennylane layer, default.qubit as backend, and backprop as diff method"""
    dev_kwargs = {"wires":n_qubits}
    if shots is not None:
        dev_kwargs['shots'] = shots

    dev = qml.device("default.qubit", **dev_kwargs)
    @qml.qnode(dev, interface='torch', diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_ql_layers, n_qubits, 3)}
    torch_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
    return torch_layer

class QEncoder(nn.Module):
    """RNN Encoder for Seq2Seq"""
    def __init__(self, num_hidden:int, num_layers:int, input_size:int, n_qubits:int, n_ql_layers:int, dropout=0):
        super().__init__()
        self.rnn = nn.GRU(input_size, num_hidden, num_layers,dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.fcr = nn.Linear(num_hidden, n_qubits)
        self.qc = build_quantum_layer(n_qubits, n_ql_layers, shots=None)
        self.fce = nn.Linear(n_qubits, num_hidden)

    def forward(self, X):
        """X.shape=(batch_size, num_steps, num_features)"""
        output, h_n = self.rnn(X)
        h_last = h_n[-1]
        h_reduced = self.fcr(h_last)
        q_out = self.qc(h_reduced)
        q_out = q_out.squeeze(-1)
        h_q = self.fce(q_out)

        qh_n = h_n.clone()
        qh_n[-1] = h_q
        return output, qh_n

