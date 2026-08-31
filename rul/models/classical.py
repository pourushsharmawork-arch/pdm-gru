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

class AdditiveAttention(nn.Module):
    def __init__(self, input_size:int, attention_len:int):
        super().__init__()
        self.attention = nn.Linear(input_size, attention_len)

    def forward(self, h_n, outputs):
        """
        outputs.shape = (batch_size, num_steps, hidden_dim)
        state.shape = (batch_size, hidden_dim)
        """
        _, seq_len, _ = outputs.shape
        averaged_state = h_n.mean(dim=0)                                # (batch, hidden)
        state_rep = averaged_state.unsqueeze(1).expand(-1, seq_len, -1) # (batch, seq_len, hidden)
        concat = torch.cat((state_rep, outputs), dim=-1)                # (batch, seq_len, attention_size)
        scores = torch.tanh(self.attention(concat))                     # (batch, seq_len, attention_size)
        attn_weights = torch.softmax(scores.sum(dim=-1), dim=-1)        # (batch, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), outputs)         # (batch, 1, hidden)

        return context

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_size=32, dropout=0):
        super().__init__()
        self.attention = AdditiveAttention(2*hidden_size, attention_size)
        self.rnn = nn.GRU(input_size + hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def init_state(self, enc_outputs):
        outputs, hidden = enc_outputs
        return outputs, hidden

    def forward(self, dec_input, state):
        """
        dec_input   : (batch, 1, input_size)
        enc_outputs : (batch, seq_len, hidden)
        hidden      : (num_layers, batch, hidden)
        """
        enc_outputs, hidden = state
        context = self.attention(hidden, enc_outputs)
        rnn_input = torch.cat((dec_input, context), dim=-1)
        rnn_output, hidden = self.rnn(rnn_input, hidden)
        output = self.fc(rnn_output.squeeze(1))
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, e_X, d_X):
        enc_outputs = self.encoder(e_X)
        dec_state = self.decoder.init_state(enc_outputs)
        output, hidden = self.decoder(d_X, dec_state)
        return output, hidden