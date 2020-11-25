import torch
from torch import nn
from config import RNN_CONFIG
import torch.functional as F


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(input_size=RNN_CONFIG['random_input'], hidden_size=RNN_CONFIG['hidden_size'],
                           num_layers=RNN_CONFIG['num_layers'], dropout=RNN_CONFIG['dropout'],
                           batch_first=True)
        self.lin = nn.Linear(in_features=RNN_CONFIG['hidden_size'], out_features=RNN_CONFIG['music_size'])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # expects x to be of format (batch_size, sequence_length, random_units)
        rnn_out, _ = self.rnn(x)
        # rnn_out: (batch, sequence_length, hidden_dim)
        x = self.lin(rnn_out)
        x = self.softmax(x)
        return x  # the output is of shape (batch_size, sequence_length, music_size), with a probability for each note


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(input_size=RNN_CONFIG['music_size'], hidden_size=RNN_CONFIG['hidden_size'],
                           num_layers=RNN_CONFIG['num_layers'], dropout=RNN_CONFIG['dropout'],
                           batch_first=True, bidirectional=True)
        self.lin = nn.Linear(in_features=RNN_CONFIG['hidden_size']*2, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # expects x to be of format (batch_size, sequence_length, music_size) with a one-hot encoding of the notes
        r, _ = self.rnn(x)  # r sequence representation of shape (batch, sequence_length, hidden_dim)
        out = self.lin(r)  # shape (batch, sequence_length, 1)
        out = self.sigmoid(out)
        out = out.squeeze()  # removing the last dimension
        out = torch.mean(out, dim=-1)  # averaging over time
        return out, r
