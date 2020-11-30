import torch
from torch import nn
from config import RNN_CONFIG


class Generator(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config else RNN_CONFIG
        self.rnn = nn.LSTM(input_size=self.config['random_input'], hidden_size=self.config['hidden_size'],
                           num_layers=self.config['num_layers'], dropout=self.config['dropout'],
                           batch_first=True)
        self.lin = nn.Linear(in_features=self.config['hidden_size'], out_features=self.config['music_size'])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # expects x to be of format (batch_size, sequence_length, random_units)
        rnn_out, _ = self.rnn(x)
        # rnn_out: (batch, sequence_length, hidden_dim)
        x = self.lin(rnn_out)  # now (batch, sequence_length, music_size)
        left = self.softmax(x[:, :, :89])  # left hand
        right = self.softmax(x[:, :, 89:])  # right hand
        x = torch.cat((left, right), dim=-1)
        return x  # the output is of shape (batch_size, sequence_length, music_size), with a probability for each note


class Discriminator(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config else RNN_CONFIG
        self.rnn = nn.LSTM(input_size=self.config['music_size'], hidden_size=self.config['hidden_size'],
                           num_layers=self.config['num_layers'], dropout=self.config['dropout'],
                           batch_first=True, bidirectional=True)
        self.lin = nn.Linear(in_features=self.config['hidden_size']*2, out_features=1)

    def forward(self, x):
        # expects x to be of format (batch_size, sequence_length, music_size) with a one-hot encoding of the notes
        r, _ = self.rnn(x)  # r sequence representation of shape (batch, sequence_length, hidden_dim)
        out = self.lin(r)  # shape (batch, sequence_length, 1)
        out = out.squeeze()  # removing the last dimension
        out = torch.mean(out, dim=-1)  # averaging over time
        return out, r
