import torch
from torch import nn
from config import RNN_CONFIG
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config else RNN_CONFIG
        self.rnn = nn.LSTM(input_size=self.config['random_input'], hidden_size=self.config['hidden_size'],
                           num_layers=self.config['num_layers_G'], dropout=self.config['dropout_G'],
                           batch_first=True)
        self.lin = nn.Linear(in_features=self.config['hidden_size'], out_features=2 * self.config['music_size'])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # expects x to be of format (batch_size, sequence_length, random_units)
        x[:, :, ::4] = 4 * x[:, :, ::4]
        rnn_out, _ = self.rnn(x)
        # rnn_out: (batch, sequence_length, hidden_dim)
        x = self.lin(rnn_out)  # now (batch, sequence_length, music_size)
        left = self.softmax(x[:, :, :89])  # left hand
        right = self.softmax(x[:, :, 89:])  # right hand
        # the output is of shape (batch_size, sequence_length, music_size), with a probability for each note
        return left + right


class Discriminator(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config else RNN_CONFIG
        self.c1d = nn.Conv1d(in_channels=self.config['music_size'], out_channels=self.config['out_channels'],
                             kernel_size=self.config['kernel_size'], padding=self.config['padding'],
                             stride=self.config['stride'])
        self.maxpool = nn.MaxPool1d(kernel_size=self.config['kernel_size'], padding=self.config['padding'],
                                    stride=self.config['stride'])
        self.rnn = nn.LSTM(input_size=self.config['out_channels'], hidden_size=self.config['hidden_size'],
                           num_layers=self.config['num_layers_D'], dropout=self.config['dropout_D'],
                           batch_first=True, bidirectional=True)
        self.lin = nn.Linear(in_features=self.config['hidden_size'] * 2, out_features=1)

    def forward(self, x):
        # expects x to be of format (batch_size, sequence_length, music_size) with a one-hot encoding of the notes
        x = x.permute(0, 2, 1)  # reshape to have channel first; shape is (batch_size, music_size, sequence_length)
        x = self.c1d(x)  # shape (batch_size, out_channels, (sequence_length + 2*padding - kernel_size)/stride +1))
        # to shape (batch_size, out_channels,
        # (h_conv := (sequence_length + 2*padding - kernel_size)/stride +1) + 2*padding - kernel_size)/stride + 1)):
        x = self.maxpool(x)
        x = F.leaky_relu(x, 0.2)
        # to shape (batch_size, h_conv, out_channels)
        x = x.permute(0, 2, 1)
        # r sequence representation (batch, h_conv, h_rnn)
        r, _ = self.rnn(x)
        out = self.lin(r)  # shape (batch, h_conv, 1)
        out = out.squeeze()  # removing the last dimension
        out = torch.mean(out, dim=-1)  # averaging over time
        return out, r
