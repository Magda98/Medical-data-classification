import torch
from torch import nn
import numpy as np


class lstmNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):

        super(lstmNet, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=False)
        self.lstm.cuda()
        self.fo = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if len(list(x.shape)) == 1:
            x = x.view(1,1,x.shape[0])
        else:
            x = x.view(1,x.shape[0], x.shape[1])
        batch_size = x.shape[1]
        hidden = self.init_hidden(batch_size, self.hidden_size, self.n_layers)
        hidden = (hidden[0].cuda(), hidden[1].cuda())
        out, hidden = self.lstm(x, hidden)
        output = out.contiguous().view(-1, self.hidden_size)
        output = self.fo(output)
        return output, hidden

    def init_hidden(self, batch_size, hidden_size, n_layers):
        h0 = torch.zeros(n_layers, batch_size, hidden_size)
        c0 = torch.zeros(n_layers, batch_size, hidden_size)
        return h0, c0
