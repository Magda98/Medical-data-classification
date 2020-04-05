
import torch
from torch import nn
import numpy as np

class Lstm_Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):

        super(Lstm_Net, self).__init__()
        self.hidden_size = hidden_size
        self.output_size  = output_size
        self.n_layers = n_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=False)
        self.lstm.cuda()
        # change to softmax output size must be [1,class]
        self.fo = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        if len(list(x.shape)) == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        else:
            x = x.unsqueeze(0)
        batch_size = x.size(1)
        hidden = self.init_hidden(batch_size, self.hidden_size, self.n_layers)
        hidden = (hidden[0].cuda(), hidden[1].cuda())
        out, hidden = self.lstm(x, hidden)
        output = out.contiguous().view(-1, self.hidden_size)
        output = self.fo(output)
        return output, hidden

    def init_hidden(self, batch_size, hidden_size, n_layers):
        return torch.zeros(n_layers, batch_size, hidden_size), torch.zeros(n_layers, batch_size, hidden_size)