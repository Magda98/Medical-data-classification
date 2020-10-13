import torch
from torch import nn
import numpy as np


class lstmNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, classes=0):
        """
        klasa inplementująca neuronową sieć rekurencyjną LSTM
        :param input_size: wielkość wejścia
        :param output_size: wielkość wyjścia
        :param hidden_size: ilość neuronów w warstwie ukrytej
        :param n_layers: ilość warstw ukrytych
        """
        super(lstmNet, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=False)
        self.lstm.cuda()
        self.fo = nn.Linear(hidden_size, output_size)
        self.classes = classes

    def forward(self, x):
        # przekształcenie danych do tensora w postaci (seq_len, batch, input_size)
        if len(list(x.shape)) == 1:
            x = x.view(1,1,x.shape[0])
        else:
            x = x.view(1,x.shape[0], x.shape[1])
        batch_size = x.shape[1]
        # inicializacja hidden state oraz cell state
        hidden = self.init_hidden(batch_size, self.hidden_size, self.n_layers)
        hidden = (hidden[0].cuda(), hidden[1].cuda())
        # przejście sieci w przód
        out, hidden = self.lstm(x, hidden)
        output = out.view(-1, self.hidden_size)
        output = self.fo(output)

        return output, hidden

    def init_hidden(self, batch_size, hidden_size, n_layers):
        """
        funkcja inicializująca stan krótkoterminowy (h) oraz długoterminowy (c)
        :param batch_size: wielkość partii danych
        :param hidden_size: wielkość warstwy ukrytej
        :param n_layers: ilość warstw ukrytych
        :return:
        """
        h0 = torch.zeros(n_layers, batch_size, hidden_size)
        c0 = torch.zeros(n_layers, batch_size, hidden_size)
        return h0, c0
