
import torch
from torch import nn
import numpy as np

class Cnn_Net(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride):

        super(Cnn_Net, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size, stride=stride)
        self.pool = nn.MaxPool1d(output_size)
        self.cnn.cuda()
        self.fo = nn.Linear( 7, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        # print(x.shape)
        out = self.cnn(x)
        # print(out.shape)
        out = self.pool(out)
        out = out.squeeze(1)
        out = self.fo(out)
        # print(out.shape)
        # print(out.shape)
        return out
