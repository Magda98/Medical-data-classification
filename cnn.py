
from torch import nn
import torch as torch

class Cnn_Net(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride):

        super(Cnn_Net, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=kernel_size, stride=stride)
        self.pool = nn.MaxPool1d(1)
        self.cnn1 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1, stride=stride)
        self.pool1 = nn.MaxPool1d(1)
        # self.cnn2 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2, stride=stride)
        # self.pool2 = nn.MaxPool1d(output_size)
        self.cnn.cuda()
        self.fc11 = nn.Linear(7040, 50)
        self.fc1 = nn.Linear(352, 50)
        self.fc2 = nn.Linear(50, 1)
        self.fc3 = nn.Linear(5, output_size)

    def forward(self, x, batch=False):

        if not batch:
            x = x.view(1,1,x.shape[0])
        else:
            x = x.view(x.shape[0],1,x.shape[1])

        out = self.cnn(x)
        out = self.pool(out)
        out = self.cnn1(out)
        out = self.pool1(out)
        # out = self.cnn2(out)
        # out = self.pool2(out)
        # out = out.squeeze(1)
        out = out.view(out.shape[0], out.shape[1]*out.shape[2])
        # out = out.unsqueeze(0)
        out = torch.sigmoid(self.fc1(out))

        out = self.fc2(out)
        # out = self.fc3(out)
        return out
