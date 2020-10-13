from torch import nn
import torch as torch
import numpy as np


class cnnNet(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, stride, pool, features, classes=0, padding=[0, 0], dilation=[1, 1]):
        """
        klasa implementująca konwolucyjną sieć neuronową
        :param input_ch: ilość kanałów wejsciowych
        :param output_ch: ilość kanałów wyjściowych
        :param kernel_size: wielkość filtrów
        :param stride: wielkość kroku
        :param pool: wielkość regionu z którego należy wyciągnać maksymalną wartość (używana funkcja MaxPool1d)
        :param features: ilość danych podawanych na wjeście sieci
        :param padding: wartość zwiększenia powierzchni przetwarzanych danych , domyślnie 0
        :param dilation: wartość domyślna 1
        """
        super(cnnNet, self).__init__()
        self.layers = []
        out_size = [int((features + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)]
        for index, (i, o, k, s, p, d) in enumerate(zip(input_ch, output_ch, kernel_size, stride, padding, dilation)):
            self.layers.append(nn.Conv1d(in_channels=i, out_channels=o, kernel_size=k, stride=s, padding=p, dilation=d))
            self.layers.append(nn.MaxPool1d(pool[index]))
            if index >= 1:
                out_size.append(int(
                    (out_size[-1] + 2 * padding[index] - dilation[index] * (kernel_size[index] - 1) - 1) / stride[
                        index] + 1))
            # wyjście "pooling layer" # domyślną wartością kroku jest wielkość filtru
            out_size.append(int((out_size[-1] + 2 * 0 - 1 * (pool[index] - 1) - 1) / pool[index] + 1))
        self.layers = nn.ModuleList(self.layers)

        # definiowanie warstw w pełni połączonych
        self.fc1 = nn.Linear(out_size[-1] * output_ch[-1], 25)
        self.fc2 = nn.Linear(25, 10)
        if classes == 0:
            self.fc3 = nn.Linear(10, 1)
        elif classes > 0:
            self.fc3 = nn.Linear(10, classes)
        self.classes = classes
    def forward(self, x, batch=False):
        # reshape data
        if not batch:
            out = x.view(1, 1, x.shape[0])
        else:
            out = x.view(x.shape[0], 1, x.shape[1])
        # warstawy konwolucyjne
        for layer in self.layers:
            out = layer(out)
        # przekształcenie wyjścia do odpowiedniego kształtu
        out = out.view(out.shape[0], out.shape[1] * out.shape[2])
        # warstawy w pełni połączone
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        if self.classes == 0:
            out = self.fc3(out)
        elif self.classes > 0:
            m = torch.nn.LogSoftmax(dim=1)
            out = m(self.fc3(out))

        return out

    def initNW(self, neuronsInLayers, layerNum, base):
        # inicializacja wag i biasów Nguyen-Widrow'a
        """
        funkja wzorowana na funkcji z bliblioteki NeuroLab
        https://pythonhosted.org/neurolab/index.html
        """
        weights = []
        bias = []  # tablica przechowujaca wektory przesuniec
        w_fix = 0.7 * (neuronsInLayers[0] ** (1 / base))
        w_rand = (np.random.rand(neuronsInLayers[0], base) * 2 - 1)
        w_rand = np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(neuronsInLayers[0], 1)) * w_rand
        w = w_fix * w_rand
        b = np.array([0]) if neuronsInLayers[0] == 1 else w_fix * np.linspace(-1, 1, neuronsInLayers[0]) * np.sign(
            w[:, 0])

        weights.append(w)
        bias.append(b)
        for i in range(1, layerNum):
            w_fix = 0.7 * (neuronsInLayers[i] ** (1 / neuronsInLayers[i - 1]))
            w_rand = (np.random.rand(neuronsInLayers[i], neuronsInLayers[i - 1]) * 2 - 1)
            w_rand = np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(neuronsInLayers[i], 1)) * w_rand
            w = w_fix * w_rand
            b = np.array([0]) if neuronsInLayers[i] == 1 else w_fix * np.linspace(-1, 1, neuronsInLayers[i]) * np.sign(
                w[:, 0])
            weights.append(w)
            bias.append(b)
        # dla ostatniej warstwy
        weights.append(np.random.rand(neuronsInLayers[-1]))
        bias.append(np.random.rand(1))
        return [weights, bias]
