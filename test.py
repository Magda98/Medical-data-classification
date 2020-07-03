import torch
import matplotlib.pyplot as plt
import random
from lstm import lstmNet
from cnn import cnnNet


def test(cross_data, netType):
    # loading model
    model = torch.load("model")

    cross_data.k = random.randint(0, 9)
    cross_data.select_k()
    cross_data.input_output()

    if netType == 'lstm':
        t_out, t_hidden = model(cross_data.test_inp.float())
    if netType == 'cnn':
        t_out = model(cross_data.test_inp.float(), True)

    plt.plot(t_out.cpu().detach().numpy(), color='#4daf4a', marker='o', label="wyjscie sieci")
    plt.plot(cross_data.test_out.cpu().detach().numpy(), color='#e55964', marker='o', label="target")
    plt.axhline(y=0.5, color='#3372b5', linestyle='-')
    plt.grid()
    plt.draw()
    plt.legend()
    plt.show()
