import matplotlib.pyplot as plt
import torch
from torch import nn
from crossValidation import Crossvalidation
from rnn import Net
from lstm import Lstm_Net
from prepareData import Data
import numpy as np
import time


def valid_classification(out, y):
    out = out.cpu().detach().numpy();
    y = y.cpu().detach().numpy();
    x = abs(out - y)
    valid = sum(i < 0.5 for i in x[0])
    percent = valid / x.shape[1] * 100
    print("%.2f" % percent, "%")
    return percent


if __name__ == "__main__":

    start = time.time()

    torch.cuda.empty_cache()

    # parkinson
    # data = Data("parkinson.csv")
    # data.toFloat()
    # data.Normalize(22)
    # cross_data = Crossvalidation(data.features, 4, 2, -1)

    # breast cancer
    data = Data("breast-cancer.csv")
    data.toNumbers()
    data.Normalize(0)
    cross_data = Crossvalidation(data.features, 5, 2, 0)

    # heart disease
    # data = Data("heart-disease.csv")
    # data.toFloat()
    # data.Normalize(13)
    # cross_data = Crossvalidation(data.features, 6, 2, -1)

    # diabetics
    # data = Data("diabetic_data.csv")
    # data.toNumbers()
    # cross_data = Crossvalidation(data.features, 10, 3, -1)

    # zoo
    # data = Data("zoo.txt")
    # data.features = data.features[0:100, :]
    # cross_data = Crossvalidation(data.features, 5, 7, -1)

    model = Lstm_Net(input_size=data.features.shape[1] - 1, output_size=1, hidden_size=22, n_layers=4)

    is_cuda = torch.cuda.is_available()
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        model = model.cuda()
        cross_data.data = cross_data.data.cuda()
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    cross_data.select_k()
    cross_data.input_output()
    n_epochs = 5000
    lr = 0.1

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    epochs_sh = int(n_epochs * 2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.99, total_steps=epochs_sh, epochs=epochs_sh, steps_per_epoch=1, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, last_epoch=-1)

    for epoch in range(1, n_epochs + 1):
        for index, data in enumerate(cross_data.training_inp):
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            output, hidden = model(data.float())
            loss = criterion(output.view(1, data.size(0)), cross_data.training_out[index].view(1, data.size(0)).float())
            loss.backward()  # Does backpropagation and calculates gradients
            optimizer.step()  # Updates the weights accordingly
            scheduler.step()

        if epoch % 200 == 0:

            print("lerarning rate:", optimizer.param_groups[0]['lr'])
            cross_data.select_k()
            cross_data.input_output()
            valid = valid_classification(output.view(1, data.size(0)), cross_data.training_out[index].view(1, data.size(0)).float())
            if (valid > 95):
                break
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))

    # model = torch.load("model")
    t_out, t_hidden = model(cross_data.test_inp.float())
    plt.plot(t_out.cpu().detach().numpy(), color='#4daf4a', marker='o', label="wyjscie sieci")
    plt.plot(cross_data.test_out.cpu().detach().numpy(), color='#e55964', marker='o', label="target")
    plt.draw()
    plt.pause(1e-17)
    plt.clf()
    plt.show()
    end = time.time()
    delta = end - start
    print("took %.2f seconds to process" % delta)
    # torch.save(model, 'model')
