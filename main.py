import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import time
from crossValidation import Crossvalidation
from lstm import Lstm_Net
from prepareData import Data
from cnn import Cnn_Net


def valid_classification(out, y):
    out = out.cpu().detach().numpy();
    y = y.cpu().detach().numpy();
    x = abs(out - y)
    valid = sum(i < 0.5 for i in x[0])
    percent = valid / x.shape[1] * 100
    return percent


if __name__ == "__main__":

    start = time.time()

    torch.cuda.empty_cache()

    # parkinson
    # data = Data("parkinson.csv")
    # data.toFloat()
    # data.Normalize(22)
    # cross_data = Crossvalidation(data.features, 10, 2, -1)

    # breast cancer
    # data = Data("breast-cancer.csv")
    # data.toNumbers()
    # data.Normalize(0)
    # cross_data = Crossvalidation(data.features, 10, 2, 0)

    # heart disease
    data = Data("heart-disease.csv")
    data.toFloat()
    data.Normalize(13)
    cross_data = Crossvalidation(data.features, 10, 2, -1)

    # diabetics
    # data = Data("diabetic_data.csv")
    # data.toNumbers()
    # cross_data = Crossvalidation(data.features, 10, 3, -1)

    # zoo
    # data = Data("zoo.txt")
    # data.features = data.features[0:100, :]
    # cross_data = Crossvalidation(data.features, 10, 7, -1)

    model = Lstm_Net(input_size=data.features.shape[1] - 1, output_size=1, hidden_size=20, n_layers=2)

    # model = Cnn_Net(input_size=1, output_size=1, kernel_size=3, stride=1)

    is_cuda = torch.cuda.is_available()
    # If we GPU available, computation will run at GPU
    if is_cuda:
        model = model.cuda()
        cross_data.data = tuple(item.cuda() for item in cross_data.data)
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    cross_data.select_k()
    cross_data.input_output()
    n_epochs = 2000
    lr = 0.3

    er = 1.04
    lr_inc = 1.05
    lr_desc = 0.7
    old_loss = 0

    max_epoch = 20
    # print(list(model.named_parameters()))
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    epoch = 0
    # for epoch in range(1, n_epochs + 1):
    while 1:
        for data_out, data_in in zip(cross_data.training_out, cross_data.training_inp):
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            # lstm
            output, hidden = model(data_in.float())
            # cnn
            # output = model(data_in.float())
            # loss = criterion(output.view(1, data_in.size(0)), data_out.view(1, data_in.size(0)).float())
            # lstm
            t_out, t_hidden = model(cross_data.test_inp.float())
            # cnn
            # t_out = model(cross_data.test_inp.float())

            old_param = model.parameters

            loss = criterion(t_out.view(1, cross_data.test_inp.size(0)),
                             cross_data.test_out.view(1, cross_data.test_inp.size(0)).float())
            loss.backward()  # Does backpropagation and calculates gradients
            optimizer.step()  # Updates the weights accordingly
            # scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            if (loss > old_loss * er):
                model.parameters = old_param #get old weights and bias
                if (lr >= 0.0001):
                    lr = lr_desc * lr
            elif (loss < old_loss):
                lr = lr_inc * lr
                if (lr > 0.99):
                    lr = 0.99
            optimizer.param_groups[0]['lr'] = lr
            old_loss = loss
        epoch+=1
        valid = valid_classification(t_out.view(1, cross_data.test_out.size(0)),
                                     cross_data.test_out.view(1, cross_data.test_out.size(0)).float())
        if valid > 100 or epoch > max_epoch:
            cross_data.select_k()
            cross_data.input_output()
            print("learning rate:", optimizer.param_groups[0]['lr'])
            print('Epoch: {}.............'.format(epoch), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
            print("%.2f" % valid, "%")
            print(cross_data.k)
            epoch = 0
        if cross_data.k == 1 and cross_data.stop == 0:
            break
    # model = torch.load("model")
    # lstm
    print("value:", cross_data.k)
    cross_data.k =9
    cross_data.select_k()
    cross_data.input_output()
    t_out, t_hidden = model(cross_data.test_inp.float())
    # cnn
    # t_out = model(cross_data.test_inp.float())
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
