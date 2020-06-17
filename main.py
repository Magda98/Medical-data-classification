import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import time
from crossValidation import Crossvalidation
from lstm import lstmNet
from prepareData import Data
from cnn import cnnNet
import statistics


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

    # # parkinson
    # data = Data("parkinson.csv")
    # data.toFloat()
    # data.Normalize(22)
    # cross_data = Crossvalidation(data.features, 10, -1)

    # breast cancer
    # data = Data("breast-cancer.csv")
    # data.toNumbers()
    # data.Normalize(0)
    # cross_data = Crossvalidation(data.features, 10, 0)


    # heart disease
    # data = Data("heart-disease.csv")
    # data.toFloat()
    # data.Normalize(13)
    # cross_data = Crossvalidation(data.features, 10, -1)

    # diabetics
    # data = Data("diabetic_data.csv")
    # data.toNumbers()
    # # data.Normalize(46)
    # cross_data = Crossvalidation(data.features, 10, 3)

    # zoo
    data = Data("zoo.csv")
    data.toNumbers()
    data.Normalize(16)
    cross_data = Crossvalidation(data.features, 10, -1)

    # model = Lstm_Net(input_size=data.features.shape[1] - 1, output_size=1, hidden_size=10, n_layers=2)

    model = cnnNet(input_ch=[1, 10], output_ch=[10, 5], kernel_size=[2, 2], stride=[1, 1], pool=[1, 1], features=cross_data.data[0].shape[1] - 1)
    is_cuda = torch.cuda.is_available()
    # If GPU available - run at GPU
    if is_cuda:
        model = model.cuda()
        cross_data.data = tuple(item.cuda() for item in cross_data.data)
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    cross_data.select_k()
    cross_data.input_output()
    n_epochs = 10000
    lr = 0.05

    er = 1.04
    lr_inc = 1.05
    lr_desc = 0.7
    old_loss = 0

    max_epoch = 5000
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    stop = 0.3
    epoch = 0
    cv = []
    cv_mse =[]

    mse = []
    output_train = []
    while 1:
        for data_out, data_in in zip(cross_data.training_out, cross_data.training_inp):
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            # lstm
            # output, hidden = model(data_in.float())
            # cnn
            output = model(data_in.float())
            output_train.append(output)
            loss = criterion(output.view(1), data_out.view(1).float())
            optimizer.zero_grad()
            old_param = model.parameters
            loss.backward()  # backpropagation
            optimizer.step()  # Update weights
            epoch+=1
            mse.append(loss.item())

        mse_loss = statistics.mean(mse)
        mse = []
        # Adaptive learning rate
        lr = optimizer.param_groups[0]['lr']
        if mse_loss > old_loss * er:
            # get old weights and bias
            model.parameters = old_param
            if lr >= 0.0001:
                lr = lr_desc * lr
        elif mse_loss < old_loss:
            lr = lr_inc * lr
            if lr > 0.99:
                lr = 0.99
        optimizer.param_groups[0]['lr'] = lr
        old_loss = mse_loss
        ############################
        output_train = torch.tensor(output_train)
        valid = valid_classification(output_train.view(1, cross_data.training_out.size(0)),
                                     cross_data.training_out.view(1, cross_data.training_out.size(0)).float())
        output_train = []
        print("learning rate:", optimizer.param_groups[0]['lr'])
        print('Epoch: {}.............'.format(epoch), end=' ')
        print("Loss: {:.4f}".format(mse_loss))
        print("%.2f" % valid, "%")
        if mse_loss <= stop:
            # Testing model
            with torch.no_grad():
                t_out = model(cross_data.test_inp.float(), True)
                loss_test = criterion(t_out.view(1, cross_data.test_inp.size(0)),
                cross_data.test_out.view(1, cross_data.test_inp.size(0)).float()).sum()
                cv_mse.append(loss_test.item())
                cv.append(valid_classification(t_out.view(1, cross_data.test_out.size(0)),
                                             cross_data.test_out.view(1, cross_data.test_out.size(0)).float()))
            if cross_data.stop == 0:
                break
            cross_data.select_k()
            cross_data.input_output()
            print(cross_data.k)
            epoch = 0
    # loading model
    # model = torch.load("model")

    print("value:", cross_data.k)
    # lstm
    # t_out, t_hidden = model(cross_data.test_inp.float())
    # cnn
    t_out = model(cross_data.test_inp.float(), True)

    print(statistics.mean(cv_mse))
    print(statistics.mean(cv))
    # Plot
    plt.plot(t_out.cpu().detach().numpy(), color='#4daf4a', marker='o', label="wyjscie sieci")
    plt.plot(cross_data.test_out.cpu().detach().numpy(), color='#e55964', marker='o', label="target")
    plt.axhline(y=0.5, color='#3372b5', linestyle='-')
    # plt.yticks(np.arange(-0.2, 1.4, 0.1))
    plt.grid()
    plt.draw()
    plt.legend()
    plt.pause(1e-17)
    plt.clf()
    # plt.figure()
    # plt.plot(cv_mse)
    # plt.show()

    # saving model
    torch.save(model, 'model')

    # learning time
    end = time.time()
    delta = end - start
    print("took %.2f seconds to process" % delta)