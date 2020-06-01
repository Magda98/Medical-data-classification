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
    data = Data("parkinson.csv")
    data.toFloat()
    data.Normalize(22)
    cross_data = Crossvalidation(data.features, 10, -1)

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
    # data = Data("zoo.txt")
    # data.features = data.features[0:100, :]
    # cross_data = Crossvalidation(data.features, 10, 7, -1)

    model = Lstm_Net(input_size=data.features.shape[1] - 1, output_size=1, hidden_size=10, n_layers=2)

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
    n_epochs = 10000
    lr = 0.5

    er = 1.04
    lr_inc = 1.05
    lr_desc = 0.7
    old_loss = 0

    max_epoch = 5000
    # print(list(model.named_parameters()))
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    stop = 0
    epoch = 0
    sse=[]
    output_train=[]
    # for epoch in range(1, n_epochs + 1):
    while 1:
        for data_out, data_in in zip(cross_data.training_out, cross_data.training_inp):
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            # lstm
            output, hidden = model(data_in.float())
            output_train.append(output.item())
            # cnn
            # output = model(data_in.float())
            loss = criterion(output.view(1), data_out.view(1).float())
            optimizer.zero_grad()
            old_param = model.parameters
            loss.backward()  # Does backpropagation and calculates gradients
            optimizer.step()  # Updates the weights accordingly
            # sse.append(0.5 * (output.item() - data_out.float().item()) ** 2)
            # lstm
            epoch+=1
            with torch.no_grad():
                t_out, t_hidden = model(cross_data.test_inp.float())
                loss_test = criterion(t_out.view(1, cross_data.test_inp.size(0)),cross_data.test_out.view(1, cross_data.test_inp.size(0)).float()).sum()
                sse.append(loss_test.item())
        # cnn
        # t_out = model(cross_data.test_inp.float())
        # scheduler.step()
            sse_loss = loss_test.item()
            sse = []
            lr = optimizer.param_groups[0]['lr']
            if (sse_loss > old_loss * er):
                model.parameters = old_param #get old weights and bias
                if (lr >= 0.0001):
                    lr = lr_desc * lr
            elif (sse_loss < old_loss):
                lr = lr_inc * lr
                if (lr > 0.99):
                    lr = 0.99
            optimizer.param_groups[0]['lr'] = lr
            old_loss = sse_loss
        valid = valid_classification(t_out.view(1, cross_data.test_out.size(0)),
                                     cross_data.test_out.view(1, cross_data.test_out.size(0)).float())
        output_train = np.array(output_train)
        output_train = torch.from_numpy(output_train)
        # valid = valid_classification(output_train.view(1, cross_data.training_out.size(0)),
        #                              cross_data.training_out.view(1, cross_data.training_out.size(0)).float())
        output_train = []
        print("learning rate:", optimizer.param_groups[0]['lr'])
        print('Epoch: {}.............'.format(epoch), end=' ')
        print("Loss: {:.4f}".format(sse_loss))
        print("%.2f" % valid, "%")
        if valid >= 95 or epoch > max_epoch:
            cross_data.select_k()
            cross_data.input_output()
            print("learning rate:", optimizer.param_groups[0]['lr'])
            print('Epoch: {}.............'.format(epoch), end=' ')
            print("Loss: {:.4f}".format(sse_loss))
            print("%.2f" % valid, "%")
            print(cross_data.k)
            epoch = 0
        if cross_data.k == 1 and cross_data.stop == 0:
            stop = 1
            break
        if stop == 1:
            break
    # model = torch.load("model")
    # lstm
    print("value:", cross_data.k)
    cross_data.k = 0
    cross_data.select_k()
    cross_data.input_output()
    t_out, t_hidden = model(cross_data.test_inp.float())
    # cnn
    # t_out = model(cross_data.test_inp.float())
    plt.plot(t_out.cpu().detach().numpy(), color='#4daf4a', marker='o', label="wyjscie sieci")
    plt.plot(cross_data.test_out.cpu().detach().numpy(), color='#e55964', marker='o', label="target")
    plt.draw()
    plt.legend()
    plt.pause(1e-17)
    plt.clf()
    plt.show()
    end = time.time()
    delta = end - start
    print("took %.2f seconds to process" % delta)
    torch.save(model, 'model')
