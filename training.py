import numpy as np
import matplotlib.pyplot as plt
import statistics
import torch
from valid import valid_classification
from lstm import lstmNet
from cnn import cnnNet


def training(cross_data, data, netType):
    PK = []
    s1 = np.arange(64, 65, 1)
    s2 = np.arange(16, 17, 1)
    for i in s1:
        for j in s2:
            if netType == 'lstm':
                model = lstmNet(input_size=data.features.shape[1] - 1, output_size=1, hidden_size=int(j),
                                n_layers=int(i))
            elif netType == 'cnn':
                model = cnnNet(input_ch=[1, i], output_ch=[i, j], kernel_size=[2, 2], stride=[1, 1], pool=[1, 1],
                               features=cross_data.data[0].shape[1] - 1)

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
            max_epochs = 500
            lr = 0.05

            er = 1.04
            lr_inc = 1.05
            lr_desc = 0.7
            old_loss = 0

            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            criterion = torch.nn.MSELoss()
            stop = 0.8
            epoch = 0
            cv = []
            cv_mse = []

            sse = []
            output_train = []
            while 1:
                for data_out, data_in in zip(cross_data.training_out, cross_data.training_inp):
                    optimizer.zero_grad()  # Clears existing gradients from previous epoch
                    if netType == 'lstm':
                        output, hidden = model(data_in.float())
                    elif netType == 'cnn':
                        output = model(data_in.float())

                    output_train.append(output)
                    loss = (0.5 * ((data_out.view(1) - output.view(1)) ** 2)).squeeze(0)
                    old_param = model.parameters
                    loss.backward()  # backpropagation
                    optimizer.step()  # Update weights
                    sse.append(loss.item())

                sse_loss = sum(sse)
                sse = []

                # Adaptive learning rate
                lr = optimizer.param_groups[0]['lr']
                if sse_loss > old_loss * er:
                    # get old weights and bias
                    model.parameters = old_param
                    if lr >= 0.0001:
                        lr = lr_desc * lr
                elif sse_loss < old_loss:
                    lr = lr_inc * lr
                    if lr > 0.99:
                        lr = 0.99
                optimizer.param_groups[0]['lr'] = lr
                old_loss = sse_loss
                ############################

                output_train = torch.tensor(output_train)
                valid = valid_classification(output_train.view(1, cross_data.training_out.size(0)),
                                             cross_data.training_out.view(1, cross_data.training_out.size(0)).float())
                output_train = []
                print("learning rate:", optimizer.param_groups[0]['lr'])
                print('Epoch: {}.............'.format(epoch), end=' ')
                print("Loss: {:.4f}".format(sse_loss))
                print("%.2f" % valid, "%")
                epoch += 1

                if sse_loss <= stop or epoch > max_epochs:

                    # Testing model
                    with torch.no_grad():
                        if netType == 'lstm':
                            t_out, t_hidden = model(cross_data.test_inp.float())
                        elif netType == 'cnn':
                            t_out = model(cross_data.test_inp.float(), True)

                        loss_test = (0.5 * criterion(t_out.view(1, cross_data.test_inp.size(0)),
                                                     cross_data.test_out.view(1, cross_data.test_inp.size(
                                                         0)).float())).sum()
                        cv_mse.append(loss_test.item())
                        cv.append(valid_classification(t_out.view(1, cross_data.test_out.size(0)),
                                                       cross_data.test_out.view(1,
                                                                                cross_data.test_out.size(0)).float()))
                    if cross_data.stop == 0:
                        break
                    cross_data.select_k()
                    cross_data.input_output()
                    print(cross_data.k)
                    epoch = 0
            PK.append([i, j, statistics.mean(cv)])
            cross_data.stop = 1
            cross_data.k = 0

    PK = np.asarray(PK)
    np.savetxt("test.csv", PK, delimiter=";")

    if netType == 'lstm':
        t_out, t_hidden = model(cross_data.test_inp.float())
    if netType == 'cnn':
        t_out = model(cross_data.test_inp.float(), True)


    # saving model
    torch.save(model, 'model')

    #cv10 mse and pk
    print(statistics.mean(cv_mse))
    print(statistics.mean(cv))

    # Plot
    plt.plot(t_out.cpu().detach().numpy(), color='#4daf4a', marker='o', label="wyjscie sieci")
    plt.plot(cross_data.test_out.cpu().detach().numpy(), color='#e55964', marker='o', label="target")
    plt.axhline(y=0.5, color='#3372b5', linestyle='-')
    plt.grid()
    plt.draw()
    plt.legend()
    plt.show()
