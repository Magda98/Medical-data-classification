import numpy as np
import matplotlib.pyplot as plt
import statistics
import torch
from valid import valid_classification
from lstm import lstmNet
from cnn import cnnNet


def training(cross_data, data, netType):
    """
    funkcja przeprowadzająca trening sieci
    :param cross_data: obiekt przechowujący zbiór danych przygotowanych do kroswalidacji
    :param data: obiekt przechowujący dane
    :param netType: typ sieci - 'lstm' lub 'cnn'
    """
    PK = []
    # pętle do wykonywania eksperymwntów - poszukiwanie optymalnych parametów sieci
    s1 = np.arange(1, 102, 10)
    s2 = np.arange(1, 102, 10)
    for i in s1:
        for j in s2:
            # deklaracje odpowiedniego modelu sieci w zależności od parametru netType przekazanego do funkcji
            if netType == 'lstm':
                model = lstmNet(input_size=data.features.shape[1] - 1, output_size=1, hidden_size=int(j),
                                n_layers=int(i))
            elif netType == 'cnn':
                model = cnnNet(input_ch=[1, i], output_ch=[i, j], kernel_size=[2, 2], stride=[1, 1], pool=[1, 1],
                               features=cross_data.data[0].shape[1] - 1)

            # Jeżeli jest dostępny GPU obliczenia będą wykonywane na GPU, w innym wypadku na CPU
            # Obliczenia wykonywane na GPU znacznie przyśiepszają czas nauki sieci
            is_cuda = torch.cuda.is_available()
            if is_cuda:
                model = model.cuda()
                cross_data.data = tuple(item.cuda() for item in cross_data.data)
                print("GPU is available")
            else:
                device = torch.device("cpu")
                print("GPU not available, CPU used")

            # Przygotowanie danych do nauki
            cross_data.select_k()
            cross_data.input_output()

            # maksymalna liczba epok
            max_epochs = 500
            # warunek stopu, jeżeli wartość błędu sieci będzie mniejsza niż 0.8 nauka jest przerywana
            stop = 0.8

            # współczynnik uczenia
            lr = 0.05

            # Adaptive Learing Rate:
            # error ratio - wspołczynnik błędu,
            # wspołczynnik inkrementacji
            # wspołczynnik dekrementacji
            er = 1.04
            lr_inc = 1.05
            lr_desc = 0.7
            # zmienna przechowująca starą wartość błędu sieci
            old_loss = 0

            # inicializacja metody optymaliacji SGD (Stochastic gradient descent)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            # Funkcja celu
            criterion = torch.nn.MSELoss()

            epoch = 0

            cv = [] # lista przechowująca poprawność klasyfikacji dla danej częsci podzbiorów testowych
            cv_mse = [] # lista przechowująca błąd sieci dla danej częsci podzbiorów testowych

            sse = [] # lista przechowująca wartości błędu sieci podczas nauki
            output_train = [] # lista przechowująca wyjście sieci podczas nauki,
            # w celu sprawdzenia poprawności klasyfikacji dla zbioru uczącego
            while 1:
                for data_out, data_in in zip(cross_data.training_out, cross_data.training_inp):
                    optimizer.zero_grad()  # Wyczyszczenie gradientów z poprzedniej epoki

                    # przejście sieci w przód
                    if netType == 'lstm':
                        output, hidden = model(data_in.float())
                    elif netType == 'cnn':
                        output = model(data_in.float())

                    output_train.append(output)
                    loss = (0.5 * ((data_out.view(1) - output.view(1)) ** 2)).squeeze(0)
                    # zapisanie poprzednich parametrów sieci
                    old_param = model.parameters
                    loss.backward()  # wstecznapropagacja
                    optimizer.step()  # aktualizacja współczynników wagowych
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

                # obliczanie poprawności klasyfikacji dla danych treningowych
                output_train = torch.tensor(output_train)
                valid = valid_classification(output_train.view(1, cross_data.training_out.size(0)),
                                             cross_data.training_out.view(1, cross_data.training_out.size(0)).float())
                output_train = []

                # wypisywanie do konsoli wartości wspołczynnnika uczenia,
                # epoki, błędu sieci oraz poprawności klasyfikacji
                print("learning rate:", optimizer.param_groups[0]['lr'])
                print('Epoch: {}.............'.format(epoch), end=' ')
                print("Loss: {:.4f}".format(sse_loss))
                print("%.2f" % valid, "%")
                epoch += 1

                # sprawdzenaie warunku stopu
                if sse_loss <= stop or epoch > max_epochs:
                    # Przeprowadzenie testów na danych testowych
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
                    # jeżeli parametr stop obiektu cross_data będzie równy 0 oznacza to że należy zakończyć
                    # naukę sieci algorytmem kroswalidacji
                    if cross_data.stop == 0:
                        break

                    cross_data.select_k()
                    cross_data.input_output()
                    print(cross_data.k)
                    epoch = 0
            PK.append([i, j, statistics.mean(cv)])
            cross_data.stop = 1
            cross_data.k = 0

    # zapisywanie wyników eksperymentów do pliku csv
    PK = np.asarray(PK)
    np.savetxt("test.csv", PK, delimiter=";")

    if netType == 'lstm':
        t_out, t_hidden = model(cross_data.test_inp.float())
    if netType == 'cnn':
        t_out = model(cross_data.test_inp.float(), True)

    # zapisywanie nauczonego modelu sieci
    torch.save(model, 'model')

    # błąd kroswalidacji oraz poprawność klasyfikacji
    print(statistics.mean(cv_mse))
    print(statistics.mean(cv))

    # Wykres przedstawiający target oraz wyjście sieci dla danych testowych
    plt.plot(t_out.cpu().detach().numpy(), color='#4daf4a', marker='o', label="wyjscie sieci")
    plt.plot(cross_data.test_out.cpu().detach().numpy(), color='#e55964', marker='o', label="target")
    plt.axhline(y=0.5, color='#3372b5', linestyle='-')
    plt.grid()
    plt.draw()
    plt.legend()
    plt.show()
