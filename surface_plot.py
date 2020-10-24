import statistics

from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import itertools

from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import axes3d
import scipy.linalg


def polyfit2d(x, y, z, order=3):
    z = np.array(list(map(int, z)))
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z, rcond=None)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x, dtype='float64')
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z


if __name__ == "__main__":
    with open('parkinson_test_lstm.csv') as f:
        X, Y, data_3 = [], [], []
        set_y, size = set(), 0

        for line in f:
            x, y, z = [float(s) for s in line.split(';')]
            set_y.add(y)
            X.append(x)
            Y.append(y)
            data_3.append(z)
        size = len(set_y)

        X = np.array(X)
        Y = np.array(Y)
        # data_3 = np.array(data_3).reshape((size,-1))
        data_3 = np.array(data_3)

        data_1 = X.reshape((-1,size))
        data_2 = Y.reshape((-1, size))


    data = np.c_[X,Y,data_3]
    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    order = 3
    if order == 1:
        # best-fit linear plane
        A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
        C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
        
        # evaluate it on grid
        # Z = C[0]*X + C[1]*Y + C[2]
        # or expressed using matrix/vector product
        Z = np.dot(np.c_[X, Y, np.ones(X.shape)], C).reshape(data_1.shape)

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
        C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
        Z = np.dot(np.c_[np.ones(X.shape), X, Y, X*Y, X**2, Y**2], C).reshape(data_1.shape)
    elif order == 3:
        Z = polyval2d(data_1, data_2, polyfit2d(data_1.flatten(), data_2.flatten(), data_3))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # plt.xlabel('F1')
    # plt.ylabel('F2')
    # plt.xlabel('liczba filtrów conv 1.')
    # plt.ylabel('liczba filtrów conv 2.')
    plt.xlabel('liczba warstw')
    plt.ylabel('liczba neuronów')
    ax.set_zlabel('PK[%]')
    ax.plot_surface(data_1, data_2, Z, cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=2, antialiased=True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.scatter(X, Y, data_3, c='r', s=50)
    plt.show()

    loss=[]
    with open('cnn_loss.csv') as f:
        for line in f:
            x = [float(s) for s in line.split(';')]
            loss.append(x[0])
    ma_loss = []
    loss = np.asarray(loss)
    loss = np.array_split(loss, 20)
    for index, x in enumerate(loss):
        ma_loss.append(statistics.mean(x))
    y_t = np.arange(round(min(ma_loss),0)+1, round(max(ma_loss),0), step=1)
    y_t = np.insert(y_t, 0,round(min(ma_loss),2))
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.yticks(y_t)
    plt.xlabel('kolejne etapy uczenia')
    plt.ylabel('wartość błędu')
    plt.plot(ma_loss, color='#008080', label="loss")
    plt.grid()
    plt.draw()
    plt.legend()
    plt.show()

    test_1 = []
    test_2 = []
    with open('pk_heart-disease-clevlend_softmax.csv') as f:
        for line in f:
            x = [float(s) for s in line.split(';')]
            test_1.append(x[0])
    with open('pk_heart-disease-clevlend.csv') as f:
        for line in f:
            x = [float(s) for s in line.split(';')]
            test_2.append(x[0])
    ax = plt.figure().gca()
    x = ['warstwa softmax', 'neuron o liniowej \n funkcji aktywacji']
    print(statistics.mean(test_1))
    energy = [statistics.mean(test_1), statistics.mean(test_2)]
    std = [np.std(test_1), np.std(test_2)]
    print(statistics.mean(test_2))
    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, energy, color='teal', yerr=std, capsize=10, alpha=0.5, width=0.4)
    plt.xlabel("Sieć w zależności od typu wyjścia")
    plt.ylabel("Poprawność klasyfikacji [%]")
    plt.tight_layout()
    ax.yaxis.grid(True)
    plt.xticks(x_pos, x)

    plt.show()
    # n_groups = 4
    # means_frank = (90, 55, 40, 65)
    # means_guido = (85, 62, 54, 20)
    # # create plot
    # fig, ax = plt.subplots()
    # index = np.arange(n_groups)
    # bar_width = 0.35
    # opacity = 0.8
    #
    # rects1 = plt.bar(index, means_frank, bar_width,
    #                  alpha=opacity,
    #                  color='b',
    #                  label='Frank')
    #
    # rects2 = plt.bar(index + bar_width, means_guido, bar_width,
    #                  alpha=opacity,
    #                  color='g',
    #                  label='Guido')
    #
    # plt.xlabel('Person')
    # plt.ylabel('Scores')
    # plt.title('Scores by person')
    # plt.xticks(index + bar_width, ('A', 'B', 'C', 'D'))
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()
   
