from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import itertools
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
    with open('parkinson_test-n.csv') as f:
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
    plt.xlabel('ilość filtrów conv1')
    plt.ylabel('ilość filtrów conv2')
    ax.set_zlabel('PK[%]')
    ax.plot_surface(data_1, data_2, Z, cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=2, antialiased=True)
    # ax.scatter(X, Y, data_3, c='r', s=50)
    plt.show()
    # random = []
    # cost = []
    # with open('C:\\Users\\Magda\\Desktop\\random50_15.csv') as f:
    #     random = [float(line.rstrip('\n')) for line in f]
    # with open('C:\\Users\\Magda\\Desktop\\error.csv') as f:
    #     x = [float(line.rstrip('\n').split(';')[0]) for line in f]
    #     f.seek(0)
    #     y = [float(line.rstrip('\n').split(';')[1]) for line in f]
    
    # # handles, labels = ax.get_legend_handles_labels()
    # # ax.legend(handles, labels)
    # plt.plot(x, y,color = '#4daf4a' , linewidth=2.0, marker = "o",  label='')
    # plt.ylabel("PK[%]")
    # plt.xlabel("error ratio")
    # plt.ylim(5, 95)
    # plt.grid()
    # plt.show()
    # plt.plot(cost,color= '#e55964', linewidth=2.0, label='algorytm NW')
    # plt.legend(handles=[ra, nw])
    # plt.legend([ra, nw], ['losowe wagi', 'algorytm NW'])
    # plt.ylabel('koszt')
    # plt.xlabel('epoka')
    # plt.legend(loc='upper right')
    # ax.axis('equal')
    # ax.axis('tight')
    # plt.show()
    # with open('test.csv', 'w') as f:
    #     for x, y, z in zip(X.flatten(), Y.flatten(), Z):
    #         f.write(f'{x};{y};{z}\n')
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=2, antialiased=True)
    # plt.xlabel('S1')
    # plt.ylabel('S2')
    # ax.set_zlabel('PK')
    # ax.axis('equal')
    # ax.axis('tight')

   
