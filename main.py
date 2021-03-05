import torch
import time
from training import training
from test import test
from data import getData

if __name__ == "__main__":
    # odmierzanie czasu nauki
    start = time.time()
    # czyszczenie cache'u z GPU
    torch.cuda.empty_cache()
    # zbiór danych choroby parkinsona - 'parkinson'
    # zbiór danych chorób serca - 'heart-disease-clevlend'
    # zbiór danych dotyczących raka piersi - 'breast-cancer'
    # zbiór danych dotyczących raka piersi wisconsin - 'breast-cancer-wisconsin'
    cv_data, data = getData(set='heart-disease-clevlend')
    # wybór rodzaju sieci do treningu
    # LSTM - 'lstm'
    # CNN - 'cnn'
    netType = 'cnn'
    training(cv_data, data, netType=netType,
             fileName="experiment1", softmax=True)

    # czas nauki przez sieć*+
    delta = time.time() - start
    print("took %.2f seconds to process" % delta)

    # testowanie
    # test(cv_data, netType=netType)
