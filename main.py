import torch
import time
from training import training
from test import test
from data import getData

if __name__ == "__main__":
    start = time.time()
    torch.cuda.empty_cache()

    # parkinson data set - 'parkinson'
    # heart disease data set - 'heart-disease'
    # breast cancer data set - 'breast-cancer'
    cv_data, data = getData(set='heart-disease')

    # LSTM - 'lstm'
    # CNN - 'cnn'
    netType = 'cnn'
    # training(cv_data, data, netType=netType)

    # learning time
    delta = time.time() - start
    print("took %.2f seconds to process" % delta)

    # testing model
    test(cv_data, netType=netType)
