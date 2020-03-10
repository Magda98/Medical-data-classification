
import numpy as np

class Data:
    def __init__(self, file, calssFirst=False):
        # load and prepare data
        with open(file) as f:
            try:
                self.features = [list( map(float, x.strip().split(',')[0:])) for x in f]
            except:
                self.features = [list( x.strip().split(',')[0:]) for x in f]


    def normalization(self, x, xmin, xmax):
        # normalizacja danych
        if(xmin == xmax):
            return x
        for idv, val in enumerate(x):
            x[idv] = (2 * (val - xmin)) / (xmax - xmin) - 1
            # x[idv] = (val - xmin) / (xmax - xmin)
        return x

    def Normalize(self, classId):
        features = []
        for (index, item) in enumerate(np.array(self.features).transpose()):
            if index != classId:
                temp = self.normalization(item, np.amin(item), np.amax(item))
            else:
                temp = item
            features.append(temp)
        self.features = np.array(features).transpose()

    def toFloat(self):
        features = []
        for (feId, d) in enumerate(np.array(self.features).transpose()):
            temp = [float(item) for item in d]
            features.append(temp)
        self.features = np.array(features).transpose()

    def toNumbers(self, dictID=None):
        if dictID is None:
            dictID = []
        features =[]
        for  (feId,d) in enumerate(np.array(self.features).transpose()):
            s = list(dict.fromkeys(d).keys())
            temp=[]
            if  feId not in dictID:
                for x in d:
                    for (index, i) in enumerate(s):
                        if x == i:
                            temp.append(index+1)
                temp = np.array(temp)
            else:
                temp = [float(item) for item in d]

            features.append(temp)
        self.features = np.array(features).transpose()