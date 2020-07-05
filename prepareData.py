import numpy as np

"""
klasa wczytująca oraz przygotowująca wstępnie dane
"""


class Data:
    def __init__(self, file):
        """
        Dane są wczytywane oraz rzutowane na float, jeżeli sie to nie powiedzie,
         to dane zapisywane są bez mapowania do listy
        :param file: nazwa pliku z jakiego należy wczytać dane
        """
        with open(file) as f:
            try:
                self.features = [list(map(float, x.strip().split(',')[0:])) for x in f]
            except:
                self.features = [list(x.strip().split(',')[0:]) for x in f]

    def normalization(self, x, xmin, xmax):
        # normalizacja danych
        if (xmin == xmax):
            return x
        for idv, val in enumerate(x):
            x[idv] = (2 * (val - xmin)) / (xmax - xmin) - 1
            # x[idv] = (val - xmin) / (xmax - xmin)
        return x

    def Normalize(self, classId):
        """
        funkcja przeprowadzająca normalizację danych z wyjątkiem kolumny klasy
        :param classId: kolumna w której znajduje się klasa
        """
        features = []
        for (index, item) in enumerate(np.array(self.features).transpose()):
            if index != classId:
                temp = self.normalization(item, np.amin(item), np.amax(item))
            else:
                temp = item
            features.append(temp)
        self.features = np.array(features).transpose()

    def toFloat(self):
        # funkcja która mapuje wartości danych na Float
        features = []
        for (feId, d) in enumerate(np.array(self.features).transpose()):
            temp = [float(item) for item in d]
            features.append(temp)
        self.features = np.array(features).transpose()

    def toNumbers(self, dictID=None):
        """
        funkcja przekształcająca wartości wyrażone słownie na ich reprezentację liczbową
        :param dictID: tablica zwierająca numery kolumn dla których nie potrzeba
         przerprowadzać operacji zmiany na wartości liczbowe
        """
        # Tworzenie słownika, na podstawie którego zostaną przypisane odpowiednie wartości liczbowe
        # odpowiednim wyrażeniom
        if dictID is None:
            dictID = []
        features = []
        for (feId, d) in enumerate(np.array(self.features).transpose()):
            # tworzenie słownika z danej kolumny (poprzednio została przeprowadzona operacja transpozycji)
            s = list(dict.fromkeys(d).keys())
            temp = []
            if feId not in dictID:
                for x in d:
                    for (index, i) in enumerate(s):
                        if x == i:
                            temp.append(index + 1)
                temp = np.array(temp)
            else:
                temp = [float(item) for item in d]

            features.append(temp)
        self.features = np.array(features).transpose()
