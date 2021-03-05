from collections import OrderedDict
import random
import torch
import numpy as np

"""
Klasa dzieląca dane na x podzbiorów w celu nauki sieci metodą kroswalidacji
Domyślnie klasa realizuje CV10
"""


class Crossvalidation:

    def __init__(self, data, chunks=10, classCol=-1):
        """
        W kontruktorze klasy najpierw obliczana jest procentowa ilość danej klasy w zbiorze danych,
         aby każda z częsci po podzieleniu zawierała odpowiednią część przypadków danej klasy
        :param data: zbiór danych
        :param chunks: liczba określająca na ile części należy podzielić dane
        :param classCol: kolumna w której znajduje się informacja o klasie
        """

        # flaga stop, jest ustawiana na 0 w przypadku gdy zakończony zostanie trening sieci
        # - wszystkie części ze zbioru zostaną użyte jako testowe
        self.stop = 1
        self.data = data
        self.classCol = classCol
        # losowe wymieszanie danych - tylko wiersze
        np.random.shuffle(self.data)
        # utworzenie słownika na podstawie klas
        class_quantity = dict.fromkeys(data[:, classCol], 0)
        self.classes = len(class_quantity)
        # obliczanie ilośći każdej z klas w zbiorze danych
        for x in self.data:
            class_quantity[int(x[classCol])] += 1

        # obliczanie ilośći przykładów danej klasy w każdym z podzbiorów
        quantity_chunk = dict.fromkeys(data[:, classCol], 0)
        quantity_chunk = OrderedDict(
            sorted(quantity_chunk.items(), key=lambda t: t[0]))
        for key in class_quantity:
            quantity_chunk[key] = (
                round(class_quantity[key]/len(data) * len(data)/chunks))
        # podział danych na podzbiory z odpowiednią liczbą przykładów danej klasy w każdym
        # oraz następne wymieszanie danych w podzbiorach - tylko wiersze
        sorted_data = []
        for k in range(chunks):
            organized_data = []
            tab = []
            for (index, i) in enumerate(quantity_chunk):
                q = 0
                for pos, d in enumerate(self.data):
                    if d[self.classCol] == i:
                        q += 1
                        organized_data.append(d)
                        tab.append(pos)
                    if q == quantity_chunk[i]:
                        break
            self.data = np.delete(self.data, tab, 0)
            np.random.shuffle(np.array(organized_data))
            sorted_data.extend(np.array(organized_data))
        if self.data.size:
            np.random.shuffle(self.data)
            sorted_data.extend(self.data)
        sorted_data = np.array(sorted_data)

        self.data = sorted_data

        # torch.stack(torch.chunk(torch.from_numpy(self.data), chunks))
        # przekształcenie tablic biblioteki NumPy na tensory w PyTroch
        self.data = torch.chunk(torch.from_numpy(self.data), chunks)
        # ustawienie zmiennej klasy k na 0, mówi ona który z podzbiorów jest aktualnie traktowany jako testowy
        self.k = 0
        # wymieszanie podzbiorów
        self.data = list(self.data)
        random.shuffle(self.data)
        self.data = tuple(self.data)

    def select_k(self):
        """
        funkcja dzieli odpowiednio dane na testowe i treningowe zgodnie z algorytmem kroswalidacji,
        w przypadku zakończenia atrybut klasy stop ma przypisywaną wartość 0
        """
        self.test_data = self.data[self.k]
        self.test_data = self.test_data.cpu().numpy()
        self.test_data = self.test_data[np.argsort(
            self.test_data[:, self.classCol])]
        self.test_data = torch.from_numpy(self.test_data)
        # self.training_data = torch.cat([self.data[:self.k], self.data[self.k+1:]], 0)
        self.training_data = [x for index,
                              x in enumerate(self.data) if index != self.k]
        self.training_data = torch.cat(self.training_data, 0)
        if self.k < (len(self.data) - 1):
            self.k += 1
        else:
            self.stop = 0
            self.k = 0

    def input_output(self):
        """
        funkcja dzieli dane na dane wejściowe i dane wyjściowe odpowiednio dla treningu oraz testu sieci.
        dane są dzielone w zależności od kolumny w której znajduje się klasa
        """
        if self.classCol == 0:
            self.training_inp = self.training_data[:, 1:self.data[0].shape[1]]
            self.training_out = self.training_data[:, 0]

            self.test_inp = self.test_data[:, 1:self.data[0].shape[1]]
            self.test_out = self.test_data[:, 0]

        elif self.classCol == -1:
            self.training_inp = self.training_data[:,
                                                   0:self.data[0].shape[1]-1]
            self.training_out = self.training_data[:, self.data[0].shape[1]-1]

            self.test_inp = self.test_data[:, 0:self.data[0].shape[1]-1]
            self.test_out = self.test_data[:, self.data[0].shape[1]-1]
