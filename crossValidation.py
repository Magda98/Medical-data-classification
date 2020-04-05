from collections import OrderedDict

import torch
import numpy as np

class Crossvalidation:

    def __init__(self, data, chunks, quantity, classCol = -1):
        self.stop = 1
        self.data = data
        self.classCol = classCol
        #1 shuffle only rows
        np.random.shuffle(self.data)
        #2
        class_quantity = dict.fromkeys(data[:, classCol], 0)
        #calculate the quantity of class reprezentation for every chunk
        for x in self.data:
            class_quantity[ int(x[classCol])] += 1

        quantity_chunk = dict.fromkeys(data[:, classCol], 0)
        quantity_chunk = OrderedDict(sorted(quantity_chunk.items(), key=lambda t: t[0]))
        for key in class_quantity:
            quantity_chunk[key]=(round(class_quantity[key]/len(data) * len(data)/chunks))
        #organize data that every chunk have every class reprezentation
        sorted_data = []
        for k in range(chunks):
            organized_data = []
            tab=[]
            for (index, i) in enumerate(quantity_chunk):
                q = 0
                for pos, d in enumerate(self.data):
                    if d[self.classCol] ==  i:
                        q+=1
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
        self.data = torch.chunk(torch.from_numpy(self.data), chunks)
        self.k = 0
        self.chunk_size = data.shape[0]/chunks
        # self.to_reshape = int(self.data.shape[0] * self.data.shape[1] - self.chunk_size)

    def select_k(self):
        self.test_data = self.data[self.k]
        # self.training_data = torch.cat([self.data[:self.k], self.data[self.k+1:]], 0)
        self.training_data = [x for index,x in enumerate(self.data) if index!=self.k]
        self.training_data = torch.cat(self.training_data, 0)
        if self.k < (len(self.data) -1):
            self.k+=1
        else:
            self.stop = 0
            self.k=0
    def input_output(self):

        # self.training_data = tuple(np.random.shuffle(item) for item in  self.training_data)
        # self.training_data = self.training_data.view(self.to_reshape, self.data.shape[2])
        if self.classCol == 0:
            # self.training_inp = self.training_data[ : ,: , (self.classCol+1):10]
            # self.training_out = self.training_data[ :, :,self.classCol]
            self.training_inp = tuple(item[:, (self.classCol+1):self.data[0].shape[1]] for item in self.training_data)
            self.training_out = tuple(item[:, self.classCol] for item in self.training_data)

            self.test_inp = self.test_data[:, (self.classCol+1):self.data[0].shape[1]]
            self.test_out = self.test_data[:, self.classCol]
        elif self.classCol == -1:
            self.training_inp = self.training_data[ :, 0:self.data[0].shape[1]-1]
            self.training_out = self.training_data[ :, self.data[0].shape[1]-1]

            # self.training_inp = tuple(item[:, 0:self.data[0].shape[1]-1] for item in self.training_data )
            # self.training_out = tuple(item[:, self.data[0].shape[1]-1] for item in self.training_data)

            self.test_inp = self.test_data[:, 0:self.data[0].shape[1]-1]
            self.test_out = self.test_data[:, self.data[0].shape[1]-1]
