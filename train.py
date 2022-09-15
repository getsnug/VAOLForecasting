import csv
import numpy
import network as n
class Trainer(object):
    def __init__(self, c, network):
        self.network = network
        self.data = []
        with open(c) as csv_file:
            reader = csv.reader(csv_file)
            self.data = list(reader)
        self.tDat = []
        for i in range(len(self.data)-5):
            x = [i,i+1,i+2,i+3,i+4]
            y = i+5
            self.tDat.append(tuple((x,y)))
    def train(self):
        self.network.SGD(self.tDat,10,100,.5)

nn = n.Network([5,10,10,10,1])
t = Trainer("dat.csv",nn)
t.train()
