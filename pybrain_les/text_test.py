import pybrain
from pybrain.tools.shortcuts import *
from pybrain.datasets import *
from pybrain.supervised.trainers import *

nn = buildNetwork(6,10,6)

data_s = SupervisedDataSet(6,6)
data_s.addSample([ord(c) for c in "Привет"],[ord(c) for c in "Привет"])
data_s.addSample([ord(c) for c in "Абажур"],[ord(c) for c in "Абажур"])
data_s.addSample([ord(c) for c in "Абсурд"],[ord(c) for c in "Абсурд"])
data_s.addSample([ord(c) for c in "Абсент"],[ord(c) for c in "Абсент"])
trainer = BackpropTrainer(nn, data_s)
#trainer.setData(data_s)
trainer.trainUntilConvergence(validationProportion=0.75)

n = nn.activate([ord(c) for c in "Абсент"])
print(n)
print(''.join(chr(int(r)) for r in n))

#print([ord(r) for r in "Привет"])

n = nn.activate([ord(c) for c in "Привет"])
print(n)
print(''.join(chr(int(r)) for r in n))

n = nn.activate([ord(c) for c in "Абажур"])
print(n)
print(''.join(chr(int(r)) for r in n))

n = nn.activate([ord(c) for c in "Абсурд"])
print(n)
print(''.join(chr(int(r)) for r in n))

