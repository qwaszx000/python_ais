import pybrain
from pybrain.tools.shortcuts import*
from pybrain.datasets import *
from pybrain.supervised.trainers import *

global ep_g
ep_g = 0

nn = buildNetwork(2,9,1)#2 input,3 hiden,2 output

data_s = SupervisedDataSet(2,1)#2 inp 2 out

for x in range(11):#0-10
    for y in range(11):#0-10
        print("[ADD_SAMPLE] "+str((x,y))+" = "+str(x+y))
        data_s.addSample((x,y),(x+y))

#trainer = BackpropTrainer(nn,data_s)
#trainer.trainUntilConvergence()
#trainer.train()

trainer = RPropMinusTrainer(nn)
trainer.setData(data_s)
def train_n_log(ep):
    global ep_g
    ep_g += ep
    trainer.trainEpochs(ep)
    r = nn.activate((10,15))#input 2 and 1;
    print("NN output("+ str(ep_g)+" ep): "+str(r))
    return r

#trainer.trainEpochs(400)
#r = nn.activate((10,15))
#print("Out(500 ep): "+str(r))

while(train_n_log(100)!=[25.0]):
    train_n_log(100)

