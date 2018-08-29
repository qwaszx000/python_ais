#----------------------------------------
#Расчет дульной енергии
#   E = (mv^2)/2
#----------------------------------------
import pybrain
from pybrain.tools.shortcuts import*
from pybrain.datasets import *
from pybrain.supervised.trainers import *

ai = buildNetwork(2,50,1)

ds = SupervisedDataSet(2,1)
#---------------------------
#M V Out
ds.addSample((0.5,5000),(6250))
ds.addSample((0.5,100),(2.5))
ds.addSample((0.5,1000),(250))
ds.addSample((2,100),(10))
ds.addSample((2,1000),(1000))
ds.addSample((2,500),(250))
ds.addSample((2,5000),(25000))
ds.addSample((0.75,1000),(375))
ds.addSample((3000,1000),(1500000))
ds.addSample((0.60,1000),(300))
ds.addSample((0.55,1000),(275))
ds.addSample((0.90,1000),(450))
ds.addSample((1,1000),(500))
ds.addSample((1.20,1000),(600))
ds.addSample((1.50,1000),(750))
ds.addSample((10,1000),(5000))
#---------------------------
trainer = BackpropTrainer(ai,ds)
trainer.trainUntilConvergence(validationProportion=0.75)
trainer.trainEpochs(2000)

while(1):
    m = float(input("M:"))
    v = float(input("V:"))
    print("Ответ ИИ:"+str(ai.activate((m,v)))+" Джоуля")
