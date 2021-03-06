from sklearn.tree import DecisionTreeRegressor
import numpy as np

global xs,ys,nn

def add_l(x1, x2, y):
    global xs,ys,nn
    xs = np.append(xs, [[x1, x2]])
    #print(xs.shape[0])
    xs = xs.reshape((int(xs.shape[0]/2), 2))
    ys = np.append(ys,[[y]])
    ys = ys.reshape(ys.shape)

def learn_n(r = 1.0):
    global xs,ys,nn
    nn.fit(xs, ys)
    while(nn.score(xs, ys) != r):#Учим
        nn.fit(xs, ys)

xs = np.array([])

ys = np.array([])

nn = DecisionTreeRegressor(max_depth=9999)

while(1):
    xi1 = int(input('First num:'))
    xi2 = int(input('Second num:'))
    yi = (input('result(\"-\" if answering nn):'))
    if(yi == "-"):
        ans = np.array([xi1,xi2]).reshape(1, 2)
        print(nn.predict(ans))
    else:
        add_l(xi1, xi2, int(yi))
        learn_n()
        print("Score:", nn.score(xs, ys))#1 - отличные знания  0 - незнание 
        print("Ответ:", nn.predict(xs))  #Получаем ответ нейросети
