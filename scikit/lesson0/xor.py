import sklearn.neural_network
import numpy as np

xs = np.array([
    0, 0,
    0, 1,
    1, 0,
    1, 1
]).reshape(4, 2)#2 входа

ys = np.array([0, 1, 1, 0]).reshape(4,)#1 выход

nn = sklearn.neural_network.MLPClassifier(
    activation="relu", max_iter=99999999, hidden_layer_sizes=(4,2))#2 скрытых слоя 
#входной - 2, первый слой - 4, второй - 2, выходной - 1

nn.fit(xs, ys)
while(nn.score(xs, ys) != 1.0):#Учим
    nn.fit(xs, ys)

print("Score:", nn.score(xs, ys))#1 - отличные знания  0 - незнание 
print("Ответ:", nn.predict(xs))  #Получаем ответ нейросети
