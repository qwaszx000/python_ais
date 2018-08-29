from sklearn.tree import DecisionTreeRegressor
import numpy as np

xs = np.array([
    0, 0,
    0, 1,
    1, 0,
    1, 1
]).reshape(4, 2)

ys = np.array([0, 1, 1, 0]).reshape(4,)

nn = DecisionTreeRegressor(
    max_depth=8)

nn.fit(xs, ys)

print(nn.predict(xs))
