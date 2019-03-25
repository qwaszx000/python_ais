class KNN:

    def fit(self, X, Y):
        self.x = X
        self.y = Y

    def predict(self, X):
        distances = []
        for p in self.x:
            distances.append(self.dist(X, p))

        near_dist = min(distances)
        near_x = self.x[distances.index(near_dist)]
        return self.y[distances.index(near_dist)]

    #distance between 2 points in some dimension
    def dist(self, X1, X2):
        summ = 0.0
        for x in X1:
            summ += x
            
        for x in X2:
            summ -= x

        return abs(summ**( 2/( len(X1)+len(X2) ) ))


x_learn = [[0.5], [1.0], [1.5], [5.0], [5.5], [6.0]]
x_test = [3.0]
y_learn = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0]
y_test = 1.0

ai = KNN()
ai.fit(x_learn, y_learn)

print(ai.predict(x_test))
print("Expected:")
print(y_test)
