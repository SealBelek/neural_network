import numpy as np
from math import exp


class Neural_network:
    def __init__(self, k):
        self.K = k
        with open('data.csv', 'r') as data:
            self.X = np.loadtxt(data, dtype=int, delimiter=',')
        self.X_shape = self.X.shape

    def normalize(self, type="l", a=1.0):
        X_min = np.zeros(self.X_shape[1], dtype=int)
        X_max = np.zeros(self.X_shape[1], dtype=int)
        X_normolize = np.zeros(self.X_shape)
        for k in range(self.X_shape[1]):
            X_min[k] = min(item[k] for item in self.X)
            X_max[k] = max(item[k] for item in self.X)
        if type == "l":
            for i in range(self.X_shape[0]):
                for k in range(self.X_shape[1]):
                    X_normolize[i][k] = (self.X[i][k] - X_min[k]) / (X_max[k] - X_min[k])
            return X_normolize
        if type == "nl":
            X_c = np.zeros(self.X_shape[1])
            for k in range(self.X_shape[1]):
                X_c[k] = (X_min[k] + X_max[k]) / 2
            for i in range(self.X_shape[0]):
                for k in range(self.X_shape[1]):
                    X_normolize[i][k] = 1 / (exp(-a * (self.X[i][k] - X_c[k])) + 1)
            return X_normolize
