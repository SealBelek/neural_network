import numpy as np
import random
from math import exp, sqrt


class Neural_network:
    def __init__(self):
        with open('data.csv', 'r') as data:
            self.X = np.loadtxt(data, dtype=int, delimiter=',')
        self.X_shape = self.X.shape

    def normalize(self, type="l", a=1.0):
        X_min = np.zeros(self.X_shape[1], dtype=int)
        X_max = np.zeros(self.X_shape[1], dtype=int)
        self.X_normolize = np.zeros(self.X_shape)
        for k in range(self.X_shape[1]):
            X_min[k] = min(item[k] for item in self.X)
            X_max[k] = max(item[k] for item in self.X)
        if type == "l":
            for i in range(self.X_shape[0]):
                for k in range(self.X_shape[1]):
                    self.X_normolize[i][k] = (self.X[i][k] - X_min[k])/(X_max[k] - X_min[k])
        if type == "nl":
            X_c = np.zeros(self.X_shape[1])
            for k in range(self.X_shape[1]):
                X_c[k] = (X_min[k] + X_max[k]) / 2
            for i in range(self.X_shape[0]):
                for k in range(self.X_shape[1]):
                    self.X_normolize[i][k] = 1 / (exp(-a * (self.X[i][k] - X_c[k])) + 1)

    def self_learning(self, k=4, v=0.3, m=0.05, count=6):
        K = k
        a = 0.5 - (1 / sqrt(self.X_shape[1]))
        b = 0.5 + (1 / sqrt(self.X_shape[1]))
        w = (b - a) * np.random.random_sample((k, self.X_shape[1])) + a
        print(w)
        index = np.arange(self.X_shape[0])
        while(v and count):
            np.random.shuffle(index)
            print(index)
            for ind in index:
                R = np.zeros(k)
                for i in range(k):
                    for j in range(self.X_shape[1]):
                        R[i] += (self.X_normolize[ind][j] - w[i][j])**2
                    R[i] = sqrt(R[i])
                min = np.argmin(R)
                for j in range(self.X_shape[1]):
                    w[min][j] = w[min][j] + v*(self.X_normolize[ind][j] - w[min][j])
            v -= m
            count -= 1
            print('e')
        print(w)
