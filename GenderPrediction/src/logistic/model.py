"""
    Created by @namhainguyen2803 on 29/05/2023
"""

import numpy as np

class LogisticRegression():
    def __init__(self, X_train, y_train, learning_rate=0.01, penalty=None, num_iterations=10, intercept=True,
                 **kwargs):
        self.X = X_train
        self.y = y_train
        self.num_examples = X_train.shape[0]
        self.num_features = X_train.shape[1]
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.intercept = intercept
        self.penalty = penalty
        self.lambd = kwargs.pop("lambda", 1e-2)

        self.initialize_weight()
        self.X = self.add_intercept(self.X, self.intercept)

    def add_intercept(self, X, intercpt):
        if intercpt == True:
            if X.ndim == 1:
                num_example = 1
            else:
                num_example = X.shape[0]
            intercept = np.ones((num_example, 1))
            return np.concatenate((intercept, X), axis=1)
        else:
            return X

    def initialize_weight(self):
        if self.intercept == False:
            self.W = np.zeros(self.num_features)
        else:
            self.W = np.zeros(self.num_features + 1)

    def sigmoid_function(self, x):
        return 1 / (1 + np.exp((-1) * x))

    def loss(self, y_pred, y):
        return (-1) * (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)).mean()

    def step(self, X, y):
        y_pred = self.sigmoid_function(np.dot(X, self.W))
        loss = self.loss(y_pred, y)
        gradient = np.dot(X.T, (y_pred - y))
        if self.penalty == None:
            self.W -= self.learning_rate * gradient / X.shape[0]
        elif self.penalty == "l2":
            self.W[0] -= (self.learning_rate * gradient[0]) / X.shape[0]
            self.W[1:] -= (self.learning_rate * gradient[1:] + self.lambd * self.W[1:]) / X.shape[0]
        elif self.penalty == "l1":
            self.W[0] -= (self.learning_rate * gradient[0]) / X.shape[0]
            self.W[1:] -= (self.learning_rate * gradient[1:] + self.lambd * np.sign(self.W[1:])) / X.shape[0]

    def data_generator(self, x_train, y_train, batch_size, randomize=False):
        num_examples = len(x_train)
        num_features = len(x_train[0])
        if randomize == True:
            ind_arr = np.arange(num_examples)
            np.random.shuffle(ind_arr)
            x_train, y_train = x_train[ind_arr], y_train[ind_arr]
        for i in range(0, num_examples, batch_size):
            idx = i
            idy = min(i + batch_size, num_examples)
            yield x_train[idx:idy], y_train[idx:idy]

    def fit(self):
        history_loss = list()
        for i in range(self.num_iterations):
            batch_obj = self.data_generator(self.X, self.y, batch_size=64, randomize=True)
            prev_w = self.W.copy()
            for x_batch, y_batch in batch_obj:
                self.step(x_batch, y_batch)
                y_pred = self.sigmoid_function(np.dot(self.X, self.W))
                loss = self.loss(y_pred, self.y)
                history_loss.append(loss)
            if np.mean(np.abs(self.W - prev_w)) < 1e-4:
                break
        return history_loss

    def predict_proba(self, X_test):
        X_test = self.add_intercept(X_test, self.intercept)
        return self.sigmoid_function(np.dot(X_test, self.W))