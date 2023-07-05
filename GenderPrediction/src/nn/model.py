"""
    Created by @namhainguyen2803 on 29/05/2023
"""

import numpy as np
import copy

class VanillaFeedForwardNetwork(object):
    def __init__(self, architecture, X_train, y_train, optim, **kwargs):
        self.model = architecture
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = None
        self.y_test = None
        self.update_rule = optim

        self.best_model = None

        self.lr_tolerance = 5
        self.lr_min = 1e-6

        self.batch_size = kwargs.pop("batch_size", None)
        self.num_epochs = kwargs.pop("num_epochs", 40)
        self.verbose = kwargs.pop("verbose", True)
        self.print_every = kwargs.pop("print_every", 10)
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.learning_rate = kwargs.pop("learning_rate", 0.01)
        self.model.regularization = kwargs.pop("regularization", 0.0)

        self.optim_configs = {}
        for p in self.model.params:
            self.optim_configs[p] = {"learning_rate": self.learning_rate}

    def step(self, X_batch, y_batch):
        loss, grads = self.model.loss(X_batch, y_batch)
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def check_accuracy(self, model, X, y):
        y_pred = model.predict_proba(X)
        y_pred = np.hstack(y_pred)
        y_pred[y_pred < 0.5] = 0
        y_pred[y_pred >= 0.5] = 1
        acc = np.mean(y_pred == y)
        return acc

    def predict_proba(self, x):
        return self.best_model.predict_proba(x)


    def data_generator(self, x_train, y_train, randomize=False):
        num_examples = len(x_train)
        if self.batch_size != None:
          if randomize == True:
              ind_arr = np.arange(num_examples)
              np.random.shuffle(ind_arr)
              x_train, y_train = x_train[ind_arr], y_train[ind_arr]
          for idx in range(0, num_examples, self.batch_size):
              idy = min(idx + self.batch_size, num_examples)
              yield x_train[idx:idy], y_train[idx:idy]
        else:
          yield x_train, y_train

    def train(self):
        best_loss = 1e9
        tolerance = self.lr_tolerance
        for t in range(self.num_epochs):
            batch_obj = self.data_generator(self.X_train, self.y_train, True)
            for x_batch, y_batch in batch_obj:
                self.step(x_batch, y_batch)
            train_loss, _ = self.model.loss(self.X_train, self.y_train)
            if self.verbose == True:
                if t % self.print_every == 0:
                    train_acc = self.check_accuracy(self.model, self.X_train, self.y_train)
                    if self.X_test is not None and self.y_test is not None:
                        test_acc = self.check_accuracy(self.model, self.X_test, self.y_test)
                        print(f"Iteration {t}: training loss {train_loss}, training accuracy: {train_acc}, test accuracy: {test_acc}")
                    else:
                        print(f"Iteration {t}: training loss {train_loss}, training accuracy: {train_acc}")

            if train_loss < best_loss:
                best_loss = train_loss
                tolerance = self.lr_tolerance
                self.best_model = copy.deepcopy(self.model)
            else:
                tolerance = tolerance - 1
                if tolerance <= 0:
                    tolerance = self.lr_tolerance
                    for k in self.optim_configs:
                        if self.optim_configs[k]["learning_rate"] * self.lr_decay > self.lr_min:
                            self.optim_configs[k]["learning_rate"] *= self.lr_decay
                        else:
                            self.optim_configs[k]["learning_rate"] = self.lr_min

    def eval(self, X_test, y_test):
        return self.check_accuracy(self.best_model, X_test, y_test)