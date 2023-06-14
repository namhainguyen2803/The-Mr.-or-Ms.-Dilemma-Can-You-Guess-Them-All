"""
    Created by @namhainguyen2803 on 1/06/2023
"""

import numpy as np

class NaiveBayes():
  def __init__(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train
    self.num_examples = X_train.shape[0]
    self.num_vocabulary = X_train.shape[1]
    self.phi_1 = np.zeros(self.num_vocabulary)
    self.phi_0 = np.zeros(self.num_vocabulary)
    self.phi = 0
  def fit(self):
    sum_men = np.sum(self.X_train[self.y_train==1], axis=0).reshape(1, -1)
    sum_women = np.sum(self.X_train[self.y_train==0], axis=0).reshape(1, -1)
    num_men = np.sum(self.y_train)
    self.phi_1 = (sum_men + 1) / (num_men + 2)
    self.phi_0 = (sum_women + 1) / (self.num_examples - num_men + 2)
    self.phi = num_men / self.num_examples
  def predict(self, x_test):
    if np.ndim(x_test) == 1:
      x_test = x_test.reshape(1, -1)
    a = x_test * self.phi_1
    a[a==0] = 1
    x_given_men = np.prod(a, axis=1) * self.phi
    b = x_test * self.phi_0
    b[b==0] = 1
    x_given_women = np.prod(b, axis=1) * (1 - self.phi)
    return x_given_men / (x_given_men + x_given_women)
  def predict_class(self, x_test):
    prob = self.predict(x_test)
    prob[prob<0.5] = 0
    prob[prob>=0.5] = 1
    return prob
  def test_accuracy(self, x_test, y_test):
    predict_class = self.predict_class(x_test)
    return np.mean(predict_class == y_test)