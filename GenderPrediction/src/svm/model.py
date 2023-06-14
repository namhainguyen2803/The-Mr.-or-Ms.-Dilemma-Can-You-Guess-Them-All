"""
    Created by @namhainguyen2803 on 29/05/2023
"""

import numpy as np


class Support_Vector_Machine():
    def __init__(self, X_train, y_train, C=1, kernel_type=None, **kwargs):
        self.X = X_train
        self.y = y_train
        self.num_examples = X_train.shape[0]
        self.num_features = X_train.shape[1]
        self.C = C
        self.K = self.set_K(kernel_type)
        self.sp = np.zeros((self.num_examples, 1))
        self.b_sp = None
        self.b_t = None

        self.tolerance = 1e-5
        self.gamma = kwargs.pop("gamma", "scale")
        self.degree = kwargs.pop("degree", 3)
        self.t = np.random.randn(1)

        self.H = None
        self.initialize_H()

    def initialize_H(self):
        self.H = self.K(self.X, self.X)

    def set_K(self, kernel_type):
        if kernel_type == "rbf":
            return self.compute_radial_basic_function
        elif kernel_type == None:
            return self.compute_original_kernel

    def compute_original_kernel(self, x1, x2):
        return np.dot(x1, x2.T)

    def compute_radial_basic_function(self, x1, x2):
        g = 1
        if np.ndim(x1) == 1:
            x1 = x1[np.newaxis, :]
        if np.ndim(x2) == 1:
            x2 = x2[np.newaxis, :]
        dist_squared = np.linalg.norm(x1[:, :, np.newaxis] - x2.T[np.newaxis, :, :], axis=1) ** 2
        dist_squared = np.squeeze(dist_squared)
        if self.gamma == "scale":
            g = 1 / (self.num_features * self.X.var())
        return np.exp(-g * dist_squared)

    def clip_sp(self, a, l, u):
        if a < l:
            a = l
        if a > u:
            a = u
        return a

    def get_lu(self, i, j):
        if self.y[i] != self.y[j]:
            l = max(0, self.sp[j] - self.sp[i])  # L
            u = min(self.C, self.C + self.sp[j] - self.sp[i])  # H
        else:
            l = max(0, self.sp[i] + self.sp[j] - self.C)
            u = min(self.C, self.sp[i] + self.sp[j])
        return l, u

    def get_er(self, i):
        x_i = self.X[i, :]
        y_pred = self.__predict(x_i, i)
        return y_pred - self.y[i]

    def __check(self, i, h_i):
        if -self.tolerance < self.sp[i] < self.tolerance:
            if self.y[i] * h_i < 1 + self.tolerance:
                return False
        elif -self.tolerance + self.C < self.sp[i] < self.tolerance + self.C:
            if self.y[i] * h_i > 1 - self.tolerance:
                return False
        elif -self.tolerance < self.sp[i] < self.tolerance + self.C:
            if 1 - self.tolerance < self.y[i] * h_i or self.y[i] * h_i < 1 - self.tolerance:
                return False
        if self.sp[i] < self.tolerance:
            return 10
        return True

    def fit(self, num_iteration=50):
        b_acc = 1e-9
        for iter in range(num_iteration):
            for i in range(self.num_examples):
                h_i = self.get_er(i)
                if ((self.y[i] * h_i < -self.tolerance) and (self.sp[i] < self.C)) or (
                        (self.y[i] * h_i > self.tolerance) and (self.sp[i] > 0)):
                    j, h_j = self.heu_sel(i, h_i)
                    self.step(i, j, h_i, h_j)
            y_pred = self.__predict_class(self.X, "H")
            acc = np.mean(y_pred == self.y)
            if acc > b_acc:
                b_acc = acc
                self.b_sp = self.sp.copy()
                self.b_t = self.t.copy()

    def predict(self, x):
        y = np.reshape(self.y, (-1, 1))
        a = np.reshape(self.b_sp, (-1, 1))
        k = self.K(self.X, x)
        if np.ndim(k) == 1:
            k = k[:, np.newaxis]
        return np.sum(y * a * k, axis=0) + self.b_t

    def predict_class(self, x):
        res = self.predict(x)
        return np.sign(res)

    def __predict(self, x, i=None):
        y = np.reshape(self.y, (-1, 1))
        a = np.reshape(self.sp, (-1, 1))
        k = self.K(self.X, x) if i == None else self.H[i, :] if i != "H" else self.H
        if np.ndim(k) == 1:
            k = k[:, np.newaxis]
        return np.sum(y * a * k, axis=0) + self.t

    def __predict_class(self, x, i=None):
        res = self.__predict(x, i)
        return np.sign(res)

    def update(self, oi, ni, oj, nj, i, j, h_i, h_j):
        x_i = self.X[i, :]
        x_j = self.X[j, :]
        y_i = self.y[i]
        y_j = self.y[j]

        t_1 = self.t - h_i - y_i * (ni - oi) * self.H[i, i] - y_j * (nj - oj) * \
              self.H[i, j]
        t_2 = self.t - h_j - y_i * (ni - oi) * self.H[i, j] - y_j * (nj - oj) * \
              self.H[j, j]

        if 0 < ni < self.C:
            self.t = t_1
        elif 0 < nj < self.C:
            self.t = t_2
        else:
            self.t = (t_1 + t_2) / 2

    def heu_sel(self, i, h_i):
        j = 0
        max_e = 1e-9
        min_e = 1e9
        h_j = 0
        for el in range(self.num_examples):
            if el == i:
                continue
            else:
                e_el = self.get_er(el)
                if h_i < 0:
                    if e_el > max_e:
                        j = el
                        h_j = e_el
                        max_e = e_el
                else:
                    if e_el < min_e:
                        j = el
                        h_j = e_el
                        min_e = e_el
            return j, h_j

    def naive_sel(self, i, h_i):
        candidate = [element for element in range(self.num_examples) if element != i]
        j = np.random.choice(candidate)
        h_j = self.get_er(j)
        return j, h_j

    def step(self, i, j, h_i, h_j):
        x_i = self.X[i, :]
        x_j = self.X[j, :]
        y_i = self.y[i]
        y_j = self.y[j]
        l, u = self.get_lu(i, j)

        if l == u:
            return

        if 2 * self.H[i, j] - self.H[i, i] - self.H[j, j] >= 0:
            return

        _sp_j = self.sp[j] - ((y_j * (h_i - h_j)) / (2 * self.H[i, j] - self.H[i, i] - self.H[j, j]))
        _sp_j = self.clip_sp(_sp_j, l, u)

        if np.abs(self.sp[j] - _sp_j) < 1e-5:
            return
        _sp_i = self.sp[i] + y_i * y_j * (self.sp[j] - _sp_j)
        self.update(self.sp[i], _sp_i, self.sp[j], _sp_j, i, j, h_i, h_j)
        self.sp[j] = _sp_j
        self.sp[i] = _sp_i