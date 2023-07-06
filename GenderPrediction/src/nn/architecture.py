"""
    Created by @namhainguyen2803 on 29/05/2023
"""

import numpy as np
from src.nn.utils import *
class FullyConnectedNet(object):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            regularization=0.0,
            weight_scale=5e-2,
            dtype=np.float64
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.weight_scale = weight_scale
        self.regularization = regularization
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.initialization_type = "he"
        self.params = {}
        self.initialize_parameters()

    def initialize_parameters(self):
        all_layers_dim = np.hstack([self.input_dim, self.hidden_dims, 1])

        if self.initialization_type == None:
            for i in range(self.num_layers):
                self.params["W" + str(i + 1)] = np.random.normal(loc=0.0, scale=self.weight_scale,
                                                                 size=(all_layers_dim[i].astype(np.int64), all_layers_dim[i + 1].astype(np.int64)))
                self.params["b" + str(i + 1)] = np.zeros((all_layers_dim[i + 1]))
            for k, v in self.params.items():
                self.params[k] = v.astype(self.dtype)

        elif self.initialization_type == "he":
            for i in range(self.num_layers):
                self.params["W" + str(i + 1)] = np.random.normal(loc=0.0, scale=np.sqrt(2/all_layers_dim[i]),
                                                                 size=(all_layers_dim[i].astype(np.int64), all_layers_dim[i + 1].astype(np.int64)))
                self.params["b" + str(i + 1)] = np.zeros((all_layers_dim[i + 1]))
            for k, v in self.params.items():
                self.params[k] = v.astype(self.dtype)

    def loss(self, X, y):
        x = X.astype(self.dtype)
        caches = []
        for i in range(self.num_layers - 1):
            W = self.params["W" + str(int(i + 1))]
            b = self.params["b" + str(int(i + 1))]
            x, cache = affine_relu_forward(x, W, b)
            caches.append(cache)

        W = self.params["W" + str(self.num_layers)]
        b = self.params["b" + str(self.num_layers)]
        scores, cache = affine_forward(x, W, b)
        caches.append(cache)

        loss, out_grad = binary_cross_entropy_loss_with_sigmoid(scores, y)
        dout = out_grad

        for i in range(self.num_layers):
            w = self.params["W" + str(i + 1)]
            loss += 0.5 * self.regularization * np.sum(w * w)

        dout, dw, db = affine_backward(dout, caches[self.num_layers - 1])
        grads = {}
        grads["W" + str(self.num_layers)] = dw + self.regularization * self.params["W" + str(self.num_layers)]
        grads["b" + str(self.num_layers)] = db
        for i in range(self.num_layers - 2, -1, -1):
            dx, dw, db = affine_relu_backward(dout, caches[i])
            grads["W" + str(i + 1)] = dw + self.regularization * self.params["W" + str(i + 1)]
            grads["b" + str(i + 1)] = db
            dout = dx
        return loss, grads

    def loss_only(self, X, y):
        x = X.astype(self.dtype)
        caches = []
        for i in range(self.num_layers - 1):
            W = self.params["W" + str(int(i + 1))]
            b = self.params["b" + str(int(i + 1))]
            x, cache = affine_relu_forward(x, W, b)
            caches.append(cache)

        W = self.params["W" + str(self.num_layers)]
        b = self.params["b" + str(self.num_layers)]
        scores, cache = affine_forward(x, W, b)
        caches.append(cache)

        loss, out_grad = binary_cross_entropy_loss_with_sigmoid(scores, y)

        for i in range(self.num_layers):
            w = self.params["W" + str(i + 1)]
            loss += 0.5 * self.regularization * np.sum(w * w)
        return loss

    def predict_proba(self, X):
        x = X.astype(self.dtype)
        for i in range(self.num_layers - 1):
            W = self.params["W" + str(int(i + 1))]
            b = self.params["b" + str(int(i + 1))]
            x, cache = affine_relu_forward(x, W, b)

        W = self.params["W" + str(self.num_layers)]
        b = self.params["b" + str(self.num_layers)]
        scores, _ = affine_forward(x, W, b)
        prob, _ = sigmoid_forward(scores)
        return prob