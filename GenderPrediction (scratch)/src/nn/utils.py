"""
    Created by @namhainguyen2803 on 29/05/2023
"""

import numpy as np

def affine_forward(x, w, b):
    out = x.dot(w) + np.reshape(b, (-1, b.shape[0]))
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    x, w, b = cache
    dw = x.T.dot(dout)
    dx = np.reshape(dout.dot(w.T), x.shape)
    db = dout.T.dot(np.ones((np.shape(dout)[0],)))
    return dx, dw, db

def relu_forward(x):
    out = np.copy(x)
    out[out < 0] = 0
    cache = x
    return out, cache

def relu_backward(dout, cache):
    x = cache
    x_d = np.copy(x)
    x_d[x < 0] = 0
    x_d[x > 0] = 1
    dx = dout * x_d
    return dx

def binary_cross_entropy_loss(x, y):
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    N = np.shape(x)[0]
    loss = -1 * (y * np.log(x)  + (1 - y) * np.log(1 - x))
    dx = - (np.divide(y, x) - np.divide(1 - y, 1 - x))
    loss = np.mean(loss)
    dx /= N
    return loss, dx

def sigmoid_forward(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-1 * x)), x

def sigmoid_backward(dout, cache):
    x = cache
    sig_x,_ = sigmoid_forward(x)
    return dout * sig_x * (1 - sig_x)

def binary_cross_entropy_loss_with_sigmoid(x, y):
    N = np.shape(x)[0]
    z, _ = sigmoid_forward(x)
    y = y.reshape(z.shape)
    loss = -1 * (y * np.log(z + 1e-8)  + (1 - y) * np.log(1 - z + 1e-8))
    loss = np.mean(loss)
    dy_to_z = - (np.divide(y, z + 1e-8) - np.divide(1 - y, 1 - z + 1e-8))
    dz_to_x = z * (1 - z)
    dx = dy_to_z * dz_to_x
    dx /= N
    return loss, dx

def affine_sigmoid_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, sigmoid_cache = sigmoid_forward(a)
    cache = (fc_cache, sigmoid_cache)
    return out, cache

def affine_sigmoid_backward(dout, cache):
    fc_cache, sigmoid_cache = cache
    da = sigmoid_backward(dout, sigmoid_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db