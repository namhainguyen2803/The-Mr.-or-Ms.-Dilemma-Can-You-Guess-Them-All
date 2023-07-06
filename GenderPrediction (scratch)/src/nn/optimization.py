"""
    Created by @namhainguyen2803 on 29/05/2023
"""

import numpy as np
def sgd(w, dw, config=None):
    if config is None:
        config = {}
    lr = config.setdefault("learning_rate", 1e-2)
    w = w - lr * dw
    return w, config


def rmsprop(w, dw, config=None):
    if config is None:
        config = {}
    lr = config.setdefault("learning_rate", 1e-2)
    decay_rate = config.setdefault("decay_rate", 0.99)
    epsilon = config.setdefault("epsilon", 1e-8)
    velocity = config.setdefault("velocity", np.zeros_like(w))

    config["w"] = decay_rate * velocity + (1 - decay_rate) * dw * dw
    w = w - lr * np.divide(dw, (np.sqrt(config["w"]) + epsilon))
    return w, config