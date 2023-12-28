import numpy as np
import scipy.special
import torch


def add(x, y):
    return x + y


def mul(x, y):
    return x * y


def dot(x, y):
    if type(x) is np.ndarray and type(y) is np.ndarray:
        return np.sum(x * y, dtype=np.float64)
    else:
        return (x * y).sum(dim=1).unsqueeze(1)


def sigmoid(x):
    if type(x) is np.ndarray or type(x) is np.float64 or type(x) is np.float32:
        # to avoid overflow
        return scipy.special.expit(x)
    else:
        return torch.sigmoid(x)


def tanh(x):
    if type(x) is np.ndarray or type(x) is np.float64 or type(x) is np.float32:
        return np.tanh(x)
    else:
        return torch.tanh(x)
