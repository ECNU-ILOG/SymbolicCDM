import numpy as np


# Since most values fall between 0 and 1, offset is needed
def sigmoid(x, offset=0.5):
    ret = 1 / (1 + np.exp(-(x - offset)))
    return ret


def dot(x: np.ndarray, y: np.ndarray):
    return np.sum(x * y)


# Tanh function
def tanh(x, offset=0.5, sharp=2):
    return np.tanh(sharp * (x - offset)) / 2 + 0.5


def sigArctan(x):
    return sigmoid(2 * np.pi * np.arctan(x))


# subtraction combined with Relu
def relu(x):
    return np.maximum(0, x)


def reluSub(x, y):
    return relu(x - y)

