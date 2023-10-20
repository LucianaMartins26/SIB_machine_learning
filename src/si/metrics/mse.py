import numpy as np


def mse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / len(y_true)