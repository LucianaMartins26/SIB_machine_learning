import numpy as np


def manhattan_distance(x: np.ndarray, y: np.ndarray):
    return abs(x - y).sum(axis=1)
