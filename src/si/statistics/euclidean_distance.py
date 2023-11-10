import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    if len(x.shape) == 1:
        diff = (x - y) ** 2
    else:
        diff = ((x - y) ** 2).sum(axis=1)
    return np.sqrt(diff)
