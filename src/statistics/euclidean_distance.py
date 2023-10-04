import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff = ((x - y) ** 2).sum(axis=1)
    return np.sqrt(diff)
