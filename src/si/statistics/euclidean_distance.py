import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((x - y)**2, axis=-1))


if __name__ == '__main__':
    x_example = np.array([1, 2, 3])
    y_example = np.array([4, 5, 6])

    distance = euclidean_distance(x_example, y_example)
    print(distance)