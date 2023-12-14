import numpy as np


def manhattan_distance(x: np.ndarray, y: np.ndarray):
    return abs(x - y).sum(axis=1)


if __name__ == '__main__':
    x_example = np.array([[1, 2, 3],
                          [4, 5, 6]])
    y_example = np.array([7, 8, 9])

    distances = manhattan_distance(x_example, y_example)
    print(distances)