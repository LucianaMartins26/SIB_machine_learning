import numpy as np


def mse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / len(y_true)


if __name__ == '__main__':
    y_true_example = np.array([1, 2, 3, 4, 5])
    y_pred_example = np.array([2, 3, 2, 4, 5])

    mse_value = mse(y_true_example, y_pred_example)

    print(mse_value)