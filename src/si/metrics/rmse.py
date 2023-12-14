import numpy as np


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


if __name__ == '__main__':
    y_true_example = np.array([1, 2, 3, 4, 5])
    y_pred_example = np.array([2, 3, 2, 4, 5])

    rmse_value = rmse(y_true_example, y_pred_example)

    print(rmse_value)
