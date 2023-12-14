import numpy as np


def accuracy(y_true, y_pred):
    correct_predictions = np.sum(np.round(y_pred) == y_true)
    total_samples = len(y_true)
    accuracy_value = correct_predictions / total_samples
    return accuracy_value


if __name__ == '__main__':
    y_true_example = np.array([1, 0, 1, 1, 0])
    y_pred_example = np.array([1, 0, 1, 0, 1])

    accuracy_value = accuracy(y_true_example, y_pred_example)
    print(accuracy_value)
