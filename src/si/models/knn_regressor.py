from typing import Callable

import numpy as np

from SIB_machine_learning.src.si.data.dataset import Dataset
from SIB_machine_learning.src.si.statistics.euclidean_distance import euclidean_distance
from SIB_machine_learning.src.si.metrics.rmse import rmse


class KNNRegressor:

    def __init__(self, k: int, distance: Callable = euclidean_distance):
        self.k = k
        self.distance = distance
        self.dataset = None

    def fit(self, dataset: Dataset):
        self.dataset = dataset

    def predict(self, dataset: Dataset):
        predictions = []

        for x in dataset.X:
            distances = []

            for i, x_train in enumerate(self.dataset.X):
                distances.append((i, self.distance(x, x_train)))

            distances.sort(key=lambda x: x[1])
            k_nearest = distances[:self.k]
            y_k_nearest = [self.dataset.y[i] for i, _ in k_nearest]
            predictions.append(np.mean(y_k_nearest))
        return np.array(predictions)

    def score(self, dataset: Dataset):
        y_pred = self.predict(dataset)
        return rmse(dataset.y, y_pred)


if __name__ == '__main__':
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([10, 20, 15, 25])
    dataset_train = Dataset(X_train, y_train)

    X_test = np.array([[1.5, 2.5], [3.5, 4.5]])
    y_test = np.array([15, 22])
    dataset_test = Dataset(X_test, y_test)

    k = 2
    knn_regressor = KNNRegressor(k, distance=euclidean_distance)

    knn_regressor.fit(dataset_train)

    y_pred = knn_regressor.predict(dataset_test)

    rmse_value = rmse(dataset_test.y, y_pred)

    print(f"Predicted values: {y_pred}")
    print(f"RMSE: {rmse_value}")