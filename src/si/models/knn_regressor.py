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
        """
        Stores the training dataset
        """
        self.dataset = dataset

    def predict(self, dataset: Dataset):
        predictions = []

        for x in dataset.X:
            distances = []

            for i, x_train in enumerate(self.dataset.X):
                distances.append((i, self.distance(x, x_train)))

            distances.sort(key=lambda x: x[1][0])
            k_nearest = distances[:self.k]
            y_k_nearest = [self.dataset.y[i] for i, _ in k_nearest]
            predictions.append(np.mean(y_k_nearest))
        return np.array(predictions)

    def score(self, dataset: Dataset):
        """ calculates the error between the estimated values and the real ones (rmse)"""
        y_pred = self.predict(dataset)
        return rmse(dataset.y, y_pred)
