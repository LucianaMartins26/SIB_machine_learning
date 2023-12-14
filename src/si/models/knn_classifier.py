from typing import Callable

import numpy as np

from SIB_machine_learning.src.si.data.dataset import Dataset
from SIB_machine_learning.src.si.statistics.euclidean_distance import euclidean_distance
from SIB_machine_learning.src.si.metrics.accuracy import accuracy


class KNNClassifier:

    def __init__(self, k: int, distance: Callable = euclidean_distance):
        self.k = k
        self.distance = distance
        self.X_train = None

    def fit(self, dataset: Dataset):
        self.X_train = dataset
        return self

    def get_closest_label(self, sample: np.ndarray):
        distances = self.distance(sample, self.X_train.X)
        k_nearest = np.argsort(distances)[:self.k]
        k_nearest_labels = self.X_train.y[k_nearest]
        labels, counts = np.unique(k_nearest_labels, return_counts=True)
        return labels[np.argmax(counts)]

    def predict(self, dataset: Dataset):
        return np.apply_along_axis(self.get_closest_label, 1, dataset.X)

    def score(self, dataset: Dataset):
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)


if __name__ == '__main__':
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([0, 1, 0, 1])
    dataset_train = Dataset(X_train, y_train)

    X_test = np.array([[1.5, 2.5], [3.5, 4.5]])
    y_test = np.array([0, 1])
    dataset_test = Dataset(X_test, y_test)

    k = 2
    knn_classifier = KNNClassifier(k, distance=euclidean_distance)

    knn_classifier.fit(dataset_train)

    y_pred = knn_classifier.predict(dataset_test)

    accuracy_value = accuracy(dataset_test.y, y_pred)

    print(f"Predicted labels: {y_pred}")
    print(f"Accuracy: {accuracy_value}")