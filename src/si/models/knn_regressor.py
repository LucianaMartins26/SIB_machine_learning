from typing import Callable

from SIB_machine_learning.src.si.data.dataset import Dataset
from SIB_machine_learning.src.statistics.euclidean_distance import euclidean_distance
from SIB_machine_learning.src.metrics.rmse import rmse


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
        """
        Calculate the distance between each sample and various samples in the training dataset;
        Obtain the indexes of the k most similar examples (shortest distance);
        Use the previous indexes to retrieve the corresponding values in Y;
        Calculate the average of the values obtained in step 3;
        Apply steps 1, 2, 3, and 4 to all samples in the testing dataset.
        """
        predictions = []

        for x in dataset.X:
            distances = []

            for x_train in self.dataset.X:
                distances.append(self.distance(x, x_train))

            indexes = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
            y = [self.dataset.Y[i] for i in indexes]
            predictions.append(sum(y) / len(y))

        return predictions

    def score(self, dataset: Dataset):
        """
        Get the predictions (y_pred);
        Calculate the rmse between actual values and predictions.
        """
        y_pred = self.predict(dataset)
        return rmse(dataset.y, y_pred)
