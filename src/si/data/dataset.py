from typing import Tuple, Sequence, List

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None) -> None:
        """
        Dataset represents a tabular dataset for single output classification.

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names
        label: str (1)
            The label name
        """
        if X is None:
            raise ValueError("X cannot be None")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        if features is None:
            features = [f"feat_{str(i)}" for i in range(X.shape[1])]
        if y is not None and label is None:
            label = "y"
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the dataset
        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Returns True if the dataset has a label
        Returns
        -------
        bool
        """
        return self.y is not None

    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset
        Returns
        -------
        numpy.ndarray (n_classes)
        """
        if self.has_label():
            return np.unique(self.y)
        else:
            raise ValueError("Dataset does not have a label")

    def get_mean(self) -> np.ndarray:
        """
        Returns the mean of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmean(self.X, axis=0)

    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmax(self.X, axis=0)

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset
        Returns
        -------
        pandas.DataFrame (n_features, 5)
        """
        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data

        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)

    def dropna(self):
        """
        Remove samples containing at least one NaN value and update X and y accordingly.
        """
        nan_indices = np.isnan(self.X).any(axis=1)

        self.X = self.X[~nan_indices]

        if self.has_label():
            self.y = self.y[~nan_indices]

        return self

    def fillna(self, values_list: List[float]):
        """
        Replace NaN values in the feature matrix with values from the provided list.

        Args:
            values_list: list of floats - List of medians or means to use as replacements for NaN values.

        Returns:
            Dataset: The modified Dataset object.
        """

        if not all(isinstance(value, (int, float)) for value in values_list):
            raise ValueError("values_list should contain only floats")

        num_columns = self.X.shape[1]

        if len(values_list) < num_columns:
            raise ValueError("values_list should have at least as many values as columns in X")

        if not np.array_equal(values_list, self.get_mean()) and not np.array_equal(values_list, self.get_median()):
            raise ValueError("values_list should be the array of means or medians of the variables")

        for col_idx in range(num_columns):
            col_values = self.X[:, col_idx]
            nan_mask = np.isnan(col_values)
            if np.any(nan_mask):
                replacement_value = values_list[col_idx]
                col_values[nan_mask] = replacement_value
                self.X[:, col_idx] = col_values

        return self

    def remove_by_index(self, index: int):
        if index < 0 or index >= len(self.X):
            raise ValueError("Index is out of bounds")

        self.X = np.delete(self.X, index, axis=0)

        if self.has_label():
            self.y = np.delete(self.y, index)

        return self


if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    features = np.array(['a', 'b', 'c'])
    label = 'y'
    dataset = Dataset(X, y, features, label)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())
    print(dataset.summary())
    X = np.array([[1.0, 2.0, 3.0],
                  [4.0, np.nan, 6.0],
                  [7.0, 8.0, 9.0]])

    y = np.array([0, 1, 2])

    dataset = Dataset(X, y)
    print(dataset.dropna().shape())
    X = np.array([[1.0, 2.0, 3.0],
                  [4.0, np.nan, 6.0],
                  [7.0, 8.0, 9.0]])

    y = np.array([0, 1, 2])

    dataset = Dataset(X, y)
    print(dataset.fillna([4.0, 5.0, 6.0]).X)
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    features = np.array(['a', 'b', 'c'])
    label = 'y'
    dataset = Dataset(X, y, features, label)
    print(dataset.remove_by_index(1).X)
