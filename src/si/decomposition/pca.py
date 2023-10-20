import numpy as np


class PCA:

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None

    def fit(self, X: np.ndarray):
        """ Estimates the mean, principal components and explained variance
        of the given data using SVD."""

        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        U, S, V = np.linalg.svd(X)
        self.components = V[:self.n_components, :]
        n_samples = X.shape[0]
        self.explained_variance = (S[:self.n_components] ** 2) / (n_samples - 1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """ Calculates the reduced dataset using the principal components."""

        X = X - self.mean
        return np.dot(X, self.components.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """ Estimates the mean, principal components and explained variance
        of the given data using SVD and calculates the reduced dataset
        using the principal components."""

        self.fit(X)
        return self.transform(X)


if __name__ == '__main__':
    pca = PCA(n_components=2)
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    reduced_data = pca.fit_transform(X)
    mean = pca.mean
    components = pca.components
    explained_variance = pca.explained_variance

    print("Original Data:\n", X)
    print("Mean:", mean)
    print("Principal Components:\n", components)
    print("Explained Variance:", explained_variance)
    print("Reduced Data:\n", reduced_data)
