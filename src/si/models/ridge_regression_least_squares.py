import numpy as np
from SIB_machine_learning.src.si.metrics.mse import mse


class RidgeRegressionLeastSquares:
    def __init__(self, l2_penalty=1.0, scale=False):
        self.l2_penalty = l2_penalty
        self.scale = scale
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def fit(self, X, y):
        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X = (X - self.mean) / self.std

        X = np.c_[np.ones(X.shape[0]), X]

        penalty_matrix = self.l2_penalty * np.eye(X.shape[1])
        penalty_matrix[0, 0] = 0

        self.theta = np.linalg.inv(X.T @ X + penalty_matrix) @ X.T @ y
        self.theta_zero = self.theta[0]
        self.theta = self.theta[1:]

    def predict(self, X):
        if self.scale:
            X = (X - self.mean) / self.std

        X = np.c_[np.ones(X.shape[0]), X]
        thetas = np.r_[self.theta_zero, self.theta]

        y_pred = X.dot(thetas)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        score = mse(y, y_pred)
        return score


if __name__ == '__main__':
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([10, 20, 15, 25])

    X_test = np.array([[1.5, 2.5], [3.5, 4.5]])
    y_test = np.array([12, 22])

    l2_penalty = 1.0
    ridge_reg = RidgeRegressionLeastSquares(l2_penalty, scale=True)

    ridge_reg.fit(X_train, y_train)

    y_pred = ridge_reg.predict(X_test)

    mse_value = ridge_reg.score(X_test, y_test)

    print(f"Predicted values: {y_pred}")
    print(f"Mean Squared Error: {mse_value}")