import numpy as np

from SIB_machine_learning.src.si.data.dataset import Dataset


class RidgeRegression:

    def __init__(self, l2_penalty, alpha, max_iter, patience, scale):
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale
        self.theta = None
        self.cost_history = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def fit(self, dataset: Dataset):

        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std

        else:
            X = dataset.X

        m, n = dataset.shape()

        self.theta = np.zeros(n)
        self.theta_zero = 0

        i = 0
        early_stopping = 0

        while i < self.max_iter and early_stopping < self.patience:

            y_pred = self.theta_zero + np.dot(self.theta, X)
            gradient = self.alpha / m * np.sum(np.dot(y_pred - dataset.y, X))
            regularization = self.theta * (1 - (self.alpha * self.l2_penalty) / m)

            self.theta = self.theta * regularization - gradient
            self.theta_zero = self.theta_zero - ((self.alpha / m) * np.sum(y_pred - dataset.y))
            self.cost_history[i] = self.cost(dataset)

            if i > 0 and self.cost_history[i] > self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0

            i += 1

    def cost(self, dataset: Dataset):

        if self.scale:
            X = (dataset.X - self.mean) / self.std

        else:
            X = dataset.X

        m, n = dataset.shape()
        y_pred = self.theta_zero + np.dot(self.theta, X)
        cost = (1 / (2 * m)) * np.sum(np.square(y_pred - dataset.y)) + (self.l2_penalty / (2 * m)) * np.sum(np.square(self.theta))
        return cost
