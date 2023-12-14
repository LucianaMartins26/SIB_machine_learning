import numpy as np
from abc import abstractmethod


class Optimizer:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        if learning_rate <= 0:
            raise ValueError('learning_rate must be positive')

    @abstractmethod
    def update(self, w, grad_loss_w):
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate=0.01, momentum=0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w, grad_loss_w):
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))

        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w

        return w - self.learning_rate * self.retained_gradient


class Adam(Optimizer):

    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, grad_loss_w):

        if self.m is None:
            self.m = np.zeros(w.shape)
        if self.v is None:
            self.v = np.zeros(w.shape)

        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_loss_w
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_loss_w ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        w = w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return w


if __name__ == '__main__':
    w = np.array([2.0, 1.0])
    grad_loss_w = np.array([1.0, -2.0])

    sgd_optimizer = SGD(learning_rate=0.01)
    updated_w_sgd = sgd_optimizer.update(w, grad_loss_w)
    print("Updated Weights (SGD):", updated_w_sgd)

    adam_optimizer = Adam(learning_rate=0.01)
    updated_w_adam = adam_optimizer.update(w, grad_loss_w)
    print("Updated Weights (Adam):", updated_w_adam)

    sgd_optimizer_lr_001 = SGD(learning_rate=0.01)
    adam_optimizer_lr_001 = Adam(learning_rate=0.01)

    updated_w_sgd_lr_001 = sgd_optimizer_lr_001.update(w, grad_loss_w)
    updated_w_adam_lr_001 = adam_optimizer_lr_001.update(w, grad_loss_w)

    print("Updated Weights (SGD, LR=0.01):", updated_w_sgd_lr_001)
    print("Updated Weights (Adam, LR=0.01):", updated_w_adam_lr_001)