import numpy as np

from SIB_machine_learning.src.si.neural_networks.layers import Layer


class Dropout(Layer):

    def __init__(self, probability):
        super().__init__()
        self.probability = probability

        self.mask = None
        self.input = None
        self.output = None

    def forward_propagation(self, input_shape: np.ndarray, training: bool):

        if training is True:
            scaling_factor = 1 / (1 - self.probability)
            self.mask = np.random.binomial(1, self.probability, size=input_shape.shape) / self.probability
            self.output = input_shape * self.mask * scaling_factor

            return self.output
        else:
            return input_shape

    def backward_propagation(self, output_error: float) -> float:
        return output_error * self.mask

    def output_shape(self) -> tuple:
        return self.input_shape

    def parameters(self) -> int:
        return 0


if __name__ == '__main__':
    np.random.seed(42)
    random_input = np.random.rand(10)
    dropout_layer = Dropout(probability=0.5)
    output_training = dropout_layer.forward_propagation(random_input, training=True)
    print("Output during training:\n", output_training)
    output_testing = dropout_layer.forward_propagation(random_input, training=False)
    print("Output during testing:\n", output_testing)
