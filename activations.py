import numpy as np
from layers import Module


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def backward(self, input, grad_output):
        grad_input = np.multiply(grad_output, (input > 0))
        return grad_input


class LeakyReLU(Module):
    __slots__ = 'slope'

    def __init__(self, slope=0.03):
        super().__init__()
        self.slope = slope

    def forward(self, input):
        self.output = np.multiply(input, input > 0) + np.multiply(self.slope, np.multiply(input, input <= 0))
        return self.output

    def backward(self, input, grad_output):
        grad_input = np.multiply(grad_output, input > 0) + np.multiply(self.slope, np.multiply(grad_output, input <= 0))
        return grad_input


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, input, grad_output):
        grad_input = np.multiply(grad_output, np.multiply(self.output, 1 - self.output))
        return grad_input


class SoftMax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.output = np.exp(np.subtract(input, input.max(axis=1, keepdims=True)))
        self.output = self.output / np.sum(self.output, axis=1, keepdims=True)
        return self.output

    def backward(self, input, grad_output):
        grad_input = []
        for k in range(grad_output.shape[0]):
            grad_input.append(np.sum(np.diagflat(grad_output[k]) - np.dot(grad_output[k], grad_output[k].T), axis=1))
        return grad_input
