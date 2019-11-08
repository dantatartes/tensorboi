import numpy as np


class Criterion:
    def forward(self, input, target):
        raise NotImplementedError

    def backward(self, input, target):
        raise NotImplementedError


class MSE(Criterion):
    def forward(self, input, target):
        batch_size = input.shape[0]
        self.output = np.sum(np.power(input - target, 2)) / batch_size
        return self.output

    def backward(self, input, target):
        self.grad_output = (input - target) * 2 / input.shape[0]
        return self.grad_output


class CrossEntropy(Criterion):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input_clamp = np.clip(input, 1e-9, 1 - 1e-9)
        m = target.shape

        log_likelihood = -np.log(input_clamp[range(m[0]), np.where(target == 1)[1]])
        self.output = log_likelihood / m[1]

        return self.output

    def backward(self, input, target):
        input_clamp = np.clip(input, 1e-9, 1 - 1e-9)
        m = target.shape

        grad_input = input_clamp
        grad_input[range(m[0]), np.where(target == 1)[1]] -= 1
        grad_input /= m[1]

        return grad_input
