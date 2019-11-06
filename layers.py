import numpy as np


class Module:
    __slots__ = '_train'

    def __init__(self):
        self._train = True

    def forward(self, input):
        raise NotImplementedError

    def backward(self, input, grad_output):
        raise NotImplementedError

    def parameters(self):
        return []

    def grad_parameters(self):
        return []

    def train(self):
        self._train = True

    def eval(self):
        self._train = False


class Sequential(Module):
    __slots__ = 'layers', 'output'

    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)

        self.output = input
        return self.output

    def backward(self, input, grad_output):
        for i in range(len(self.layers) - 1, 0, -1):
            grad_output = self.layers[i].backward(self.layers[i - 1].output, grad_output)

        grad_input = self.layers[0].backward(input, grad_output)

        return grad_input

    def parameters(self):
        res = []
        for l in self.layers:
            res += l.parameters()
        return res

    def grad_parameters(self):
        res = []
        for l in self.layers:
            res += l.grad_parameters()
        return res

    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()


class Linear(Module):
    __slots__ = 'W', 'b', 'grad_W', 'grad_b', 'output'

    def __init__(self, dim_in, dim_out):
        super().__init__()

        stdv = 1. / np.sqrt(dim_in)
        self.W = np.random.uniform(-stdv, stdv, size=(dim_in, dim_out))
        self.b = np.random.uniform(-stdv, stdv, size=dim_out)

    def forward(self, input):
        self.output = np.dot(input, self.W) + self.b
        return self.output

    def backward(self, input, grad_output):
        self.grad_b = np.mean(grad_output, axis=0)

        #     in_dim x batch_size
        self.grad_W = np.dot(input.T, grad_output)
        #                 batch_size x out_dim

        grad_input = np.dot(grad_output, self.W.T)

        return grad_input

    def parameters(self):
        return [self.W, self.b]

    def grad_parameters(self):
        return [self.grad_W, self.grad_b]
