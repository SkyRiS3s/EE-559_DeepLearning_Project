import torch
import math
from torch.autograd import grad
torch.set_grad_enabled(False)


class myModule(object):
    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_wrt_output):
        raise NotImplementedError

    def param(self):
        return []


class Linear(myModule):
    "Linear fully connected"

    def __init__(self, in_size, out_size, lr = 1e-6):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.W = torch.empty(self.in_size, self.out_size).normal_()
        self.gW = torch.empty(self.in_size, self.out_size).fill_(0)
        self.b = torch.empty(1, self.out_size).normal_()
        self.gb = torch.empty(1, self.out_size).fill_(0)
        self.lr = lr

    def forward(self, input):
        self.input = input
        self.output = self.input.mm(self.W).add(self.b)
        return self.output

    def grad_descent(self, lr):
        """Gradient descent"""
        self.W.sub_(lr * self.gW)
        self.b.sub_(lr * self.gb)

    def backward(self, grad_wrt_output):
        self.gW.add_(self.input.T.mm(grad_wrt_output))
        self.gb.add_(grad_wrt_output.sum(axis=0))
        tmp = grad_wrt_output.mm(self.W.T)
        self.grad_descent(self.lr)
        return tmp

    def param(self):
        return [(self.W, self.gW), (self.b, self.gb)]


class Sequential(myModule):
    """Sequential model created from a list of layers"""
    def __init__(self, layer_list):
        super().__init__()
        self.layer_list = layer_list
        self.input = None
        self.output = None
        self.g = None
        self.param = []

    def forward(self, input):
        self.input = input
        self.output = input
        for current_layer in self.layer_list:
            self.output = current_layer.forward(self.output)
        return self.output

    def backward(self, grad):
        self.g = grad
        # iterating through list in reverse
        for layer in self.layer_list[::-1]:
            self.g = layer.backward(self.g)
        return self.g

    def param(self):
        for layer in self.layer_list:
            for layer_params in layer.param():
                self.param.append(layer_params)
        return self.param


# Losses
# MSET
class MSE_loss(myModule):
    """Mean Squared Error"""
    def forward(self, y, y_h):
        self.y = y
        self.y_h = y_h
        diffs = y-y_h
        return diffs.pow(2).mean()

    def backward(self):
        diffs = self.y - self.y_h
        N = self.y.size(0)
        self.g = 2*diffs / N
        return self.g

# MAE

class MAE_loss(myModule):
    """Mean Absolute Error"""
    def forward(self, y, y_h):
        self.y = y
        self.y_h = y_h
        diffs = y-y_h
        diffs[diffs < 0] = diffs[diffs < 0] * (-1)
        return torch.mean(diffs)

    def backward(self):
        diffs = self.y-self.y_h
        N = self.y.size(0)
        self.g = ((-1)*(diffs < 0)+(diffs >= 0)) / N  # sign func
        return self.g


# Activation Layers
# Relu
class ReLU(myModule):
    def forward(self, input):
        self.input = input
        input[input<0] = 0
        self.output = input
        return self.output

    def backward(self, grad):
        tmp = self.input
        tmp[tmp > 0 ] = 1
        tmp[tmp < 0] = 0
        return tmp.mul(grad)

# Sigmoid
class Sigmoid(myModule):
    def forward(self, input):
        self.input = input
        self.output = 1/(1+(-self.input).exp())
        return self.output

    def backward(self, grad):
        self.g = grad * (1/(1+(-self.input).exp()) -
                         (1/(1+(-self.input).exp()))**2)
        return self.g

# Tanh
class Tanh(myModule):
    def forward(self, input):
        self.input = input
        self.output = torch.tanh(input)
        return self.output
    def backward(self, grad):
        self.g = grad * (self.output.pow(2).mul(-1).add(1))
        return self.g
