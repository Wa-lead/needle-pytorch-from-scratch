"""The module.
"""

from typing import List, Callable, Any
from needle.autograd import Tensor
from needle.ops import *
from needle.init import *
import numpy as np
from abc import abstractmethod
from typing import List


def nchw_to_nhwc(x):
    return x.transpose().transpose((1, 3))


def nhwc_to_nchw(x):
    return x.transpose((1, 3)).transpose()


class Parameter(Tensor):
    """Indicates a trainable tensor"""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self):
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    @abstractmethod
    def forward(self, *args, **kwargs): ...

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            kaiming_uniform(
                fan_in=self.in_features,
                fan_out=self.out_features,
                device=device,
                dtype=dtype,
            )
        )

        self.bias = (
            Parameter(
                transpose(
                    kaiming_uniform(
                        fan_in=out_features, fan_out=1, device=device, dtype=dtype
                    )
                )
            )
            if bias
            else None
        )

    def forward(self, x: Tensor):
        out = x @ self.weight
        if self.bias:
            out += broadcast_to(self.bias, out.shape)
        return out


class Flatten(Module):
    def forward(self, x):
        return reshape(x, (x.shape[0], -1))


class ReLU(Module):
    def forward(self, x):
        return relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor):
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        num, classes = logits.shape
        y_one_hot = one_hot(classes, y)
        loss = (
            log(exp(logits))
            - broadcast_to(reshape(logsumexp(logits, 1), (num, 1)), logits.shape)
        ) * y_one_hot
        loss = divide_scalar(loss, -num)
        loss = summation(loss)
        return loss


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.running_mean = zeros(dim, device=device, dtype="float32")
        self.running_var = ones(dim, device=device, dtype="float32")
        self.weight = Parameter(ones(dim, device=device, dtype="float32"))
        self.bias = Parameter(zeros(dim, device=device, dtype="float32"))

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        if self.training:
            mean = summation(x, 0) / shape[0]
            x = x - broadcast_to(mean, x.shape)
            var = summation(power_scalar(x, 2), 0) / shape[0]
            x = x / broadcast_to(power_scalar(var + self.eps, 0.5), shape)
            self.running_mean = (
                self.momentum * mean + (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                self.momentum * var + (1 - self.momentum) * self.running_var
            )
        else:
            x = x - self.running_mean.broadcast_to(x.shape)
            x = x / power_scalar(self.running_var + self.eps, 0.5).broadcast_to(shape)
        x = x * self.weight.broadcast_to(shape) + self.bias.broadcast_to(shape)
        return x


class BatchNorm2d(BatchNorm1d):
    def __init__(self, dim, eps=0.00001, momentum=0.1, device=None, dtype="float32"):
        super().__init__(dim, eps, momentum, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        x = nchw_to_nhwc(x)
        x = super().forward(x.reshape((-1, C))).reshape(x.shape)
        return nhwc_to_nchw(x)


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = Parameter(ones(dim, device=device, dtype="float32"))
        self.beta = Parameter(zeros(dim, device=device, dtype="float32"))

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        mean = broadcast_to(reshape(summation(x, 1) / self.dim, (batch, 1)), x.shape)
        std = broadcast_to(
            reshape(summation(power_scalar(x - mean, 2), 1) / self.dim, (batch, 1)),
            x.shape,
        )
        x = (x - mean) / power_scalar(std + self.eps, 0.5) * broadcast_to(
            reshape(self.gamma, (1, self.dim)), x.shape
        ) + broadcast_to(reshape(self.beta, (1, self.dim)), x.shape)
        return x


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.p > 0.0:
            mask = randb(*x.shape, p=(1 - self.p), device=x.device, dtype="float32")
            return (x * mask) / (1 - self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)


class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_exp = exp(x)
        return x_exp / x_exp.sum(-1).reshape((-1, 1)).broadcast_to(x.shape)
