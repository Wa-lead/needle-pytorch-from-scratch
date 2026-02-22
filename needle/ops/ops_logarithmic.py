from typing import Optional
from ..autograd import NDArray
from ..autograd import Tensor, TensorOp

from .ops_mathematic import *

from needle.backend_selection import Device, array_api, NDArray, default_device

class LogSoftmax(TensorOp):
    def compute(self, Z):
        max_Z = array_api.broadcast_to(Z.max(1, keepdims=True), Z.shape)
        Z_shifted = Z - max_Z
        log_sum_exp = array_api.log(array_api.exp(Z_shifted).sum(1, keepdims=True))
        return Z - max_Z - log_sum_exp

    def gradient(self, out_grad, node):
        Z = node.inputs[0]
        Z_exp = exp(Z)
        Z_sum = summation(Z_exp, 1).reshape((-1, 1)).broadcast_to(Z.shape)
        softmax_Z = Z_exp / Z_sum
        sum_out_grad = summation(out_grad, 1).reshape((-1, 1)).broadcast_to(Z.shape)
        return (out_grad - softmax_Z * sum_out_grad,)


def logsoftmax(a):
    return LogSoftmax()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        self.mZ = Z.max(self.axes, keepdims=True)
        mZ = array_api.broadcast_to(self.mZ, Z.shape)
        Z_ = array_api.exp(Z - mZ)
        Z_ = Z_.sum(self.axes, keepdims=False)
        Z_ = array_api.log(Z_)
        return Z_ + self.mZ.reshape(Z_.shape)

    def gradient(self, out_grad, node):
        inp = node.inputs[0]
        input_shape = inp.shape
        mZ = Tensor(array_api.broadcast_to(self.mZ, input_shape), device=inp.device)
        base_shape = list(input_shape)
        if isinstance(self.axes, int): self.axes = (self.axes,)
        axes = list(range(len(base_shape))) \
            if self.axes is None else self.axes
        for ax in axes:
            base_shape[ax] = 1
        out_grad = out_grad / summation(exp((inp - mZ)), self.axes)
        out_grad = out_grad.reshape(base_shape)
        out_grad = out_grad.broadcast_to(input_shape)
        out_grad = out_grad * exp(inp - mZ)
        return (out_grad,)


def logsumexp(a, axes=None):
    return LogSumExp(axes)(a)
