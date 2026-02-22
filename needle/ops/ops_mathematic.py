"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

from ..backend_selection import array_api, NDArray

from functools import reduce
import operator


def prod(x):
    return reduce(operator.mul, x, 1)


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)
class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)
class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)
class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a**self.scalar
    def gradient(self, out_grad, node):
        node = node.inputs[0]
        return out_grad * self.scalar * power_scalar(node, self.scalar - 1)
def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)
class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b


def power(a, b):
    return EWisePow()(a, b)
class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        if a.ndim != b.ndim:
            b = b.reshape((-1, 1))
        return a / b
    def gradient(self, out_grad, node):
        lfs, rhs = node.inputs
        return out_grad / rhs, negate(out_grad) * lfs / power_scalar(rhs, 2)
def divide(a, b):
    return EWiseDiv()(a, b)
class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar
    def gradient(self, out_grad, node):
        return out_grad / self.scalar
def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)
class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            return a.swapaxes(-1, -2)

        else:
            return a.swapaxes(self.axes[0], self.axes[1])
    def gradient(self, out_grad, node):
        if self.axes is None:
            return out_grad.transpose(axes=(-1, -2))

        else:
            return transpose(out_grad, self.axes)
def transpose(a, axes=None):
    return Transpose(axes)(a)
class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)
    def gradient(self, out_grad, node):
        x = node.inputs[0]
        return reshape(out_grad, x.shape)
def reshape(a, shape):
    return Reshape(shape)(a)
class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    # this is simply broadcasting back to the same dimensions
    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        if prod(out_grad.shape) == prod(input_shape):
            return (out_grad.reshape(input_shape),)
        base_shape = [1] * (len(self.shape) - len(input_shape)) + list(input_shape)
        axes = []
        for i in range(len(base_shape)):
            if self.shape[i] != base_shape[i]:
                axes.append(i)
        out_grad = summation(out_grad, axes=tuple(axes))
        return (out_grad.reshape(input_shape),)
def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if not isinstance(self.axes, (tuple, list)):
            return a.sum(self.axes, keepdims=False)

        # (1,2,3)
        if len(self.axes) > 1:
            axes = list(self.axes)[::-1]
            increments = 0
            while len(axes) > 0:
                a = a.sum(axes[-1] - increments, keepdims=False)
                axes.pop()
                increments += 1
            return a
        elif len(self.axes) < 1:
            return a.sum(keepdims=False)
        else:
            return a.sum(self.axes, keepdims=False)
    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        base_shape = list(input_shape)
        if isinstance(self.axes, int):
            self.axes = (self.axes,)
        axes = list(range(len(base_shape))) if self.axes is None else self.axes
        for ax in axes:
            base_shape[ax] = 1
        out_grad = out_grad.reshape(base_shape)
        out_grad = out_grad.broadcast_to(input_shape)
        return (out_grad,)
def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        if a.ndim == 2 and b.ndim == 2:
            return a @ b
        # Batched matmul: loop over batch dimensions using C++ 2D matmul
        if isinstance(a, numpy.ndarray):
            return numpy.matmul(a, b)
        # Reshape to (batch, M, K) @ (batch, K, N) via numpy broadcast
        a_np, b_np = a.numpy(), b.numpy()
        # Use numpy to compute broadcast shape, then do per-slice C++ matmul
        a_bc = numpy.broadcast_to(a_np, numpy.broadcast_shapes(a_np.shape[:-2], b_np.shape[:-2]) + a_np.shape[-2:])
        b_bc = numpy.broadcast_to(b_np, numpy.broadcast_shapes(a_np.shape[:-2], b_np.shape[:-2]) + b_np.shape[-2:])
        batch_shape = a_bc.shape[:-2]
        M, K = a_bc.shape[-2], a_bc.shape[-1]
        N = b_bc.shape[-1]
        a_flat = a_bc.reshape(-1, M, K)
        b_flat = b_bc.reshape(-1, K, N)
        out_slices = []
        for i in range(a_flat.shape[0]):
            ai = NDArray(a_flat[i], device=a.device)
            bi = NDArray(b_flat[i], device=a.device)
            out_slices.append((ai @ bi).numpy())
        out_np = numpy.stack(out_slices).reshape(batch_shape + (M, N))
        return NDArray(out_np.astype(numpy.float32), device=a.device)
    def gradient(self, out_grad, node):
        a, b = node.inputs
        a_shape = a.shape
        b_shape = b.shape
        lhs = out_grad @ transpose(b, [-2, -1])
        rhs = transpose(a, [-2, -1]) @ out_grad
        while a_shape != lhs.shape:
            lhs = summation(lhs, 0)
        while b_shape != rhs.shape:
            rhs = summation(rhs, 0)
        return (lhs, rhs)


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a
    def gradient(self, out_grad, node):
        return negate(out_grad)
def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)
    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        return out_grad / lhs
def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)
    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])
def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)
    def gradient(self, out_grad, node):
        input = node.inputs[0]
        value = input.realize_cached_data()
        mask = Tensor(value > 0, device=out_grad.device)
        return out_grad * mask


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        return a.tanh()

    def gradient(self, out_grad: Value, node: Value):
        lhs = node.inputs[0]
        return (
            out_grad * (1 + (-tanh(lhs) ** 2)),
        )


def tanh(a):
    return Tanh()(a)


class Sigmoid(TensorOp):
    def compute(self, a):
        return (1 + array_api.exp(-a)) ** (-1)

    def gradient(self, out_grad, node):
        inp = node.inputs[0]
        sigmoid_out = sigmoid(inp)
        return (out_grad * sigmoid_out * (1 + negate(sigmoid_out)),)


def sigmoid(a):
    return Sigmoid()(a)


class Softmax(TensorOp):
    def __init__(self, axes) -> None:
        self.axes = axes

    def compute(self, a):
        e_a = a.exp()
        return e_a / e_a.sum(self.axes, keepdims=True).broadcast_to(e_a.shape)

    def gradient(self, out_grad: Value, node: Value):
        lhs = node.inputs[0]  # Softmax output
        # Compute sum(out_grad * y) along the last axis, keep dimensions for broadcasting
        sum_grad_y = (
            (out_grad * lhs).sum(self.axes)
        )
        # Compute the gradient
        return (lhs * (out_grad - sum_grad_y),)


def softmax(a, axes):
    return Softmax(axes)(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, arrays: TensorTuple) -> Tensor:
        return array_api.stack(arrays, self.axis)
    def gradient(self, out_grad, node):
        out_grad = split(out_grad, self.axis)
        return out_grad
def stack(arrays, axis):
    return Stack(axis)(make_tuple(*arrays))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        return array_api.split(A, self.axis)
    def gradient(self, out_grad, node):
        out_grad = stack(out_grad, self.axis)
        return (out_grad,)
def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes=None):
        self.axes = axes

    def compute(self, a):
        return a.flip(self.axes)

    def gradient(self, out_grad: Value, node: Value):
        lhs = node.inputs[0]
        return (flip(out_grad, self.axes),)


def flip(a, axes):
    return Flip(axes=axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes, dilation):
        self.axes = (axes,) if isinstance(axes, int) else axes
        self.dilation = dilation

    def compute(self, a):
        return array_api.dilate(a, self.axes, self.dilation)

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class unDilate(TensorOp):
    def __init__(self, axes, dilation):
        self.axes = (axes,) if isinstance(axes, int) else axes
        self.dilation = dilation

    def compute(self, a):
        return array_api.undilate(a, self.axes, self.dilation)

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return unDilate(axes, dilation)(a)


class Conv(TensorOp):

    def __init__(self, stride, padding):
        self.padding = padding
        self.stride = stride

    def compute(self, A, B):

        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        inner_dim = K * K * C_in

        A_ = A.pad(
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        )

        Ns, Hs, Ws, Cs = A_._strides
        H_out, W_out = (H + 2 * self.padding - K) // self.stride + 1, (
            W + 2 * self.padding - K
        ) // self.stride + 1

        A_ = (
            A_.as_strided(
                shape=(N, H_out, W_out, K, K, C_in),
                strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs),
            )
            .compact()
            .reshape((-1, inner_dim))
        )

        B_ = B.compact().reshape((-1, C_out))

        out = A_ @ B_

        return out.reshape((N, H_out, W_out, C_out))

    def gradient(self, out_grad, node):
        inp, weight = node.inputs
        # grad w.r.t input
        grad_pad = dilate(out_grad, (1, 2), self.stride - 1)
        weight_t = flip(weight, (0, 1))
        weight_t = weight_t.transpose()
        K = weight_t.shape[0]
        grad_inp = conv(grad_pad, weight_t, 1, K - 1 - self.padding)

        # grad w.r.t kernel
        grad_pad = grad_pad.transpose((0, 2)).transpose((0, 1))
        inp_t = inp.transpose((0, 3))
        grad_weight = conv(inp_t, grad_pad, 1, self.padding)
        grad_weight = grad_weight.transpose((0, 2)).transpose((0, 1))

        return (grad_inp, grad_weight)
def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
