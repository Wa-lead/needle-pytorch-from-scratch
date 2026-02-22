"""The module.
"""

from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import *


def nchw_to_nhwc(x):
    return x.transpose().transpose((1, 3))

def nhwc_to_nchw(x):
    return x.transpose((1,3)).transpose()  

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = self.kernel_size // 2
        self.weight = Parameter(
            init.kaiming_uniform(
                fan_in=self.kernel_size * self.kernel_size * self.in_channels,
                fan_out=self.kernel_size * self.kernel_size * self.out_channels,
                shape=(
                    self.kernel_size,
                    self.kernel_size,
                    self.in_channels,
                    self.out_channels,
                ),
                device=device,
                dtype=dtype,
            )
        )
        self.bias = None
        if bias:
            interval = 1 / (self.in_channels * self.kernel_size ** 2) ** 0.5
            self.bias = Parameter(
                init.rand(
                    out_channels,
                    low=interval,
                    high=interval,
                    device=device,
                    dtype=dtype,
                )
            )
    def forward(self, x: Tensor) -> Tensor:
        # shape NCHW => NCWH  
        x = nchw_to_nhwc(x)
        x = ops.conv(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            x += ops.broadcast_to(self.bias, x.shape)
        x = nhwc_to_nchw(x)
        return x
class ConvBN(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,bias=True, device=None, dtype='float32'):
        super().__init__()
        self.conv = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            device=device,
            dtype=dtype
        )
        self.batch_norm = BatchNorm2d(
            dim = out_channels,
            device=device,
            dtype=dtype
        )
        self.relu = ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
        