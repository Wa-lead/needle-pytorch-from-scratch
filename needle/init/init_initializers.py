import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    a = gain * ((6 / (fan_in + fan_out)) ** 0.5)
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    sigma = gain * ((2 / (fan_in + fan_out)) ** 0.5)
    return randn(fan_in, fan_out, mean=0, std=sigma, **kwargs)
def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = 2 ** 0.5
    bound = gain * math.sqrt(3 / fan_in)
    if shape is None:
        return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    else:
        return rand(*shape, low=-bound, high=bound, **kwargs)
def kaiming_normal(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = 2 ** 0.5
    bound = gain / math.sqrt(fan_in)
    if shape is None:
        return randn(fan_in, fan_out, mean=0, std=bound, **kwargs)
    else:
        return randn(*shape, mean = 0, std=bound, **kwargs)
