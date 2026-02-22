import needle
from typing import List, Optional, Tuple, Union, Dict
from .init import *
import numpy
from abc import abstractmethod
from .backend_selection import Device, array_api, NDArray, default_device, cpu, all_devices


LAZY_MODE = False
TENSOR_COUNTER = 0



class Op:
    "Operator definition"

    @abstractmethod
    def __call__(self, *args):
        ...

    @abstractmethod
    def compute(self, *args: Tuple[NDArray]):
        ...

    @abstractmethod
    def gradient(self, out_grad: 'Value', node: 'Value'):
        ...


    def gradient_as_tuple(self, outgrad: 'Value', node: 'Value'):
        output = self.gradient(outgrad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)



class TensorOp(Op):
    """A subclass of Op that creates a Tensor based on the operation (building graph)."""

    def __call__(self, *args):
        return Tensor.make_from_op(self,args)


class TensorTupleOp(Op):
    """Op class specialized to output TensorTuple"""

    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)





class Value:
    """A value in the computational graph"""
    op: Optional[Op]
    inputs: List["Value"]
    cached_data: NDArray
    requires_grad: bool


    def realize_cached_data(self):
        if self.cached_data is not None:
            return self.cached_data
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -=1

    def _init(
        self,
        op: Optional[Op],
        inputs: List['Tensor'],
        *,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad


    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(
            op=None,
            inputs=[],
            cached_data=data,
            requires_grad=requires_grad
        )
        return value


    @classmethod
    def make_from_op(cls, op:Op, inputs:List['Value']):
        value = cls.__new__(cls)
        value._init(op=op, inputs=inputs)
        if LAZY_MODE is False:
            if value.requires_grad is False:
                return value.detach()
            value.realize_cached_data()
        return value

class TensorTuple(Value):
    """Represents a tuple of tensors in the computational graph."""

    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return needle.ops.tuple_get_item(self, index)

    def tuple(self):
        return ([x for x in self])

    def __repr__(self):
        return "needle.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return needle.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return TensorTuple.make_const(self.realize_cached_data())

class Tensor(Value):
    grad: 'Tensor'


    def __init__(
        self,
        array,
        device: Optional[Device] = None,
        dtype=None,
        require_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else cpu()
            cached_data = Tensor._array_from_numpy(
                array, device=device, dtype=dtype
            )

        self._init(
            op=None,
            inputs=[],
            cached_data=cached_data,
            requires_grad=require_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List['Value']):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if LAZY_MODE is False:
            if tensor.requires_grad is False:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor


    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor


    @property
    def data(self):
        return self.detach()


    @data.setter
    def data(self,value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
                    value.dtype,
                    self.dtype,
                )
        self.cached_data = value.realize_cached_data()


    def detach(self):
        return Tensor.make_const(data=self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return default_device()
        return data.device


    def backward(self, out_grad=None):
        out_grad = (
            out_grad
            if out_grad
            else ones(*self.shape, dtype=self.dtype, device=self.device)
        )
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
            return "needle.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        """Support indexing and slicing on Tensors (detached, no grad)."""
        return Tensor(self.numpy()[key], device=self.device)

    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return data
        return data.numpy() if not isinstance(data, tuple) else [x.numpy() for x in data]


    # --- now overwite the builtin callables for operations

    def __add__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, other)
        else:
            return needle.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseMul()(self, other)
        else:
            return needle.ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWisePow()(self, other)
        else:
            return needle.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, needle.ops.Negate()(other))
        else:
            return needle.ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseDiv()(self, other)
        else:
            return needle.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return needle.ops.MatMul()(self, other)

    def matmul(self, other):
        return needle.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return needle.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return needle.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return needle.ops.Reshape(shape)(self)

    def __neg__(self):
        return needle.ops.Negate()(self)

    def transpose(self, axes=None):
        return needle.ops.Transpose(axes)(self)



    def __rsub__(self, other):
        return (-self) + other

    __radd__ = __add__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__



def compute_gradient_of_variables(output_tensor, out_grad):
    node_to_grad = {
        output_tensor: [out_grad]
    }
    reversed_nodes = list(reversed(find_topo_sort([output_tensor])))
    for node in reversed_nodes:
        node.grad = sum_node_list(node_to_grad[node])
        if node.is_leaf():
            continue
        gradient_w_r_inputs = node.op.gradient_as_tuple(
            outgrad=node.grad,
            node=node
        )
        for idx, input_node in enumerate(node.inputs):
            if input_node in node_to_grad:
                node_to_grad[input_node].append(gradient_w_r_inputs[idx])
            else:
                node_to_grad[input_node] = [gradient_w_r_inputs[idx]]


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    topo_order = []
    visited = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order
def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.append(node)
    for inp in list(node.inputs):
        topo_sort_dfs(inp, visited, topo_order)
    topo_order.append(node)
    return
def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce

    return reduce(add, node_list)
