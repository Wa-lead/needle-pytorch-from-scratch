# Needle

A from-scratch deep learning framework featuring automatic differentiation, a custom NDArray with C++ CPU backend, neural network modules, and optimizers.

## Features

- **Automatic Differentiation** -- reverse-mode autograd with lazy evaluation support
- **Custom NDArray** -- strided N-dimensional array backed by a C++ (pybind11) CPU kernel with tiled matrix multiplication; optional CUDA backend
- **Neural Network Modules** -- Linear, Conv2d, BatchNorm, LayerNorm, Dropout, RNN, LSTM, Embedding, and more
- **Optimizers** -- SGD (with momentum) and Adam, with gradient clipping support
- **Datasets** -- MNIST, CIFAR-10, and Penn Treebank loaders with data augmentation transforms

## Architecture

```
needle/
  autograd.py            # Tensor, computational graph, backpropagation
  backend_selection.py   # Runtime backend dispatch (NDArray / NumPy)
  backend_ndarray/
    ndarray.py           # Strided NDArray implementation
    ndarray_backend_cpu  # C++ CPU backend (pybind11)
  ops/
    ops_mathematic.py    # Core math ops (add, matmul, conv, etc.)
    ops_logarithmic.py   # LogSoftmax, LogSumExp
  nn/
    nn_basic.py          # Linear, BatchNorm, LayerNorm, Dropout, etc.
    nn_conv.py           # Conv, ConvBN
    nn_sequence.py       # RNN, LSTM, Embedding
  init/                  # Weight initialization (Xavier, Kaiming)
  optim.py               # SGD, Adam
  data/                  # Dataset and DataLoader
src/
  ndarray_backend_cpu.cc   # C++ source for CPU backend
  ndarray_backend_cuda.cu  # CUDA source (stub)
```

## Quick Start

### Build the C++ backend

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Run

```python
import needle as ndl
import needle.nn as nn

# Create a simple model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# Forward pass
x = ndl.Tensor(np.random.randn(32, 784).astype("float32"))
out = model(x)

# Backward pass
out.sum().backward()
```

See `examples/mnist_resnet.ipynb` for a full training example on MNIST with a ResNet-style MLP.

## License

MIT
