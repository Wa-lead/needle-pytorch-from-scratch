# Needle: A Deep Learning Framework for CMU's dlsys Course

This project is part of the Carnegie Mellon University (CMU) Deep Learning Systems (dlsys) course (10-414/714)  . Needle is a deep learning framework that provides the foundational components needed to build and train neural networks, with a focus on flexibility and extensibility.

## Key Components

- **Tensor Data Structure**: 
  - The `Tensor` is the core data structure in Needle, responsible for maintaining the computation graph necessary for automatic differentiation. It wraps around the `NDArray`, which handles the actual data storage and operations. The `Tensor` can utilize different backends, such as NumPy or custom implementations, for efficient numerical computations, depending on the system's configuration.

- **Automatic Differentiation (Autodiff)**:
  - Needle includes an `autograd` module that supports automatic differentiation. This module tracks operations performed on `Tensors` and builds a computation graph, enabling efficient gradient computation necessary for training neural networks. The integration with the `Tensor` structure allows for seamless gradient propagation through complex networks.

- **Neural Network Modules (`nn.modules`)**:
  - The framework provides a set of neural network modules (e.g., layers like `Linear`, `Conv2D`) under the `nn` package. These modules are designed to be composable, enabling the construction of complex neural networks by stacking simple building blocks. The `nn.modules` are tightly integrated with the autodiff system, ensuring that gradients are automatically computed during the backward pass.

- **Backend Support (including NumPy)**:
  - Needle is designed to be backend-agnostic. The primary backend is often NumPy, but the framework also supports custom backends, allowing for flexibility in how tensor operations are executed. The `backend_numpy` module, for example, implements the necessary operations to perform tensor computations using NumPy, but alternative backends can be utilized depending on performance needs or specific use cases.

- **Operations (`ops`)**:
  - The `ops` module defines the core mathematical operations that are used with tensors in the autodiff system. These operations are the building blocks for creating computational graphs, which are essential for tracking and computing gradients. Key operations include:
    - **`ops_mathematic.py`**: Implements basic mathematical operations like addition, multiplication, and more.
    - **`ops_logarithmic.py`**: Handles logarithmic functions and related operations.
    - **`ops_tuple.py`**: Provides operations that work on tuple structures, useful in certain advanced network configurations.

## Project Structure Overview

- **`autograd.py`**: Implements the automatic differentiation engine.
- **`backend_ndarray/`**: Contains implementations of the `NDArray` for different backends.
- **`nn/`**: Includes neural network modules (`nn.modules`) for constructing models.
- **`ops/`**: Defines the mathematical operations used throughout the framework, particularly for tensors in the autodiff system.
- **`data/`**: Provides data handling and transformation utilities, including dataset loaders.

## About

This project was developed as part of the CMU dlsys course, where students learn to design and implement the core components of deep learning systems. Needle serves as a hands-on educational tool to understand the inner workings of deep learning frameworks.
