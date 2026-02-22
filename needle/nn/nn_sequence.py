"""The module.
"""

from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module, ReLU, Tanh, Sigmoid
import math
from collections import defaultdict


ACTIVATIONS = defaultdict(
    lambda: Tanh(), {"relu": ReLU(), "tanh": Tanh(), "sigmoid": Sigmoid()}
)


class RNNCell(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        k = (1 / hidden_size) ** 0.5

        self.W_ih = Parameter(
            init.rand(
                input_size,
                hidden_size,
                low=-k,
                high=k,
                device=device,
                dtype=dtype,
            )
        )

        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                hidden_size,
                low=-k,
                high=k,
                device=device,
                dtype=dtype,
            )
        )

        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    hidden_size,
                    low=-k,
                    high=k,
                    device=device,
                    dtype=dtype,
                )
            )

            self.bias_hh = Parameter(
                init.rand(
                    hidden_size,
                    low=-k,
                    high=k,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            self.bias_hh, self.bias_ih = None, None

        self.activation = ACTIVATIONS[nonlinearity]

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        X = X @ self.W_ih
        if self.bias_ih:
            X += ops.broadcast_to(self.bias_ih, X.shape)
        if h is not None:
            X += h @ self.W_hh
        if self.bias_hh:
            X += ops.broadcast_to(self.bias_hh, X.shape)

        X = self.activation(X)
        return X


class RNN(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cells = [
            RNNCell(
                input_size=input_size if _ == 0 else hidden_size,
                hidden_size=hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        seq_len, bs, input_size = X.shape

        if h0 is None:
            h0 = init.zeros(
                self.num_layers,
                bs,
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
            )

        X_ = ops.split(X, 0)
        h0_ = ops.split(h0, 0)
        outputs = []
        for timestep, x_i in enumerate(X_):
            for idx, cell in enumerate(self.rnn_cells):
                if timestep == 0:
                    h_i = cell(x_i, h0_[idx])
                else:
                    prev_hidden_idx = (timestep - 1) * self.num_layers + idx
                    h_i = cell(
                        x_i, outputs[prev_hidden_idx]
                    )
                x_i = h_i
                outputs.append(h_i)

        seq_output = [outputs[t * self.num_layers + self.num_layers - 1] for t in range(seq_len)]
        layer_output = [outputs[(seq_len - 1) * self.num_layers + l] for l in range(self.num_layers)]

        seq_output = ops.stack(seq_output, 0)
        layer_output = ops.stack(layer_output, 0)

        return seq_output, layer_output


class LSTMCell(Module):
    def __init__(
        self, input_size, hidden_size, bias=True, device=None, dtype="float32"
    ):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        k = (1 / hidden_size) ** 0.5

        self.W_ih = Parameter(
            init.rand(
                input_size,
                hidden_size * 4,
                low=-k,
                high=k,
                device=device,
                dtype=dtype,
            )
        )

        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                hidden_size * 4,
                low=-k,
                high=k,
                device=device,
                dtype=dtype,
            )
        )

        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    hidden_size * 4,
                    low=-k,
                    high=k,
                    device=device,
                    dtype=dtype,
                )
            )

            self.bias_hh = Parameter(
                init.rand(
                    hidden_size * 4,
                    low=-k,
                    high=k,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            self.bias_hh, self.bias_ih = None, None

        self.sigmoid = ACTIVATIONS["sigmoid"]
        self.tanh = ACTIVATIONS["tanh"]
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        bs, input_size = X.shape

        if h is None:
            h_0 = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
            c_0 = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h_0, c_0 = h

        X = X @ self.W_ih
        if self.bias_ih is not None:
            X += ops.broadcast_to(self.bias_ih, X.shape)
        if h_0 is not None:
            X += h_0 @ self.W_hh
        if self.bias_hh is not None:
            X += ops.broadcast_to(self.bias_hh, X.shape)

        X = X.reshape((bs, 4, self.hidden_size))
        i, f, g, o = ops.split(X, 1)
        i, f, g, o = (
            self.sigmoid(i),
            self.sigmoid(f),
            self.tanh(g),
            self.sigmoid(o),
        )
        c = f * c_0 + i * g
        h = o * self.tanh(c)

        return h, c


class LSTM(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.lstm_cells = [
            LSTMCell(
                input_size if layer == 0 else hidden_size,
                hidden_size,
                bias,
                device,
                dtype,
            )
            for layer in range(num_layers)
        ]

    def forward(self, X, h=None):
        """
        Inputs: X, h
        - X: of shape (seq_len, bs, input_size) containing the features of the input sequence.
        - h: tuple of (h0, c0) with h0 and c0 of shape (num_layers, bs, hidden_size).

        Outputs:
        - output: of shape (seq_len, bs, hidden_size) from the last layer of the LSTM, for each timestep.
        - (h_n, c_n): tuple of final hidden and cell states for each layer.
        """
        seq_len, bs, _ = X.shape

        if h is None:
            h_prev_list = [init.zeros(
                bs,
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
            ) for _ in range(self.num_layers)]
            c_prev_list = [init.zeros(
                bs,
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
            ) for _ in range(self.num_layers)]
        else:
            h_prev_list = ops.split(h[0], 0)
            c_prev_list = ops.split(h[1], 0)

        outputs = []
        X_ = ops.split(X, 0)

        for t in range(seq_len):
            x_i = X_[t]

            h_i_list = []
            c_i_list = []

            for layer in range(self.num_layers):
                h_prev, c_prev = h_prev_list[layer], c_prev_list[layer]
                cell = self.lstm_cells[layer]

                h_i, c_i = cell(x_i, (h_prev, c_prev))

                h_i_list.append(h_i)
                c_i_list.append(c_i)

                x_i = h_i

            outputs.append(h_i_list[-1])

            h_prev_list = h_i_list
            c_prev_list = c_i_list

        outputs = ops.stack(outputs, 0)

        h_n = ops.stack(h_prev_list, 0)
        c_n = ops.stack(c_prev_list, 0)

        return outputs, (h_n, c_n)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            init.randn(
                num_embeddings,
                embedding_dim,
                mean=0,
                std=1,
                device=device,
                dtype=dtype
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        seq_len, bs = x.shape
        x = init.one_hot(self.num_embeddings, x).reshape((-1, self.num_embeddings))
        return (x @ self.weight).reshape((seq_len, bs, self.embedding_dim))
