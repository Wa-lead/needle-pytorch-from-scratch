from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import Parameter, Module, ReLU, Dropout, LayerNorm1d, Linear, Sequential


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """

    def __init__(
        self,
        *,
        dropout=0.0,
        causal=False,
        device=None,
        dtype="float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1
        )

        return ndarray.array(mask, device=device)

    def matmul(self, a, b):
        """
        Batched matrix multiplication using the framework's MatMul op,
        which supports N-D tensors directly.
        """
        return a @ b


    def softmax(self, logit):
        """
        The softmax function;
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False,
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(self, q, k, v):
        batch_size, num_heads, seq_len, _ = q.shape

        attention_scores = self.matmul(q, ops.transpose(k)) / np.sqrt(q.shape[-1])

        if self.causal:
            mask = np.triu(np.ones((seq_len, seq_len)) * float('-inf'), k=1)
            mask = Tensor(mask.reshape(1, 1, seq_len, seq_len), device=q.device)
            attention_scores += mask

        attention_probs = self.softmax(attention_scores)

        if self.training:
            attention_probs = self.dropout(attention_probs)

        output = self.matmul(attention_probs, v)

        return output, attention_probs




class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout=0.0,
        causal=True,
        device=None,
        dtype="float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head

        self.q_projection = Linear(
            q_features, inner_dim, bias=False, device=device, dtype=dtype
        )
        self.k_projection = Linear(
            k_features, inner_dim, bias=False, device=device, dtype=dtype
        )
        self.v_projection = Linear(
            v_features, inner_dim, bias=False, device=device, dtype=dtype
        )

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal, device=device, dtype=dtype
        )

        self.out_projection = Linear(
            inner_dim, out_features, bias=False, device=device, dtype=dtype
        )

    def forward(
        self,
        q,
        k=None,
        v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        inner_dim = self.num_head * self.dim_head

        # Pre-normalize (LayerNorm1d expects 2D, so reshape)
        q = self.prenorm_q(q.reshape((batch_size * queries_len, q_dim)))
        q = q.reshape((batch_size, queries_len, q_dim))

        k = self.prenorm_k(k.reshape((batch_size * keys_values_len, k_dim)))
        k = k.reshape((batch_size, keys_values_len, k_dim))

        v = self.prenorm_v(v.reshape((batch_size * keys_values_len, v_dim)))
        v = v.reshape((batch_size, keys_values_len, v_dim))

        # Project (Linear expects 2D)
        q = self.q_projection(q.reshape((batch_size * queries_len, q_dim)))
        q = q.reshape((batch_size, queries_len, self.num_head, self.dim_head))

        k = self.k_projection(k.reshape((batch_size * keys_values_len, k_dim)))
        k = k.reshape((batch_size, keys_values_len, self.num_head, self.dim_head))

        v = self.v_projection(v.reshape((batch_size * keys_values_len, v_dim)))
        v = v.reshape((batch_size, keys_values_len, self.num_head, self.dim_head))

        # Reshape to (batch, heads, seq, dim_head)
        q = ops.transpose(q, axes=(1, 2))
        k = ops.transpose(k, axes=(1, 2))
        v = ops.transpose(v, axes=(1, 2))

        # Multi-head attention
        attn_out, _ = self.attn(q, k, v)

        # Reshape back: (B, heads, seq, dim) -> (B, seq, heads*dim)
        attn_out = ops.transpose(attn_out, axes=(1, 2))
        attn_out = attn_out.reshape((batch_size, queries_len, inner_dim))

        # Output projection
        result = self.out_projection(attn_out.reshape((batch_size * queries_len, inner_dim)))
        result = result.reshape((batch_size, queries_len, self.out_features))
        ### END YOUR SOLUTION

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout=0.0,
        causal=True,
        device=None,
        dtype="float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.attn = AttentionLayer(
            q_features, num_head, dim_head,
            dropout=dropout, causal=causal, device=device, dtype=dtype
        )
        self.dropout = Dropout(dropout)
        self.layernorm = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.ff = Sequential(
            Linear(q_features, hidden_size, device=device, dtype=dtype),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_size, q_features, device=device, dtype=dtype),
            Dropout(dropout),
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        # Self-attention sublayer with residual
        x = x + self.dropout(self.attn(x))

        # Feed-forward sublayer with residual (prenorm)
        y = self.layernorm(x.reshape((batch_size * seq_len, x_dim)))
        y = self.ff(y)
        x = x + y.reshape((batch_size, seq_len, x_dim))
        ### END YOUR SOLUTION

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout=0.0,
        causal=True,
        device=None,
        dtype="float32",
        batch_first=False,
        sequence_len=2048,
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        self.embedding = Embedding(
            sequence_len, embedding_size, device=device, dtype=dtype
        )
        self.layers = [
            TransformerLayer(
                embedding_size, num_head, dim_head, hidden_size,
                dropout=dropout, causal=causal, device=device, dtype=dtype
            )
            for _ in range(num_layers)
        ]
        self.sequence_len = sequence_len
        ### END YOUR SOLUTION

    def forward(self, x, h=None):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        batch_size, seq_len, embed_dim = x.shape
        pos_ids = Tensor(np.arange(seq_len), device=x.device)
        pos_embed = init.one_hot(
            self.sequence_len, pos_ids, device=x.device
        ) @ self.embedding.weight
        x = x + pos_embed.reshape((1, seq_len, embed_dim)).broadcast_to(x.shape)

        for layer in self.layers:
            x = layer(x)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)
