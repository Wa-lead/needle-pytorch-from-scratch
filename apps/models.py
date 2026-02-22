import sys

sys.path.append("./python")
import needle as ndl
import needle.nn as nn
import numpy as np


class ResNet9(nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv(3, 16, 7, stride=4, device=device, dtype=dtype),
            nn.BatchNorm2d(16, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Conv(16, 32, 3, stride=2, device=device, dtype=dtype),
            nn.BatchNorm2d(32, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Residual(
                nn.Sequential(
                    nn.Conv(32, 32, 3, stride=1, device=device, dtype=dtype),
                    nn.BatchNorm2d(32, device=device, dtype=dtype),
                    nn.ReLU(),
                    nn.Conv(32, 32, 3, stride=1, device=device, dtype=dtype),
                    nn.BatchNorm2d(32, device=device, dtype=dtype),
                    nn.ReLU(),
                )
            ),
            nn.Conv(32, 64, 3, stride=2, device=device, dtype=dtype),
            nn.BatchNorm2d(64, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Conv(64, 128, 3, stride=2, device=device, dtype=dtype),
            nn.BatchNorm2d(128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Residual(
                nn.Sequential(
                    nn.Conv(128, 128, 3, stride=1, device=device, dtype=dtype),
                    nn.BatchNorm2d(128, device=device, dtype=dtype),
                    nn.ReLU(),
                    nn.Conv(128, 128, 3, stride=1, device=device, dtype=dtype),
                    nn.BatchNorm2d(128, device=device, dtype=dtype),
                    nn.ReLU(),
                )
            ),
            nn.Flatten(),
            nn.Linear(128, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, device=device, dtype=dtype),
        )

    def forward(self, x):
        return self.model(x)


class LanguageModel(nn.Module):
    def __init__(
        self,
        embedding_size,
        output_size,
        hidden_size,
        num_layers=1,
        seq_model="lstm",
        device=None,
        dtype="float32",
    ):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.seq_model = seq_model

        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)

        if seq_model == "rnn":
            self.rnn = nn.RNN(
                embedding_size, hidden_size, num_layers, device=device, dtype=dtype
            )
        elif seq_model == "lstm":
            self.rnn = nn.LSTM(
                embedding_size, hidden_size, num_layers, device=device, dtype=dtype
            )

        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)

    def forward(self, x, h=None):
        """
        Args:
            x: shape (seq_len, batch_size) - integer token indices
            h: hidden state (for RNN) or (h, c) tuple (for LSTM)

        Returns:
            output: shape (batch_size * seq_len, output_size) - logits
            h: hidden state
        """
        seq_len, batch_size = x.shape
        emb = self.embedding(x)  # (seq_len, batch_size, embedding_size)

        output, h_out = self.rnn(emb, h)  # output: (seq_len, bs, hidden_size)

        output = output.reshape((seq_len * batch_size, self.hidden_size))
        output = self.linear(output)  # (seq_len * batch_size, output_size)

        return output, h_out
