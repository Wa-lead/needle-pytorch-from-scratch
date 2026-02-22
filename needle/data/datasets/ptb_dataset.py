import os
import numpy as np


class Dictionary:
    """Maps words to integer indices and back."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    """Tokenizes a Penn Treebank-style text corpus."""

    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, "train.txt"), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, "test.txt"), max_lines)

    def tokenize(self, path, max_lines=None):
        with open(path, "r") as f:
            lines = f.readlines()
            if max_lines is not None:
                lines = lines[:max_lines]

        tokens = 0
        for line in lines:
            words = line.split() + ["<eos>"]
            tokens += len(words)
            for word in words:
                self.dictionary.add_word(word)

        ids = np.zeros(tokens, dtype=np.int32)
        pos = 0
        for line in lines:
            words = line.split() + ["<eos>"]
            for word in words:
                ids[pos] = self.dictionary.word2idx[word]
                pos += 1

        return ids


def batchify(data, batch_size, device, dtype):
    """Reshape data into (num_batches, batch_size) and convert to Tensor."""
    import needle as ndl

    num_batches = len(data) // batch_size
    data = data[: num_batches * batch_size]
    data = data.reshape((batch_size, num_batches)).T
    return ndl.Tensor(data, device=device, dtype=dtype)


def get_batch(batches, i, bptt, device=None, dtype=None):
    """Extract input/target sequences of length bptt starting at position i.

    Returns:
        data: shape (bptt, batch_size)
        target: shape (bptt * batch_size,)
    """
    import needle as ndl

    seq_len = min(bptt, len(batches) - 1 - i)
    data = batches[i : i + seq_len]
    target = batches[i + 1 : i + 1 + seq_len]
    data = ndl.Tensor(data.numpy().astype(np.float32), device=device, dtype=dtype)
    target = ndl.Tensor(target.numpy().astype(np.float32).reshape(-1), device=device, dtype=dtype)
    return data, target
