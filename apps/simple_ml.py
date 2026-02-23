import struct
import gzip
import numpy as np
import sys

sys.path.append("./python")
import needle as ndl


def add(x, y):
    return x + y


def parse_mnist(image_filename, label_filename):
    with gzip.open(image_filename, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols).astype(np.float32)
        X /= 255.0

    with gzip.open(label_filename, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(), dtype=np.uint8)

    return X, y


def softmax_loss(Z, y):
    """Compute softmax loss.

    Works with both numpy arrays (y = integer labels) and
    ndl.Tensor (y = one-hot encoded labels).
    """
    if isinstance(Z, ndl.Tensor):
        log_sum_exp = ndl.ops.logsumexp(Z, axes=(1,))
        z_y = (Z * y).sum(axes=(1,))
        return (log_sum_exp - z_y).sum() / Z.shape[0]
    else:
        n = Z.shape[0]
        max_Z = np.max(Z, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(Z - max_Z), axis=1)) + max_Z.flatten()
        z_y = Z[np.arange(n), y]
        return np.mean(log_sum_exp - z_y)


def softmax_regression_epoch(X, y, theta, lr=0.1, batch=100):
    """Run a single epoch of softmax regression with SGD on numpy arrays.

    Modifies theta in-place.
    """
    n = X.shape[0]
    for i in range(0, n, batch):
        X_b = X[i : i + batch]
        y_b = y[i : i + batch]
        m = X_b.shape[0]

        Z = X_b @ theta
        max_Z = np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z - max_Z)
        probs = exp_Z / exp_Z.sum(axis=1, keepdims=True)

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(m), y_b] = 1

        grad = X_b.T @ (probs - one_hot) / m
        theta -= lr * grad


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of a two-layer neural network.

    If W1 is an ndl.Tensor, uses autograd and returns (W1, W2).
    If W1 is a numpy array, modifies W1, W2 in-place.
    """
    if isinstance(W1, ndl.Tensor):
        return _nn_epoch_ndl(X, y, W1, W2, lr, batch)
    else:
        _nn_epoch_numpy(X, y, W1, W2, lr, batch)


def _nn_epoch_ndl(X, y, W1, W2, lr, batch):
    n = X.shape[0]
    num_classes = W2.shape[1]
    for i in range(0, n, batch):
        X_b = ndl.Tensor(X[i : i + batch])
        y_b = y[i : i + batch]
        y_one_hot = np.zeros((len(y_b), num_classes))
        y_one_hot[np.arange(len(y_b)), y_b] = 1
        y_t = ndl.Tensor(y_one_hot)

        Z1 = ndl.relu(X_b @ W1)
        loss = softmax_loss(Z1 @ W2, y_t)
        loss.backward()

        W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
        W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())
    return W1, W2


def _nn_epoch_numpy(X, y, W1, W2, lr, batch):
    n = X.shape[0]
    for i in range(0, n, batch):
        X_b = X[i : i + batch]
        y_b = y[i : i + batch]
        m = X_b.shape[0]

        # Forward
        Z1 = X_b @ W1
        A1 = np.maximum(Z1, 0)  # ReLU
        Z2 = A1 @ W2

        # Softmax
        max_Z = np.max(Z2, axis=1, keepdims=True)
        exp_Z = np.exp(Z2 - max_Z)
        probs = exp_Z / exp_Z.sum(axis=1, keepdims=True)

        # One-hot
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(m), y_b] = 1

        # Backward
        dZ2 = (probs - one_hot) / m
        dW2 = A1.T @ dZ2
        dA1 = dZ2 @ W2.T
        dZ1 = dA1.copy()
        dZ1[Z1 <= 0] = 0
        dW1 = X_b.T @ dZ1

        W1 -= lr * dW1
        W2 -= lr * dW2


def loss_err(h, y):
    """Compute softmax loss and classification error.

    Args:
        h: logits, either numpy array or ndl.Tensor of shape (n, k)
        y: integer labels, numpy array of shape (n,)

    Returns:
        (loss, error_rate)
    """
    if isinstance(h, ndl.Tensor):
        h_np = h.numpy()
    else:
        h_np = h
    loss = softmax_loss(h_np, y)
    err = np.mean(np.argmax(h_np, axis=1) != y)
    return loss, err


def train_ptb(model, data, seq_len=40, n_epochs=1, device=None, lr=0.001, optimizer=ndl.optim.SGD, verbose=True):
    """Train a language model on PTB data."""
    import time
    np.random.seed(4)
    loss_fn = ndl.nn.SoftmaxLoss()
    opt = optimizer(model.parameters(), lr=lr)

    nbatch = (len(data) - 1) // seq_len
    for epoch in range(n_epochs):
        epoch_start = time.time()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        hidden = None
        batch_idx = 0
        for i in range(0, len(data) - 1, seq_len):
            X, y = ndl.data.get_batch(data, i, seq_len, device=device)
            batch_size = X.shape[1]
            seq = X.shape[0]
            opt.reset_grad()

            out, hidden = model(X, hidden)
            if isinstance(hidden, tuple):
                hidden = tuple(h.detach() for h in hidden)
            else:
                hidden = hidden.detach()
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            total_loss += loss.data.numpy() * y.shape[0]
            total_correct += (np.argmax(out.numpy(), axis=1) == y.numpy()).sum()
            total_samples += y.shape[0]
            batch_idx += 1

            if verbose and batch_idx % max(1, nbatch // 5) == 0:
                avg = float(total_loss / total_samples)
                acc = float(total_correct / total_samples)
                print(f"  [{batch_idx}/{nbatch}] loss={avg:.4f} acc={acc:.4f}")

        elapsed = time.time() - epoch_start
        avg_loss = float(total_loss / total_samples)
        accuracy = float(total_correct / total_samples)
        if verbose:
            print(f"Epoch {epoch+1}/{n_epochs} - loss: {avg_loss:.4f}, acc: {accuracy:.4f}, time: {elapsed:.1f}s")

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return accuracy, avg_loss


def evaluate_ptb(model, data, seq_len=40, device=None):
    """Evaluate a language model on PTB data."""
    loss_fn = ndl.nn.SoftmaxLoss()
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    hidden = None
    for i in range(0, len(data) - 1, seq_len):
        X, y = ndl.data.get_batch(data, i, seq_len, device=device)
        out, hidden = model(X, hidden)
        if isinstance(hidden, tuple):
            hidden = tuple(h.detach() for h in hidden)
        else:
            hidden = hidden.detach()
        loss = loss_fn(out, y)

        total_loss += loss.data.numpy() * y.shape[0]
        total_correct += (np.argmax(out.numpy(), axis=1) == y.numpy()).sum()
        total_samples += y.shape[0]

    model.train()
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return accuracy, avg_loss


def generate_ptb(model, corpus, prompt, max_len=50, temperature=1.0, device=None):
    """Generate text from a trained language model.

    Args:
        model: trained LanguageModel
        corpus: the Corpus object (for word<->index mapping)
        prompt: string of space-separated words to seed generation
        max_len: number of tokens to generate
        temperature: sampling temperature (lower = more greedy, higher = more random)
        device: device to run on

    Returns:
        generated string
    """
    model.eval()
    dictionary = corpus.dictionary

    words = prompt.strip().split()
    token_ids = []
    for w in words:
        if w in dictionary.word2idx:
            token_ids.append(dictionary.word2idx[w])
        else:
            token_ids.append(dictionary.word2idx.get("<unk>", 0))

    hidden = None
    generated = list(words)

    # Feed the prompt through the model to build up hidden state
    if len(token_ids) > 1:
        prompt_input = np.array(token_ids[:-1], dtype=np.float32).reshape(-1, 1)
        X = ndl.Tensor(prompt_input, device=device)
        out, hidden = model(X, hidden)
        if isinstance(hidden, tuple):
            hidden = tuple(h.detach() for h in hidden)
        elif hidden is not None:
            hidden = hidden.detach()

    # Generate tokens one at a time
    current_token = token_ids[-1]
    for _ in range(max_len):
        X = ndl.Tensor(
            np.array([[current_token]], dtype=np.float32), device=device
        )  # shape (1, 1)
        out, hidden = model(X, hidden)
        if isinstance(hidden, tuple):
            hidden = tuple(h.detach() for h in hidden)
        elif hidden is not None:
            hidden = hidden.detach()

        logits = out.numpy().flatten()

        # Temperature-scaled sampling
        if temperature <= 0:
            current_token = int(np.argmax(logits))
        else:
            logits = logits / temperature
            logits -= np.max(logits)
            probs = np.exp(logits) / np.sum(np.exp(logits))
            current_token = int(np.random.choice(len(probs), p=probs))

        word = dictionary.idx2word[current_token]
        generated.append(word)

        if word == "<eos>":
            break

    model.train()
    return " ".join(generated)
