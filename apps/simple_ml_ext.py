"""Python fallback for the C++ softmax regression extension."""
import numpy as np


def softmax_regression_epoch_cpp(X, y, theta, lr=0.1, batch=100):
    """Run a single epoch of softmax regression with SGD.

    Equivalent to the C++ extension from HW0.
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
