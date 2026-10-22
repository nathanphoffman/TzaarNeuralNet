"""A small neural network that scores a Tzaar position."""
import numpy as np

from .encoding import NUM_FEATURES
from .rng import rng


class Net:
    """inputs -> hidden -> 1 output (score from -1 = losing to +1 = winning)."""

    def __init__(self, hidden=64):
        self.W1 = rng.normal(0, 0.1, (NUM_FEATURES, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, 0.3, (hidden, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

    def predict(self, X):
        self.hidden = np.tanh(X @ self.W1 + self.b1)
        return np.tanh(self.hidden @ self.W2 + self.b2).ravel()

    def train(self, X, y, epochs=15, learning_rate=0.05, batch_size=64):
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                batch = order[start:start + batch_size]
                xb, yb = X[batch], y[batch]
                out = self.predict(xb)
                # backpropagation: nudge weights to shrink (out - yb)^2
                d_out = ((out - yb) * (1 - out ** 2))[:, None] / len(xb)
                d_hidden = (d_out @ self.W2.T) * (1 - self.hidden ** 2)
                self.W2 -= learning_rate * self.hidden.T @ d_out; self.b2 -= learning_rate * d_out.sum(0)
                self.W1 -= learning_rate * xb.T @ d_hidden;       self.b1 -= learning_rate * d_hidden.sum(0)
