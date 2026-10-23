"""A small neural network that scores a Tzaar position."""
import numpy as np

from .encoding import NUM_FEATURES
from .rng import rng


HIDDEN = 256       # neurons in the hidden layer for new nets; older, narrower nets are widened to this (see widened)


class Net:
    """inputs -> hidden -> 1 output (score from -1 = losing to +1 = winning)."""

    def __init__(self, hidden=HIDDEN):
        self.W1 = rng.normal(0, 0.1, (NUM_FEATURES, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, 0.3, (hidden, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

    def copy(self):
        """An independent copy with the same weights (used to warm-start the next generation)."""
        other = Net.__new__(Net)
        other.W1, other.b1, other.W2, other.b2 = self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()
        return other

    def widened(self, hidden=HIDDEN):
        """A copy with more hidden neurons that scores every position exactly as this net does. The new
        neurons start with zero weights from the inputs, so they output 0 and change nothing, and small random
        weights to the output, so training can start using them."""
        extra = hidden - self.W1.shape[1]
        if extra <= 0:
            return self.copy()
        other = Net.__new__(Net)
        other.W1 = np.hstack([self.W1, np.zeros((NUM_FEATURES, extra), dtype=np.float32)])
        other.b1 = np.concatenate([self.b1, np.zeros(extra, dtype=np.float32)])
        other.W2 = np.vstack([self.W2, rng.normal(0, 0.1, (extra, 1)).astype(np.float32)])
        other.b2 = self.b2.copy()
        return other

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
