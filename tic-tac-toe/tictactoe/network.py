"""A tiny neural network that scores a board position."""
import numpy as np

from .rng import rng

class Net:
    """27 inputs -> 64 hidden -> 1 output (score from -1 = losing to +1 = winning)."""

    def __init__(self):
        self.W1 = rng.normal(0, 0.3, (27, 64)); self.b1 = np.zeros(64)
        self.W2 = rng.normal(0, 0.3, (64, 1));  self.b2 = np.zeros(1)

    def predict(self, X):
        self.hidden = np.tanh(X @ self.W1 + self.b1)

        return np.tanh(self.hidden @ self.W2 + self.b2).ravel()

    def train(self, X, y, epochs=30, learning_rate=0.05, batch_size=64):
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
