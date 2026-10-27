"""A small neural network that scores a Tzaar position."""
import numpy as np

from tzaar.learning import settings
from tzaar.rules.encoding import NUM_FEATURES
from tzaar.rules.rng import random_generator

SECOND_LAYER_RATE = 0.1   # the second layer learns this much slower than the rest, so it can't unsettle a tuned net
INPUT_WEIGHT_SPREAD = 0.1     # standard deviation of the random starting weights (first layer, second layer, new neurons)
OUTPUT_WEIGHT_SPREAD = 0.3    # ...and of the weights to the output, which start larger so the first outputs differ
DEFAULT_EPOCHS = 15
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_BATCH_SIZE = 64
HIDDEN2 = 64       # neurons in the second hidden layer; older nets without one get it added (see deepened)

class Network:
    """inputs -> hidden -> 1 output (score from -1 = losing to +1 = winning).
    A net can also have a second hidden layer (V, c, U) that sits beside the direct hidden -> output path:
        hidden2 = tanh(hidden @ V + c)
        output  = tanh(hidden @ W2 + hidden2 @ U + b2)
    Nets without it (V is None) just use the direct path."""

    def __init__(self, hidden=None):
        hidden = hidden or settings.HIDDEN_NEURONS
        self.W1 = random_generator.normal(0, INPUT_WEIGHT_SPREAD, (NUM_FEATURES, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = random_generator.normal(0, OUTPUT_WEIGHT_SPREAD, (hidden, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        self.V = self.c = self.U = None
        self.add_second_layer(self)

    @staticmethod
    def add_second_layer(network, hidden2=HIDDEN2):
        network.V = random_generator.normal(0, INPUT_WEIGHT_SPREAD, (network.W1.shape[1], hidden2)).astype(np.float32)
        network.c = np.zeros(hidden2, dtype=np.float32)
        network.U = np.zeros((hidden2, 1), dtype=np.float32)

    def copy(self):
        """An independent copy with the same weights (used to warm-start the next generation)."""
        other = Network.__new__(Network)
        other.W1, other.b1, other.W2, other.b2 = self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()
        other.V, other.c, other.U = (None if weights is None else weights.copy() for weights in (self.V, self.c, self.U))

        return other

    def widened(self, hidden=None):
        """A copy with more hidden neurons (settings.HIDDEN_NEURONS if not said) that scores every position exactly as this net does. The new
        neurons start with zero weights from the inputs, so they output 0 and change nothing, and small random
        weights to the output, so training can start using them."""

        hidden = hidden or settings.HIDDEN_NEURONS
        extra = hidden - self.W1.shape[1]

        if extra <= 0:
            return self.copy()

        other = self.copy()
        other.W1 = np.hstack([self.W1, np.zeros((NUM_FEATURES, extra), dtype=np.float32)])
        other.b1 = np.concatenate([self.b1, np.zeros(extra, dtype=np.float32)])
        other.W2 = np.vstack([self.W2, random_generator.normal(0, INPUT_WEIGHT_SPREAD, (extra, 1)).astype(np.float32)])

        if self.V is not None:
            other.V = np.vstack([self.V, np.zeros((extra, self.V.shape[1]), dtype=np.float32)])

        return other

    def deepened(self, hidden2=HIDDEN2):
        """A copy with a second hidden layer added that scores every position exactly as this net does:
        its weights to the output (U) start at zero, so it adds nothing yet, and training can then switch it on."""

        other = self.copy()

        if other.V is None:
            self.add_second_layer(other, hidden2)

        return other

    def predict(self, inputs):
        self.hidden = np.tanh(inputs @ self.W1 + self.b1)
        pre_activation = self.hidden @ self.W2 + self.b2

        if self.V is not None:
            self.hidden2 = np.tanh(self.hidden @ self.V + self.c)
            pre_activation = pre_activation + self.hidden2 @ self.U

        return np.tanh(pre_activation).ravel()

    def train(self, inputs, targets, epochs=DEFAULT_EPOCHS, learning_rate=DEFAULT_LEARNING_RATE, batch_size=DEFAULT_BATCH_SIZE):
        for _ in range(epochs):
            order = random_generator.permutation(len(inputs))

            for start in range(0, len(inputs), batch_size):
                batch = order[start:start + batch_size]
                batch_inputs, batch_targets = inputs[batch], targets[batch]
                output = self.predict(batch_inputs)

                # backpropagation: nudge weights to shrink (out - yb)^2
                d_out = ((output - batch_targets) * (1 - output ** 2))[:, None] / len(batch_inputs)
                d_hidden = d_out @ self.W2.T

                if self.V is not None:
                    d_hidden2 = (d_out @ self.U.T) * (1 - self.hidden2 ** 2)
                    d_hidden = d_hidden + d_hidden2 @ self.V.T

                d_hidden = d_hidden * (1 - self.hidden ** 2)

                if self.V is not None:                  # (the gradients above all used the weights from before this step)
                    rate = learning_rate * SECOND_LAYER_RATE
                    self.U -= rate * self.hidden2.T @ d_out
                    self.V -= rate * self.hidden.T @ d_hidden2; self.c -= rate * d_hidden2.sum(0)

                self.W2 -= learning_rate * self.hidden.T @ d_out; self.b2 -= learning_rate * d_out.sum(0)
                self.W1 -= learning_rate * batch_inputs.T @ d_hidden;       self.b1 -= learning_rate * d_hidden.sum(0)
