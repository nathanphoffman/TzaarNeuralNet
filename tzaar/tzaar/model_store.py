"""Saving and loading the trained network."""
from pathlib import Path

import numpy as np

from .encoding import NUM_FEATURES
from .network import Net

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "model.npz"      # the best net so far, used by the game server


def save_net(net, path=MODEL_PATH):
    path.parent.mkdir(exist_ok=True)
    np.savez(path, W1=net.W1, b1=net.b1, W2=net.W2, b2=net.b2)


def load_net(path=MODEL_PATH):
    """Nets saved before new inputs were added have fewer rows in W1. The missing rows are filled with
    zeros, so such a net ignores the new inputs and plays exactly as it did, and can learn to use them later."""
    saved = np.load(path)
    net = Net(hidden=saved["W1"].shape[1])
    W1 = saved["W1"]
    if W1.shape[0] < NUM_FEATURES:
        W1 = np.vstack([W1, np.zeros((NUM_FEATURES - W1.shape[0], W1.shape[1]), dtype=W1.dtype)])
    net.W1, net.b1, net.W2, net.b2 = W1, saved["b1"], saved["W2"], saved["b2"]
    return net
