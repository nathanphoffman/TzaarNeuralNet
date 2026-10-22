"""Saving and loading the trained network."""
from pathlib import Path

import numpy as np

from .network import Net

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.npz"


def save_net(net, path=MODEL_PATH):
    path.parent.mkdir(exist_ok=True)
    np.savez(path, W1=net.W1, b1=net.b1, W2=net.W2, b2=net.b2)


def load_net(path=MODEL_PATH):
    saved = np.load(path)
    net = Net()
    net.W1, net.b1, net.W2, net.b2 = saved["W1"], saved["b1"], saved["W2"], saved["b2"]
    return net
