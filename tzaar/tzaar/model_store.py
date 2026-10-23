"""Saving and loading the trained networks. Each model lives in its own folder, models/<name>/.

A training process works on one model: call use_model(name) once at the start and everything that
saves or loads "the model" (checkpoint.py, save_net, load_net) then uses that model's folder."""
from pathlib import Path

import numpy as np

from .encoding import NUM_FEATURES
from .network import Net

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"

_current = None


def use_model(name):
    global _current
    _current = name


def model_dir(name=None):
    name = name or _current
    if name is None:
        raise RuntimeError("no model selected: call use_model(name) first")
    return MODELS_ROOT / name


def model_path(name=None):
    """The best net so far for a model."""
    return model_dir(name) / "model.npz"


def list_models():
    """Names of the models that exist: every folder in models/."""
    if not MODELS_ROOT.is_dir():
        return []
    return sorted(d.name for d in MODELS_ROOT.iterdir() if d.is_dir())


def save_net(net, path=None):
    path = path or model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    second_layer = {} if net.V is None else {"V": net.V, "c": net.c, "U": net.U}
    np.savez(path, W1=net.W1, b1=net.b1, W2=net.W2, b2=net.b2, **second_layer)


def load_net(path=None):
    """Nets saved before new inputs were added have fewer rows in W1. The missing rows are filled with
    zeros, so such a net ignores the new inputs and plays exactly as it did, and can learn to use them later.
    Nets saved before the second hidden layer existed load without one."""
    saved = np.load(path or model_path())
    net = Net(hidden=saved["W1"].shape[1])
    W1 = saved["W1"]
    if W1.shape[0] < NUM_FEATURES:
        W1 = np.vstack([W1, np.zeros((NUM_FEATURES - W1.shape[0], W1.shape[1]), dtype=W1.dtype)])
    net.W1, net.b1, net.W2, net.b2 = W1, saved["b1"], saved["W2"], saved["b2"]
    net.V, net.c, net.U = (saved[name] if name in saved else None for name in ("V", "c", "U"))
    return net
