"""Saving and loading the trained networks. Each model lives in its own folder, models/<name>/.

A training process works on one model: call use_model(name) once at the start and everything that
saves or loads "the model" (checkpoint.py, save_net, load_net) then uses that model's folder."""

import os
from pathlib import Path

import numpy as np

from tzaar.rules.encoding import NUM_FEATURES
from tzaar.nets.network import Network

MODELS_ROOT = Path(os.environ.get("TZAAR_MODELS_ROOT") or Path(__file__).resolve().parent.parent.parent / "models")   # the env var lets an experiment run on a copy

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

    return sorted(folder.name for folder in MODELS_ROOT.iterdir() if folder.is_dir())

def save_net(network, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    second_layer = {} if network.V is None else {"V": network.V, "c": network.c, "U": network.U}
    np.savez(path, W1=network.W1, b1=network.b1, W2=network.W2, b2=network.b2, **second_layer)

def load_net(path):
    """Nets saved before new inputs were added have fewer rows in W1. The missing rows are filled with
    zeros, so such a net ignores the new inputs and plays exactly as it did, and can learn to use them later.
    Nets saved before the second hidden layer existed load without one."""

    saved = np.load(path)
    network = Network(hidden=saved["W1"].shape[1])
    first_layer = saved["W1"]

    if first_layer.shape[0] < NUM_FEATURES:
        first_layer = np.vstack([first_layer, np.zeros((NUM_FEATURES - first_layer.shape[0], first_layer.shape[1]), dtype=first_layer.dtype)])

    network.W1, network.b1, network.W2, network.b2 = first_layer, saved["b1"], saved["W2"], saved["b2"]
    network.V, network.c, network.U = (saved[name] if name in saved else None for name in ("V", "c", "U"))

    return network
