"""Per-model settings: models/<name>/config.json.

An empty file (or none) means every setting is as it is in the code. Keys:
  "seed"     the random seed (default 0)
  "workers"  how many processes play games at the same time (default: games.WORKERS)
  anything in CAPITALS that training.py defines, e.g. {"MIN_SCORE_VS_BEST": 0.53}
Only this model's training process sees them, so models can't affect each other."""
import json

import numpy as np

from . import games, training
from .model_store import model_dir
from .rng import rng


def load_config(name):
    path = model_dir(name) / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def apply_config(config):
    for key, value in config.items():
        if key == "seed":
            rng.bit_generator.state = np.random.default_rng(value).bit_generator.state
        elif key == "workers":
            games.WORKERS = value
        elif key.isupper() and hasattr(training, key):
            setattr(training, key, value)
        else:
            raise ValueError(f"unknown setting in config.json: {key}")
