"""Per-model settings: models/<name>/config.json.

An empty file (or none) means every setting is as it is in the code. Keys:
  "seed"     the random seed (default 0)
  "workers"  how many processes play games at the same time (default: games.WORKERS)
  anything in CAPITALS that settings.py defines, e.g. {"MIN_SCORE_VS_BEST": 0.53}
Only this model's training process sees them, so models can't affect each other."""

import json

import numpy as np

from tzaar.learning import settings
from tzaar.nets import games
from tzaar.nets.model_store import model_dir
from tzaar.rules.rng import random_generator

def load_config(name):
    path = model_dir(name) / "config.json"

    return json.loads(path.read_text()) if path.exists() else {}

def apply_config(config):
    """Apply one model's settings in this process. An unknown name is an error, so a typo can't silently do nothing."""

    for setting, value in config.items():
        if setting == "seed":
            random_generator.bit_generator.state = np.random.default_rng(value).bit_generator.state
        elif setting == "workers":
            games.WORKERS = value
        elif setting.isupper() and hasattr(settings, setting):
            setattr(settings, setting, value)
        else:
            raise ValueError(f"unknown setting in config.json: {setting}")
