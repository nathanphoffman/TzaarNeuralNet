"""Saving and resuming training: every accepted net plus the history of every generation.

After each generation the files in models/checkpoint/ are updated, so you can stop
training at any time and the next run continues from the last finished generation.
"""
import json
import shutil

from .model_store import MODELS_DIR, load_net, save_net

CHECKPOINT_DIR = MODELS_DIR / "checkpoint"
HISTORY_PATH = CHECKPOINT_DIR / "history.json"


def _net_path(generation):
    return CHECKPOINT_DIR / f"net_gen{generation:04d}.npz"


def save_checkpoint(history, rated_nets):
    """history: one row per generation. rated_nets: [(generation, net, rating)], oldest first.
    Also keeps models/model.npz up to date with the best net."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for generation, net, _ in rated_nets:
        if not _net_path(generation).exists():
            save_net(net, _net_path(generation))
    save_net(rated_nets[-1][1])
    nets = [{"generation": generation, "rating": rating} for generation, _, rating in rated_nets]
    temp_path = HISTORY_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps({"history": history, "nets": nets}))
    temp_path.replace(HISTORY_PATH)          # swap in one step so a stop mid-save can't corrupt it


def load_checkpoint():
    """Returns (history, rated_nets), or None if there is nothing to resume."""
    if not HISTORY_PATH.exists():
        return None
    saved = json.loads(HISTORY_PATH.read_text())
    rated_nets = [(n["generation"], load_net(_net_path(n["generation"])), n["rating"])
                  for n in saved["nets"]]
    return saved["history"], rated_nets


def clear_checkpoint():
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
