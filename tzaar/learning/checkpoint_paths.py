"""Where a model's checkpoint files live: models/<name>/checkpoint/."""

from tzaar.nets.model_store import model_dir

CHECKPOINT_FOLDER = "checkpoint"

def checkpoint_dir(name=None):
    return model_dir(name) / CHECKPOINT_FOLDER

def history_path(name=None):
    return model_dir(name) / CHECKPOINT_FOLDER / "history.json"

def old_donor_path():
    return model_dir() / CHECKPOINT_FOLDER / "donor.npz"      # from when only one donor was kept

def donor_path(index):
    return model_dir() / CHECKPOINT_FOLDER / f"donor_{index}.npz"

def net_path(kind, generation, name=None):
    """kind is "net" (accepted) or "miss" (a near-miss that was rejected but kept as an opponent)."""

    return model_dir(name) / CHECKPOINT_FOLDER / f"{kind}_gen{generation:04d}.npz"
