"""Saving and resuming training: every accepted net, the near-miss nets, and the history of every generation.

After each generation the files in models/<name>/checkpoint/ are updated, so you can stop
training at any time and the next run continues from the last finished generation.
"""
import json
import shutil

from .encoding import NUM_FEATURES
from .model_store import load_net, model_dir, save_net

RATING_BASIS = 2          # 1 = old ratings (the random player was averaged in), 2 = head-to-head games only


def checkpoint_dir(name=None):
    return model_dir(name) / "checkpoint"


def history_path(name=None):
    return checkpoint_dir(name) / "history.json"


def _old_donor_path():
    return checkpoint_dir() / "donor.npz"      # from when only one donor was kept


def _donor_path(index):
    return checkpoint_dir() / f"donor_{index}.npz"


def _net_path(kind, generation):
    """kind is "net" (accepted) or "miss" (a near-miss that was rejected but kept as an opponent)."""
    return checkpoint_dir() / f"{kind}_gen{generation:04d}.npz"


def save_checkpoint(history, rated_nets, near_misses, donors=()):
    """history: one row per generation. rated_nets / near_misses: [(generation, net, rating)],
    oldest first. donors: [(generation, net, score vs best)], oldest first, see training.py. Also keeps
    the model's model.npz up to date with the best net."""
    checkpoint_dir().mkdir(parents=True, exist_ok=True)
    current = history[-1]["generation"]
    for kind, nets in (("net", rated_nets), ("miss", near_misses)):
        for generation, net, _ in nets:
            path = _net_path(kind, generation)
            # always rewrite the current generation's file: a stop mid-save could have left a stale one
            if generation == current or not path.exists():
                save_net(net, path)
    save_net(rated_nets[-1][1])
    _old_donor_path().unlink(missing_ok=True)
    for index in range(len(donors)):
        save_net(donors[index][1], _donor_path(index))
    for stale in checkpoint_dir().glob("donor_*.npz"):
        if int(stale.stem.split("_")[1]) >= len(donors):
            stale.unlink()

    def meta(nets):
        return [{"generation": generation, "rating": rating} for generation, _, rating in nets]

    temp_path = history_path().with_suffix(".tmp")
    temp_path.write_text(json.dumps({"rating_basis": RATING_BASIS, "history": history,
                                     "nets": meta(rated_nets), "near_misses": meta(near_misses),
                                     "donors": [{"generation": g, "score": score} for g, _, score in donors],
                                     "donor_features": NUM_FEATURES}))
    temp_path.replace(history_path())          # swap in one step so a stop mid-save can't corrupt it


def load_checkpoint():
    """Returns (history, rated_nets, near_misses, rating_basis, donors), or None if there is nothing to resume.
    Checkpoints from before near-misses, rating bases and donors existed load fine."""
    if not history_path().exists():
        return None
    saved = json.loads(history_path().read_text())

    def load(kind, entries):
        return [(n["generation"], load_net(_net_path(kind, n["generation"])), n["rating"]) for n in entries]

    entries = saved.get("donors") or ([saved["donor"]] if saved.get("donor") else [])
    paths = [_donor_path(i) if "donors" in saved else _old_donor_path() for i in range(len(entries))]
    donors = [(e["generation"], load_net(path), e["score"]) for e, path in zip(entries, paths)]
    if donors and saved.get("donor_features") != NUM_FEATURES:
        # donors made before the inputs changed have zeros for the new ones, and their scores would block better donors
        print(f"The inputs changed since the {len(donors)} saved donors were made, so they were dropped")
        donors = []
    return (saved["history"], load("net", saved["nets"]), load("miss", saved.get("near_misses", [])),
            saved.get("rating_basis", 1), donors)


def clear_checkpoint():
    shutil.rmtree(checkpoint_dir(), ignore_errors=True)
