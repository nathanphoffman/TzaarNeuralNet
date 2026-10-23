"""Saving and resuming training: every accepted net, the near-miss nets, and the history of every generation.

After each generation the files in models/checkpoint/ are updated, so you can stop
training at any time and the next run continues from the last finished generation.
"""
import json
import shutil

from .model_store import MODELS_DIR, load_net, save_net

CHECKPOINT_DIR = MODELS_DIR / "checkpoint"
HISTORY_PATH = CHECKPOINT_DIR / "history.json"
DONOR_PATH = CHECKPOINT_DIR / "donor.npz"
RATING_BASIS = 2          # 1 = old ratings (the random player was averaged in), 2 = head-to-head games only


def _net_path(kind, generation):
    """kind is "net" (accepted) or "miss" (a near-miss that was rejected but kept as an opponent)."""
    return CHECKPOINT_DIR / f"{kind}_gen{generation:04d}.npz"


def save_checkpoint(history, rated_nets, near_misses, donor=None):
    """history: one row per generation. rated_nets / near_misses: [(generation, net, rating)],
    oldest first. donor: (generation, net, score vs best) or None, see training.py. Also keeps
    models/model.npz up to date with the best net."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    current = history[-1]["generation"]
    for kind, nets in (("net", rated_nets), ("miss", near_misses)):
        for generation, net, _ in nets:
            path = _net_path(kind, generation)
            # always rewrite the current generation's file: a stop mid-save could have left a stale one
            if generation == current or not path.exists():
                save_net(net, path)
    save_net(rated_nets[-1][1])
    if donor is None:
        DONOR_PATH.unlink(missing_ok=True)
    else:
        save_net(donor[1], DONOR_PATH)

    def meta(nets):
        return [{"generation": generation, "rating": rating} for generation, _, rating in nets]

    temp_path = HISTORY_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps({"rating_basis": RATING_BASIS, "history": history,
                                     "nets": meta(rated_nets), "near_misses": meta(near_misses),
                                     "donor": donor and {"generation": donor[0], "score": donor[2]}}))
    temp_path.replace(HISTORY_PATH)          # swap in one step so a stop mid-save can't corrupt it


def load_checkpoint():
    """Returns (history, rated_nets, near_misses, rating_basis, donor), or None if there is nothing to resume.
    Checkpoints from before near-misses, rating bases and donors existed load fine."""
    if not HISTORY_PATH.exists():
        return None
    saved = json.loads(HISTORY_PATH.read_text())

    def load(kind, entries):
        return [(n["generation"], load_net(_net_path(kind, n["generation"])), n["rating"]) for n in entries]

    donor = saved.get("donor")
    donor = donor and (donor["generation"], load_net(DONOR_PATH), donor["score"])
    return (saved["history"], load("net", saved["nets"]), load("miss", saved.get("near_misses", [])),
            saved.get("rating_basis", 1), donor)


def clear_checkpoint():
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
