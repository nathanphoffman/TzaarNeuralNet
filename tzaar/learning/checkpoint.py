"""Saving and resuming training: every accepted net, the near-miss nets, and the history of every generation.

After each generation the files in models/<name>/checkpoint/ are updated, so you can stop
training at any time and the next run continues from the last finished generation.
"""

import json
import shutil

from tzaar.rules.encoding import NUM_FEATURES
from tzaar.learning.checkpoint_paths import checkpoint_dir, donor_path, history_path, net_path, old_donor_path
from tzaar.nets.model_store import load_net, model_path, save_net

GENERATION_KEY = "generation"      # the keys used in history.json and in the rows of the history
RATING_KEY = "rating"
NETS_KEY = "nets"
HISTORY_KEY = "history"
DONORS_KEY = "donors"
RATING_BASIS = 2          # 1 = old ratings (the random player was averaged in), 2 = head-to-head games only

def save_nets(history, rated_nets, near_misses):
    """Save every accepted and near-miss net that is not saved yet, and the best net as the model's model.npz."""

    checkpoint_dir().mkdir(parents=True, exist_ok=True)
    current = history[-1][GENERATION_KEY]

    for kind, nets in (("net", rated_nets), ("miss", near_misses)):
        for generation, network, _ in nets:
            path = net_path(kind, generation)

            # always rewrite the current generation's file: a stop mid-save could have left a stale one
            if generation == current or not path.exists():
                save_net(network, path)

    save_net(rated_nets[-1][1], model_path())

def save_donors(donors):
    """Save the donors as donor_0, donor_1, ... and delete the files of donors that are gone."""

    old_donor_path().unlink(missing_ok=True)

    for index in range(len(donors)):
        save_net(donors[index][1], donor_path(index))

    for stale in checkpoint_dir().glob("donor_*.npz"):
        if int(stale.stem.split("_")[1]) >= len(donors):
            stale.unlink()

def write_history(history, rated_nets, near_misses, donors):
    def meta(nets):
        return [{GENERATION_KEY: generation, RATING_KEY: rating} for generation, _, rating in nets]

    temp_path = history_path().with_suffix(".tmp")
    temp_path.write_text(json.dumps({"rating_basis": RATING_BASIS, HISTORY_KEY: history,
                                     NETS_KEY: meta(rated_nets), "near_misses": meta(near_misses),
                                     DONORS_KEY: [{GENERATION_KEY: generation, "score": score} for generation, _, score in donors],
                                     "donor_features": NUM_FEATURES}))

    temp_path.replace(history_path())          # swap in one step so a stop mid-save can't corrupt it

def save_checkpoint(history, rated_nets, near_misses, donors=()):
    """history: one row per generation. rated_nets / near_misses: [(generation, net, rating)],
    oldest first. donors: [(generation, net, score vs best)], oldest first, see training.py. Also keeps
    the model's model.npz up to date with the best net."""

    save_nets(history, rated_nets, near_misses)
    save_donors(donors)
    write_history(history, rated_nets, near_misses, donors)

def load_checkpoint():
    """Returns (history, rated_nets, near_misses, rating_basis, donors), or None if there is nothing to resume.
    Checkpoints from before near-misses, rating bases and donors existed load fine."""

    if not history_path().exists():
        return None

    saved = json.loads(history_path().read_text())

    def load_saved_nets(kind, entries):
        return [(entry[GENERATION_KEY], load_net(net_path(kind, entry[GENERATION_KEY])), entry[RATING_KEY])
                for entry in entries]

    entries = saved.get(DONORS_KEY) or ([saved["donor"]] if saved.get("donor") else [])
    paths = [donor_path(index) if DONORS_KEY in saved else old_donor_path() for index in range(len(entries))]
    donors = [(entry[GENERATION_KEY], load_net(path), entry["score"]) for entry, path in zip(entries, paths)]

    if donors and saved.get("donor_features") != NUM_FEATURES:
        # donors made before the inputs changed have zeros for the new ones, and their scores would block better donors
        print(f"The inputs changed since the {len(donors)} saved donors were made, so they were dropped")
        donors = []

    return (saved[HISTORY_KEY], load_saved_nets("net", saved[NETS_KEY]), load_saved_nets("miss", saved.get("near_misses", [])),
            saved.get("rating_basis", 1), donors)

def clear_checkpoint():
    shutil.rmtree(checkpoint_dir(), ignore_errors=True)
