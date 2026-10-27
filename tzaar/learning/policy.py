"""The rules training follows: how much to play and train at each stage, who counts as a worthy opponent,
and how to tell a narrow fail from a clear one."""

import functools
from pathlib import Path

import numpy as np

from tzaar.learning import settings
from tzaar.nets import model_store
from tzaar.nets.model_store import load_net
from tzaar.nets.players import make_net_player
from tzaar.nets.search import make_search_player
from tzaar.rules.encoding import SIGNALS_START

BASELINES = Path(__file__).resolve().parent.parent.parent / "baselines"

PROJECT_ROOT = BASELINES.parent

@functools.lru_cache(maxsize=None)
def _load_partner(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"SPARRING_PARTNER: no net at {path}")

    return load_net(path)

def sparring_partner():
    """The net named by SPARRING_PARTNER (loaded once), or None if there is none."""

    name = settings.SPARRING_PARTNER

    if not name:
        return None

    path = PROJECT_ROOT / name if name.endswith(".npz") else model_store.MODELS_ROOT / name / "model.npz"

    return _load_partner(str(path))

def epochs_for(rating):
    """A weak net learns a lot from each pass over the data, a strong one mostly picks up noise."""

    return max(1, round(settings.WARM_START_EPOCHS_AT_ZERO - rating / settings.WARM_START_ELO_PER_EPOCH))

def games_for(generation, max_games):
    """Early generations need little data and big gaps are easy to spot, so start small and grow."""

    return min(max_games, settings.START_GAMES + settings.GAMES_PER_GENERATION * generation)

def is_borderline(score_vs_best, rating, best_rating, score_vs_random):
    """A rejected net worth a closer look: one of the two checks passes and the other only narrowly fails."""

    if score_vs_random < settings.MIN_SCORE_VS_RANDOM:
        return False

    strong_rating_but_even = rating >= best_rating + settings.MIN_RATING_GAIN and score_vs_best >= settings.RETEST_MIN_SCORE
    wins_but_rating_short = (score_vs_best >= settings.MIN_SCORE_VS_BEST
                             and rating >= best_rating + settings.MIN_RATING_GAIN - settings.RETEST_RATING_SLACK)

    return strong_rating_but_even or wins_but_rating_short

def self_play_opponents(rated_nets, near_misses):
    """The best net plays each self-play game against a net drawn from the best net itself, the older
    accepted nets, and the near-miss nets (the best net looks ahead if SELF_PLAY_SEARCH_TURNS is set). The best net is the most likely, then the others newest
    first, each step back less likely, so really old nets only turn up now and then.
    Returns (the best net as an explorer, opponent players, their probabilities, labels)."""

    best_generation, best_net, _ = rated_nets[-1]
    others = ([(generation, network, False) for generation, network, _ in rated_nets[:-1]]
              + [(generation, network, True) for generation, network, _ in near_misses])

    others.sort(key=lambda entry: entry[0], reverse=True)
    entries = [(best_generation, best_net, False)] + others
    weights = np.array([settings.OLDER_NET_WEIGHT ** steps_back for steps_back in range(len(entries))])
    players = [make_net_player(network, epsilon=settings.SELF_PLAY_EPSILON) for _, network, _ in entries]

    if settings.SELF_PLAY_SEARCH_TURNS:
        searching = range(len(entries)) if settings.SELF_PLAY_OPPONENTS_SEARCH else [0]

        for index in searching:
            players[index] = make_search_player(entries[index][1], settings.SELF_PLAY_SEARCH_TURNS, settings.SELF_PLAY_SEARCH_KEEP,
                                                epsilon=settings.SELF_PLAY_EPSILON)

    labels = [f"gen {generation}{'*' if near else ''}" for generation, _, near in entries]
    weights = weights / weights.sum()

    if settings.BASELINE_SHARE:
        paths = sorted(BASELINES.glob("*.npz"))
        players += [make_net_player(load_net(path), epsilon=settings.SELF_PLAY_EPSILON) for path in paths]
        labels += [path.stem for path in paths]
        weights = np.concatenate([weights * (1 - settings.BASELINE_SHARE),
                                  np.full(len(paths), settings.BASELINE_SHARE / len(paths))])

    partner = sparring_partner()

    if partner is not None and settings.SPARRING_SHARE:
        if settings.SPARRING_PARTNER_TURNS:
            player = make_search_player(partner, settings.SPARRING_PARTNER_TURNS, settings.SPARRING_PARTNER_KEEP,
                                        epsilon=settings.SELF_PLAY_EPSILON)
        else:
            player = make_net_player(partner, epsilon=settings.SELF_PLAY_EPSILON)

        players.append(player)
        labels.append(f"partner {Path(settings.SPARRING_PARTNER).stem}")
        weights = np.append(weights * (1 - settings.SPARRING_SHARE), settings.SPARRING_SHARE)

    return players[0], players, weights, labels

def lasting_donors_to_keep():
    """The donor slots for the best rejects since the last accepted net (one is the rolling donor if that is on)."""

    return settings.DONORS_TO_KEEP - (1 if settings.RECENT_DONOR_WINDOW else 0)

class RollingDonor:
    """The best rejected net of the last RECENT_DONOR_WINDOW generations, a donor that keeps changing so the donors do not
    go stale. Nothing happens if RECENT_DONOR_WINDOW is 0."""

    def __init__(self):
        self.entry = None           # (generation, net, score vs best)

    def reject(self, generation, network, score):
        """A challenger was rejected: it takes the slot if it beats the current one or the current one is too old."""

        window = settings.RECENT_DONOR_WINDOW

        if window and (self.entry is None or generation - self.entry[0] >= window or score > self.entry[2]):
            self.entry = (generation, network, score)

    def accept(self):
        self.entry = None           # the new best net carries its own weights

    def donors_with(self, lasting):
        """The lasting donors plus the rolling one (unless it is already one of them)."""

        if self.entry is None or any(generation == self.entry[0] for generation, _, _ in lasting):
            return list(lasting)

        return list(lasting) + [self.entry]

class AntiDonors:
    """The last ANTI_DONORS_TO_KEEP rejected challengers that scored below ANTI_DONOR_BELOW against the best net.
    Nothing happens if ANTI_DONOR_BELOW is 0."""

    def __init__(self):
        self.nets = []

    def reject(self, network, score):
        if settings.ANTI_DONOR_BELOW and score < settings.ANTI_DONOR_BELOW:
            self.nets = (self.nets + [network])[-settings.ANTI_DONORS_TO_KEEP:]

    def accept(self):
        self.nets = []              # the new best net is a different starting point

def apply_anti_donors(challenger, best, anti_nets):
    """Move the challenger away from where the anti-donors went: for each part of the net, the difference between the
    anti-donors' average and the best net, restricted to the ANTI_DONOR_TOP_SHARE of weights that differ most,
    is subtracted ANTI_DONOR_STEP times."""

    for part in ("W1", "b1", "W2", "b2", "V", "c", "U"):
        base = getattr(best, part)
        current = getattr(challenger, part)

        if base is None or current is None:
            continue

        nets = [getattr(network, part) for network in anti_nets if getattr(network, part) is not None]
        nets = [weights for weights in nets if weights.shape == base.shape]

        if not nets:
            continue

        delta = np.mean(nets, axis=0) - base
        size = np.abs(delta)
        keep = max(1, int(round(size.size * settings.ANTI_DONOR_TOP_SHARE)))
        cutoff = np.partition(size.ravel(), size.size - keep)[size.size - keep]
        current -= (settings.ANTI_DONOR_STEP * delta * (size >= cutoff)).astype(current.dtype)

def apply_donors(challenger, donors):
    """Give the challenger the average of the donors' weights for the parts that are not from the best net.
    Those are the parts the best net may not have learned yet (see the donor notes at the top): the weights for the
    newest inputs, for the neurons beyond PROVEN_HIDDEN, and for the second layer. Everything else stays as the
    best net has it, so the challenger is still a small step from the best net."""

    nets = [network for _, network, _ in donors]

    def average(part, rows=slice(None), columns=slice(None)):
        return np.mean([getattr(network, part)[rows, columns] for network in nets], axis=0)

    challenger.W1[SIGNALS_START:] = average("W1", slice(SIGNALS_START, None))
    challenger.W1[:SIGNALS_START, settings.PROVEN_HIDDEN:] = average("W1", slice(None, SIGNALS_START), slice(settings.PROVEN_HIDDEN, None))
    challenger.b1[settings.PROVEN_HIDDEN:] = np.mean([network.b1[settings.PROVEN_HIDDEN:] for network in nets], axis=0)
    challenger.W2[settings.PROVEN_HIDDEN:] = average("W2", slice(settings.PROVEN_HIDDEN, None))
    with_layer = [network for network in nets if network.V is not None]         # donors from before the second layer have none

    if with_layer:
        for part in ("V", "c", "U"):
            setattr(challenger, part, np.mean([getattr(network, part) for network in with_layer], axis=0))
