"""The rules training follows: how much to play and train at each stage, who counts as a worthy opponent,
and how to tell a narrow fail from a clear one."""

import numpy as np

from tzaar.learning import settings
from tzaar.nets.players import make_net_player
from tzaar.rules.encoding import SIGNALS_START

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
    accepted nets, and the near-miss nets. The best net is the most likely, then the others newest
    first, each step back less likely, so really old nets only turn up now and then.
    Returns (the best net as an explorer, opponent players, their probabilities, labels)."""

    best_generation, best_net, _ = rated_nets[-1]
    others = ([(generation, network, False) for generation, network, _ in rated_nets[:-1]]
              + [(generation, network, True) for generation, network, _ in near_misses])

    others.sort(key=lambda entry: entry[0], reverse=True)
    entries = [(best_generation, best_net, False)] + others
    weights = np.array([settings.OLDER_NET_WEIGHT ** steps_back for steps_back in range(len(entries))])
    players = [make_net_player(network, epsilon=settings.SELF_PLAY_EPSILON) for _, network, _ in entries]
    labels = [f"gen {generation}{'*' if near else ''}" for generation, _, near in entries]

    return players[0], players, weights / weights.sum(), labels

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
