"""One generation of self-play: play games, train a challenger, test it and decide where it goes."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from tzaar.learning import settings
from tzaar.learning.checkpoint import GENERATION_KEY, RATING_KEY, save_batch
from tzaar.nets.games import generate_training_data, win_rate
from tzaar.learning.policy import apply_donors, epochs_for, games_for, is_borderline, lasting_donors_to_keep, self_play_opponents
from tzaar.nets.network import Network
from tzaar.nets.players import make_net_player, random_player
from tzaar.nets.search import make_value_labeler
from tzaar.nets.rating import elo_gap, rate_net
from tzaar.runs.progress import progress

def add_row(history, generation, rating, vs_random, games, **details):
    """Add a generation's row to the history. `details` overrides the defaults for a generation 0 row (accepted,
    no match against the best net): vs_best, accepted, near_miss, retest (the win rate vs the best net in the re-test)."""

    result_row = {GENERATION_KEY: generation, RATING_KEY: rating, "vs_random": vs_random,
                  "vs_best": None, "accepted": True, "near_miss": False, "games": games,
                  "retest": None, "time": datetime.now().isoformat(timespec="seconds")}

    result_row.update(details)
    history.append(result_row)
    progress.add_generation(result_row)

@dataclass
class Verdict:
    """How a challenger did in its tests. retest is its win rate in the re-test, or None if it had none."""

    rating: float
    score_vs_best: float
    score_vs_random: float
    accepted: bool
    near_miss: bool
    borderline: bool = False       # a narrow fail, worth a re-test
    retest: float = None

def play_self_play_games(generation, rated_nets, near_misses, batches):
    """The best net plays itself and older nets. Returns the batches of training data with the new one added."""

    self_play_games = games_for(generation, settings.MAX_SELF_PLAY_GAMES)
    progress.step(f"Generation {generation}: best net plays itself and older nets ({self_play_games} games)")
    explorer, opponents, weights, labels = self_play_opponents(rated_nets, near_misses)
    opponent_mix = ", ".join(f"{label} {weight:.0%}" for label, weight in zip(labels, weights) if weight >= settings.MIN_SHARE_TO_PRINT)
    print(f"   gen {generation}: self-play opponents: {opponent_mix}")
    looks_ahead = (f"; the best net looks {settings.SELF_PLAY_SEARCH_TURNS} turns ahead" if settings.SELF_PLAY_SEARCH_TURNS else "")
    labels_by_search = (f"; {settings.SEARCH_LABEL_SHARE:.0%} of positions labeled by lookahead" if settings.SEARCH_LABEL_SHARE else "")
    progress.set_detail(f"Generation {generation} self-play opponents: {opponent_mix} (* = near miss){looks_ahead}{labels_by_search}")

    labeler = None

    if settings.SEARCH_LABEL_SHARE:
        labeler = make_value_labeler(rated_nets[-1][1], settings.SEARCH_LABEL_TURNS, settings.SEARCH_LABEL_KEEP,
                                     settings.SEARCH_LABEL_SHARE)

    batch = generate_training_data(explorer, opponents, self_play_games, weights, labeler, settings.SEARCH_LABEL_WEIGHT)
    save_batch(batch, rated_nets[-1][0], settings.BATCHES_TO_KEEP)

    return (batches + [batch])[-settings.BATCHES_TO_KEEP:]

def train_challenger(generation, rated_nets, donors, batches):
    """A new net trained on the recent batches: a copy of the best net (with the donors' weights) or a fresh one."""

    _, best, best_rating = rated_nets[-1]
    progress.step(f"Generation {generation}: training a new net")
    inputs = np.concatenate([inputs for inputs, _ in batches])
    targets = np.concatenate([targets for _, targets in batches])

    if not settings.WARM_START:
        challenger = Network()
        challenger.train(inputs, targets)

        return challenger

    challenger = best.copy()

    if donors:
        apply_donors(challenger, donors)

    challenger.train(inputs, targets, epochs=epochs_for(best_rating), learning_rate=settings.WARM_START_LEARNING_RATE)

    return challenger

def retest_borderline(generation, challenger, rated_nets, verdict):
    """A narrow fail gets one bigger match against the best net, and that result alone decides."""

    _, best, best_rating = rated_nets[-1]
    player = make_net_player(challenger)
    progress.set_status(f"Generation {generation}: re-testing against the best net ({settings.RETEST_GAMES} games)")
    print(f"   gen {generation}: borderline, re-testing against the best net ({settings.RETEST_GAMES} games)...")
    verdict.retest = win_rate(player, make_net_player(best), settings.RETEST_GAMES)
    verdict.accepted = verdict.retest >= settings.MIN_SCORE_VS_BEST
    verdict.near_miss = not verdict.accepted and verdict.retest >= settings.NEAR_MISS_SCORE

    if verdict.accepted:
        # the big match against the best net is a more reliable measurement than the average over
        # three opponents, so base the new rating on it. Otherwise the best net's rating could
        # drift down each time a net squeaks in with a lower average
        verdict.rating = best_rating + elo_gap(verdict.retest)

def test_challenger(generation, challenger, rated_nets, match_games):
    """Play the challenger against the latest accepted nets and the random player, and decide if it is accepted."""

    _, _, best_rating = rated_nets[-1]
    progress.step(f"Generation {generation}: testing the new net ({match_games} games per match)")
    player = make_net_player(challenger)
    results = [(rating, win_rate(player, make_net_player(network), match_games))
               for _, network, rating in rated_nets[-settings.OPPONENTS_BACK:]]

    score_vs_best = results[-1][1]                     # the best net is the newest rated net
    score_vs_random = win_rate(player, random_player, match_games)     # for the floor and the dashboard only
    rating = rate_net(results)                         # head-to-head only: random is left out on purpose

    accepted = (score_vs_best >= settings.MIN_SCORE_VS_BEST
                and rating >= best_rating + settings.MIN_RATING_GAIN
                and score_vs_random >= settings.MIN_SCORE_VS_RANDOM)

    near_miss = not accepted and score_vs_best >= settings.NEAR_MISS_SCORE
    print(f"   gen {generation}: rating {rating:.0f} (best: {best_rating:.0f}) | vs best {score_vs_best:.2f} | "
          f"vs random {score_vs_random:.2f}")

    borderline = not accepted and is_borderline(score_vs_best, rating, best_rating, score_vs_random)

    return Verdict(rating, score_vs_best, score_vs_random, accepted, near_miss, borderline)

def report_verdict(generation, verdict):
    # picks the first line that fits: accepted, kept as a near miss, or plain rejected
    outcome = "ACCEPTED" if verdict.accepted else "rejected (near miss, kept as an opponent)" if verdict.near_miss else "rejected"

    if verdict.retest is not None:
        outcome += f" (re-test: {verdict.retest:.3f} over {settings.RETEST_GAMES} games)"

        if verdict.accepted:
            outcome += f", rating set to {verdict.rating:.0f} from the re-test"

    print(f"   gen {generation}: -> {outcome}")

def file_challenger(generation, challenger, verdict, rated_nets, near_misses, donors):
    """Put the challenger where its verdict says: the accepted nets, the near misses, or the donors.
    Returns the donors (an accepted net empties them, because the new best net carries its own weights)."""

    if verdict.accepted:
        rated_nets.append((generation, challenger, verdict.rating))

        return []

    if verdict.near_miss:
        near_misses.append((generation, challenger, verdict.rating))

    if not donors or verdict.score_vs_best > max(score for _, _, score in donors):
        donors = (list(donors) + [(generation, challenger, verdict.score_vs_best)])[-lasting_donors_to_keep():]

    return donors
