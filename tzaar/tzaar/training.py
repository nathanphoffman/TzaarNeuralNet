"""The training pipeline: random games first, then generations of self-play.

Training can be stopped and resumed: after every generation the nets and the history are
saved (see checkpoint.py), and the next run carries on from there.
"""
import itertools
from datetime import datetime

import numpy as np

from .checkpoint import RATING_BASIS, clear_checkpoint, load_checkpoint, save_checkpoint
from .encoding import SIGNALS_START
from .games import generate_training_data, match_score, play_match
from .network import Net
from .players import make_net_player, random_player
from .progress import progress
from .rating import elo_gap, rate_net

NUM_GENERATIONS = 5           # used when a fixed number is wanted (e.g. the game server's first-run training)
STEPS_PER_GENERATION = 3      # self-play, training, testing
RANDOM_GAMES = 1500
START_GAMES = 100             # games per self-play batch and per test match in generation 0...
GAMES_PER_GENERATION = 20     # ...growing by this many games for each generation after
MAX_SELF_PLAY_GAMES = 800
MAX_MATCH_GAMES = 600
WARM_START = True             # new nets start as a copy of the best net instead of from scratch
WARM_START_EPOCHS = 10
WARM_START_LEARNING_RATE = 0.02
OPPONENTS_BACK = 3            # a new net is rated against this many of the latest accepted nets
SELF_PLAY_EPSILON = 0.2       # share of random moves in self-play, for variety
OLDER_NET_WEIGHT = 0.5        # in self-play, each net one step older is this much less likely to be the opponent
BATCHES_TO_KEEP = 5           # how many recent batches of data to train on

# A new net is accepted only if all of these hold:
MIN_SCORE_VS_BEST = 0.52      # 1. it wins at least this share of games against the best net
MIN_RATING_GAIN = 5           # 2. its rating (head-to-head vs the last few accepted nets, random left out)
                              #    is at least this many Elo above the best net's
MIN_SCORE_VS_RANDOM = 0.90    # 3. safety floor: it still beats a random player this often
NEAR_MISS_SCORE = 0.50        # a rejected net that still won this much vs the best is kept as a self-play opponent

# The weights for the newer inputs (SIGNALS_START on) are only kept when a net is accepted, so a best net that
# was made before those inputs existed has zeros there, and every challenger would have to learn them from
# scratch and then lose them again when rejected. So each time a rejected challenger scores higher against the
# best net than any donor so far, it becomes a "donor" (the latest few are kept). New challengers get the average
# of the donors' weights for those inputs, which smooths out lucky scores. The donors are dropped when a net is
# accepted, because the new best net carries its own weights for them.
DONORS_TO_KEEP = 3

# A net that fails narrowly gets one re-test against the best net before it is rejected, because a
# normal test (a few hundred games) is only accurate to about +-2.5 points. The re-test result alone decides:
# MIN_SCORE_VS_BEST accepts it, NEAR_MISS_SCORE keeps it as an opponent.
RETEST_GAMES = 1600
RETEST_MIN_SCORE = 0.48       # "borderline" 1: rating clears the bar and it won at least this much vs the best net
RETEST_RATING_SLACK = 20      # "borderline" 2: it won enough vs the best net but its rating is short by up to this much


def games_for(generation, cap):
    """Early generations need little data and big gaps are easy to spot, so start small and grow."""
    return min(cap, START_GAMES + GAMES_PER_GENERATION * generation)


def win_rate(player_a, player_b, games):
    return match_score(*play_match(player_a, player_b, games))


def is_borderline(score_vs_best, rating, best_rating, score_vs_random):
    """A rejected net worth a closer look: one of the two checks passes and the other only narrowly fails."""
    if score_vs_random < MIN_SCORE_VS_RANDOM:
        return False
    strong_rating_but_even = rating >= best_rating + MIN_RATING_GAIN and score_vs_best >= RETEST_MIN_SCORE
    wins_but_rating_short = (score_vs_best >= MIN_SCORE_VS_BEST
                             and rating >= best_rating + MIN_RATING_GAIN - RETEST_RATING_SLACK)
    return strong_rating_but_even or wins_but_rating_short


def add_row(history, generation, rating, vs_random, games, vs_best=None, accepted=True, near_miss=False, retest=None):
    row = {"generation": generation, "rating": rating, "vs_random": vs_random,
           "vs_best": vs_best, "accepted": accepted, "near_miss": near_miss, "games": games,
           "retest": retest,               # retest: win rate vs the best net in the re-test, or None if it had none
           "time": datetime.now().isoformat(timespec="seconds")}
    history.append(row)
    progress.add_generation(row)


def start_new_run():
    """Generation 0: learn from random-vs-random games, then rate that net.
    Returns (history, rated_nets, near_misses, batches) ready for the self-play generations."""
    print(f"STEP 1: {RANDOM_GAMES:,} random-vs-random games")
    progress.step("Playing random-vs-random games")
    X, y = generate_training_data(random_player, [random_player], RANDOM_GAMES)
    print(f"   {len(X)} positions recorded")

    print("STEP 2: train generation 0 on random data")
    progress.step("Training generation 0")
    net = Net()
    net.train(X, y)

    # the one place the random player sets a rating: it anchors the scale (random = 0).
    # every later rating is measured relative to this one, so use the biggest match here
    vs_random = win_rate(make_net_player(net), random_player, MAX_MATCH_GAMES)
    rating = elo_gap(vs_random)
    print(f"   gen 0: rating {rating:.0f} | vs random {vs_random:.2f}")
    history, rated_nets = [], [(0, net, rating)]
    add_row(history, 0, rating, vs_random, MAX_MATCH_GAMES)
    save_checkpoint(history, rated_nets, [])
    return history, rated_nets, [], [(X, y)]


def self_play_opponents(rated_nets, near_misses):
    """The best net plays each self-play game against a net drawn from the best net itself, the older
    accepted nets, and the near-miss nets. The best net is the most likely, then the others newest
    first, each step back less likely, so really old nets only turn up now and then.
    Returns (the best net as an explorer, opponent players, their probabilities, labels)."""
    best_generation, best_net, _ = rated_nets[-1]
    others = [(g, net, False) for g, net, _ in rated_nets[:-1]] + [(g, net, True) for g, net, _ in near_misses]
    others.sort(key=lambda entry: entry[0], reverse=True)
    entries = [(best_generation, best_net, False)] + others
    weights = np.array([OLDER_NET_WEIGHT ** age for age in range(len(entries))])
    players = [make_net_player(net, epsilon=SELF_PLAY_EPSILON) for _, net, _ in entries]
    labels = [f"gen {g}{'*' if near else ''}" for g, _, near in entries]
    return players[0], players, weights / weights.sum(), labels


def rebuild_batches(rated_nets, near_misses, next_generation):
    """The saved checkpoint doesn't include training data, so make fresh self-play batches
    (one fewer than we keep; the first generation adds the last one)."""
    progress.step("Rebuilding training data from the best net")
    explorer, opponents, weights, _ = self_play_opponents(rated_nets, near_misses)
    games = games_for(next_generation, MAX_SELF_PLAY_GAMES)
    return [generate_training_data(explorer, opponents, games, weights)
            for _ in range(BATCHES_TO_KEEP - 1)]


def rebase_best_rating(rated_nets, games):
    """Checkpoints from before the new rating rule rated nets with the random player averaged in,
    which inflated the numbers. Re-rate the best net from head-to-head games only, so it can be
    compared fairly with the ratings new nets get."""
    generation, best, old_rating = rated_nets[-1]
    player = make_net_player(best)
    results = [(rating, win_rate(player, make_net_player(net), games))
               for _, net, rating in rated_nets[:-1][-OPPONENTS_BACK:]]
    new_rating = rate_net(results)
    print(f"Re-rated the best net (gen {generation}) from head-to-head games only: "
          f"{old_rating:.0f} -> {new_rating:.0f}")
    rated_nets[-1] = (generation, best, new_rating)


def run_self_play_generations(generations, history, rated_nets, near_misses, batches, donors=()):
    """Each generation: the best net plays older nets (and itself), a new net trains on the data,
    and it replaces the best net only if it passes the acceptance checks.
    Returns (best net, its rating)."""
    print("STEP 3: self-play generations")
    print("   (rating = Elo, random player = 0, every +100 is about a 64% win rate; "
          "later generations play more games per test; * = near-miss net used as an opponent)")
    best_generation, best, best_rating = rated_nets[-1]
    first = history[-1]["generation"] + 1
    numbers = itertools.count(first) if generations is None else range(first, first + generations)

    for generation in numbers:
        self_play_games = games_for(generation, MAX_SELF_PLAY_GAMES)
        match_games = games_for(generation, MAX_MATCH_GAMES)

        progress.step(f"Generation {generation}: best net plays itself and older nets ({self_play_games} games)")
        explorer, opponents, weights, labels = self_play_opponents(rated_nets, near_misses)
        mix = ", ".join(f"{label} {w:.0%}" for label, w in zip(labels, weights) if w >= 0.005)
        print(f"   gen {generation}: self-play opponents: {mix}")
        batches = (batches + [generate_training_data(explorer, opponents, self_play_games, weights)])[-BATCHES_TO_KEEP:]

        progress.step(f"Generation {generation}: training a new net")
        X = np.concatenate([X for X, _ in batches])
        y = np.concatenate([y for _, y in batches])
        if WARM_START:
            challenger = best.copy()
            if donors:
                challenger.W1[SIGNALS_START:] = np.mean([net.W1[SIGNALS_START:] for _, net, _ in donors], axis=0)
            challenger.train(X, y, epochs=WARM_START_EPOCHS, learning_rate=WARM_START_LEARNING_RATE)
        else:
            challenger = Net()
            challenger.train(X, y)

        progress.step(f"Generation {generation}: testing the new net ({match_games} games per match)")
        player = make_net_player(challenger)
        results = [(rating, win_rate(player, make_net_player(net), match_games))
                   for _, net, rating in rated_nets[-OPPONENTS_BACK:]]
        score_vs_best = results[-1][1]                     # the best net is the newest rated net
        score_vs_random = win_rate(player, random_player, match_games)     # for the floor and the dashboard only
        rating = rate_net(results)                         # head-to-head only: random is left out on purpose

        accepted = (score_vs_best >= MIN_SCORE_VS_BEST
                    and rating >= best_rating + MIN_RATING_GAIN
                    and score_vs_random >= MIN_SCORE_VS_RANDOM)
        near_miss = not accepted and score_vs_best >= NEAR_MISS_SCORE
        print(f"   gen {generation}: rating {rating:.0f} (best: {best_rating:.0f}) | vs best {score_vs_best:.2f} | "
              f"vs random {score_vs_random:.2f}")

        retest = None
        if not accepted and is_borderline(score_vs_best, rating, best_rating, score_vs_random):
            progress.set_status(f"Generation {generation}: re-testing against the best net ({RETEST_GAMES} games)")
            print(f"   gen {generation}: borderline, re-testing against the best net ({RETEST_GAMES} games)...")
            retest = win_rate(player, make_net_player(best), RETEST_GAMES)
            accepted = retest >= MIN_SCORE_VS_BEST
            near_miss = not accepted and retest >= NEAR_MISS_SCORE
            if accepted:
                # the big match against the best net is a more reliable measurement than the average over
                # three opponents, so base the new rating on it. Otherwise the best net's rating could
                # drift down each time a net squeaks in with a lower average
                rating = best_rating + elo_gap(retest)

        outcome = "ACCEPTED" if accepted else "rejected (near miss, kept as an opponent)" if near_miss else "rejected"
        if retest is not None:
            outcome += f" (re-test: {retest:.3f} over {RETEST_GAMES} games)"
            if accepted:
                outcome += f", rating set to {rating:.0f} from the re-test"
        print(f"   gen {generation}: -> {outcome}")
        add_row(history, generation, rating, score_vs_random, match_games, score_vs_best, accepted, near_miss, retest)
        if accepted:
            best, best_rating = challenger, rating
            rated_nets.append((generation, challenger, rating))
            donors = []
        else:
            if near_miss:
                near_misses.append((generation, challenger, rating))
            if not donors or score_vs_best > max(score for _, _, score in donors):
                donors = (list(donors) + [(generation, challenger, score_vs_best)])[-DONORS_TO_KEEP:]
        save_checkpoint(history, rated_nets, near_misses, donors)
    return best, best_rating


def train_model(generations=None, fresh=False):
    """Run `generations` more generations, or keep going until stopped if it is None.
    Continues from a saved checkpoint if there is one, unless `fresh`. Returns the best net."""
    if fresh:
        clear_checkpoint()
    checkpoint = load_checkpoint()
    open_ended = generations is None
    if checkpoint is None:
        progress.start(None if open_ended else generations,
                       None if open_ended else 2 + STEPS_PER_GENERATION * generations,
                       MAX_MATCH_GAMES, setup_steps=2, steps_per_generation=STEPS_PER_GENERATION)
        history, rated_nets, near_misses, batches = start_new_run()
        donors = []
    else:
        history, rated_nets, near_misses, basis, donors = checkpoint
        last = history[-1]["generation"]
        print(f"Resuming after generation {last}. Best net so far: generation {rated_nets[-1][0]}, "
              f"rating {rated_nets[-1][2]:.0f}; {len(near_misses)} near-miss nets saved")
        rebase = basis < RATING_BASIS and len(rated_nets) > 1
        setup_steps = 1 + rebase
        progress.start(None if open_ended else last + generations,
                       None if open_ended else setup_steps + STEPS_PER_GENERATION * generations,
                       MAX_MATCH_GAMES, rows=history, setup_steps=setup_steps,
                       steps_per_generation=STEPS_PER_GENERATION)
        if rebase:
            progress.step("Re-rating the best net (head-to-head games only)")
            rebase_best_rating(rated_nets, games_for(last + 1, MAX_MATCH_GAMES))
        batches = rebuild_batches(rated_nets, near_misses, last + 1)
        save_checkpoint(history, rated_nets, near_misses, donors)       # records the new rating basis

    best, rating = run_self_play_generations(generations, history, rated_nets, near_misses, batches, donors)
    vs_random = next(r["vs_random"] for r in history if r["generation"] == rated_nets[-1][0])
    print(f"FINAL: best net rating {rating:.0f} (random player = 0), wins {vs_random:.0%} vs random")
    progress.finish(f"Best net rating: {rating:.0f} Elo. It beat a random player {vs_random:.0%} of the time.")
    return best
