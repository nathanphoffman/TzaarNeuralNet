"""The training pipeline: random games first, then generations of self-play.

Training can be stopped and resumed: after every generation the accepted nets and the
history are saved (see checkpoint.py), and the next run carries on from there.
"""
import numpy as np

from .checkpoint import clear_checkpoint, load_checkpoint, save_checkpoint
from .games import generate_training_data, match_score, play_match
from .network import Net
from .players import make_net_player, random_player
from .progress import progress
from .rating import elo_gap, rate_net

NUM_GENERATIONS = 5           # how many generations to run each time training is started
RANDOM_GAMES = 1500
START_GAMES = 100             # games per self-play batch and per test match in generation 0...
GAMES_PER_GENERATION = 20     # ...growing by this many games for each generation after
MAX_SELF_PLAY_GAMES = 800
MAX_MATCH_GAMES = 400
WARM_START = True             # new nets start as a copy of the best net instead of from scratch
WARM_START_EPOCHS = 10
WARM_START_LEARNING_RATE = 0.02
OPPONENTS_BACK = 3            # a new net is rated against this many of the latest accepted nets
SELF_PLAY_EPSILON = 0.2       # share of random moves in self-play, for variety
BATCHES_TO_KEEP = 3           # how many recent batches of data to train on
MIN_SCORE_VS_BEST = 0.55      # challenger must beat the current best by this much


def games_for(generation, cap):
    """Early generations need little data and big gaps are easy to spot, so start small and grow."""
    return min(cap, START_GAMES + GAMES_PER_GENERATION * generation)


def win_rate(player_a, player_b, games):
    return match_score(*play_match(player_a, player_b, games))


def add_row(history, generation, rating, vs_random, games, vs_best=None, accepted=True):
    row = {"generation": generation, "rating": rating, "vs_random": vs_random,
           "vs_best": vs_best, "accepted": accepted, "games": games}
    history.append(row)
    progress.add_generation(row)


def start_new_run():
    """Generation 0: learn from random-vs-random games, then rate that net.
    Returns (history, rated_nets, batches) ready for the self-play generations."""
    print(f"STEP 1: {RANDOM_GAMES:,} random-vs-random games")
    progress.step("Playing random-vs-random games")
    X, y = generate_training_data(random_player, random_player, RANDOM_GAMES)
    print(f"   {len(X)} positions recorded")

    print("STEP 2: train generation 0 on random data")
    progress.step("Training generation 0")
    net = Net()
    net.train(X, y)

    # every later rating is measured relative to this one, so use the biggest match here
    vs_random = win_rate(make_net_player(net), random_player, MAX_MATCH_GAMES)
    rating = elo_gap(vs_random)
    print(f"   gen 0: rating {rating:.0f} | vs random {vs_random:.2f}")
    history, rated_nets = [], [(0, net, rating)]
    add_row(history, 0, rating, vs_random, MAX_MATCH_GAMES)
    save_checkpoint(history, rated_nets)
    return history, rated_nets, [(X, y)]


def rebuild_batches(best, next_generation):
    """The saved checkpoint doesn't include training data, so make fresh self-play batches
    (one fewer than we keep; the first generation adds the last one)."""
    progress.step("Rebuilding training data from the best net")
    explorer = make_net_player(best, epsilon=SELF_PLAY_EPSILON)
    games = games_for(next_generation, MAX_SELF_PLAY_GAMES)
    return [generate_training_data(explorer, explorer, games) for _ in range(BATCHES_TO_KEEP - 1)]


def run_self_play_generations(generations, history, rated_nets, batches):
    """Each generation: best net plays itself, a fresh net trains on the data,
    and it replaces `best` only if it beats it (and isn't worse vs random).
    Returns (best net, its rating, its win rate vs random)."""
    print("STEP 3: self-play generations")
    print("   (rating = Elo, random player = 0, every +100 is about a 64% win rate; "
          "later generations play more games per test)")
    best_generation, best, best_rating = rated_nets[-1]
    best_vs_random = next(r["vs_random"] for r in history if r["generation"] == best_generation)
    first = history[-1]["generation"] + 1

    for generation in range(first, first + generations):
        self_play_games = games_for(generation, MAX_SELF_PLAY_GAMES)
        match_games = games_for(generation, MAX_MATCH_GAMES)

        progress.step(f"Generation {generation}: best net plays itself ({self_play_games} games)")
        explorer = make_net_player(best, epsilon=SELF_PLAY_EPSILON)
        batches = (batches + [generate_training_data(explorer, explorer, self_play_games)])[-BATCHES_TO_KEEP:]

        progress.step(f"Generation {generation}: training a new net")
        X = np.concatenate([X for X, _ in batches])
        y = np.concatenate([y for _, y in batches])
        if WARM_START:
            challenger = best.copy()
            challenger.train(X, y, epochs=WARM_START_EPOCHS, learning_rate=WARM_START_LEARNING_RATE)
        else:
            challenger = Net()
            challenger.train(X, y)

        progress.step(f"Generation {generation}: testing the new net ({match_games} games per match)")
        player = make_net_player(challenger)
        results = [(rating, win_rate(player, make_net_player(net), match_games))
                   for _, net, rating in rated_nets[-OPPONENTS_BACK:]]
        score_vs_best = results[-1][1]                     # the best net is the newest rated net
        # random is a fixed yardstick (rating 0): the challenger must not get worse against it
        score_vs_random = win_rate(player, random_player, match_games)
        rating = rate_net(results + [(0, score_vs_random)])

        accepted = (score_vs_best >= MIN_SCORE_VS_BEST
                    and score_vs_random >= best_vs_random - 0.01)
        print(f"   gen {generation}: rating {rating:.0f} | vs best {score_vs_best:.2f} | "
              f"vs random {score_vs_random:.2f}  -> {'ACCEPTED' if accepted else 'rejected'}")
        add_row(history, generation, rating, score_vs_random, match_games, score_vs_best, accepted)
        if accepted:
            best, best_rating, best_vs_random = challenger, rating, score_vs_random
            rated_nets.append((generation, challenger, rating))
        save_checkpoint(history, rated_nets)
    return best, best_rating, best_vs_random


def train_model(generations=NUM_GENERATIONS, fresh=False):
    """Run `generations` more generations (continuing from a saved checkpoint if there is
    one, unless `fresh`) and return the best net."""
    if fresh:
        clear_checkpoint()
    checkpoint = load_checkpoint()
    if checkpoint is None:
        progress.start(generations, total_steps=2 + 3 * generations, match_games=MAX_MATCH_GAMES)
        history, rated_nets, batches = start_new_run()
    else:
        history, rated_nets = checkpoint
        last = history[-1]["generation"]
        print(f"Resuming after generation {last}. Best net so far: generation {rated_nets[-1][0]}, "
              f"rating {rated_nets[-1][2]:.0f}")
        progress.start(last + generations, total_steps=1 + 3 * generations,
                       match_games=MAX_MATCH_GAMES, rows=history)
        batches = rebuild_batches(rated_nets[-1][1], last + 1)

    best, rating, vs_random = run_self_play_generations(generations, history, rated_nets, batches)
    print(f"FINAL: best net rating {rating:.0f} (random player = 0), wins {vs_random:.0%} vs random")
    progress.finish(f"Best net rating: {rating:.0f} Elo. It beat a random player {vs_random:.0%} of the time.")
    return best
