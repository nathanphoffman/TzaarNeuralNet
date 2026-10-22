"""The training pipeline: random games first, then generations of self-play."""
import numpy as np

from .games import generate_training_data, match_score, play_match
from .network import Net
from .players import make_net_player, random_player
from .progress import progress
from .rating import elo_gap, rate_net

NUM_GENERATIONS = 5
RANDOM_GAMES = 1500
SELF_PLAY_GAMES = 800
MATCH_GAMES = 400
OPPONENTS_BACK = 3            # a new net is rated against this many of the latest accepted nets
SELF_PLAY_EPSILON = 0.2       # share of random moves in self-play, for variety
BATCHES_TO_KEEP = 3           # how many recent batches of data to train on
MIN_SCORE_VS_BEST = 0.55      # challenger must beat the current best by this much


def win_rate(player_a, player_b):
    return match_score(*play_match(player_a, player_b, MATCH_GAMES))


def train_first_net():
    """Generation 0: learn from random-vs-random games."""
    print(f"STEP 1: {RANDOM_GAMES:,} random-vs-random games")
    progress.step("Playing random-vs-random games")
    X, y = generate_training_data(random_player, random_player, RANDOM_GAMES)
    print(f"   {len(X)} positions recorded")

    print("STEP 2: train generation 0 on random data")
    progress.step("Training generation 0")
    net = Net()
    net.train(X, y)
    return net, X, y


def run_self_play_generations(best, first_X, first_y):
    """Each generation: best net plays itself, a fresh net trains on the data,
    and it replaces `best` only if it beats it (and isn't worse vs random).
    Returns (best net, its rating, its win rate vs random)."""
    print("STEP 3: self-play generations")
    print(f"   (rating = Elo, random player = 0, every +100 is about a 64% win rate; "
          f"win rates are over {MATCH_GAMES} games)")
    best_vs_random = win_rate(make_net_player(best), random_player)
    best_rating = elo_gap(best_vs_random)
    print(f"   gen 0: rating {best_rating:.0f} | vs random {best_vs_random:.2f}")
    progress.add_generation(0, rating=best_rating, vs_random=best_vs_random)

    rated_nets = [(best, best_rating)]       # accepted nets and their ratings, oldest first
    data_X, data_y = [first_X], [first_y]    # replay buffer: keep older data too

    for generation in range(1, NUM_GENERATIONS + 1):
        progress.step(f"Generation {generation}: best net plays itself")
        explorer = make_net_player(best, epsilon=SELF_PLAY_EPSILON)
        X, y = generate_training_data(explorer, explorer, SELF_PLAY_GAMES)
        data_X = (data_X + [X])[-BATCHES_TO_KEEP:]
        data_y = (data_y + [y])[-BATCHES_TO_KEEP:]

        progress.step(f"Generation {generation}: training a new net")
        challenger = Net()
        challenger.train(np.concatenate(data_X), np.concatenate(data_y))

        progress.step(f"Generation {generation}: testing the new net")
        player = make_net_player(challenger)
        results = [(rating, win_rate(player, make_net_player(net)))
                   for net, rating in rated_nets[-OPPONENTS_BACK:]]
        score_vs_best = results[-1][1]                     # the best net is the newest rated net
        # random is a fixed yardstick (rating 0): the challenger must not get worse against it
        score_vs_random = win_rate(player, random_player)
        rating = rate_net(results + [(0, score_vs_random)])

        accepted = (score_vs_best >= MIN_SCORE_VS_BEST
                    and score_vs_random >= best_vs_random - 0.01)
        print(f"   gen {generation}: rating {rating:.0f} | vs best {score_vs_best:.2f} | "
              f"vs random {score_vs_random:.2f}  -> {'ACCEPTED' if accepted else 'rejected'}")
        progress.add_generation(generation, rating=rating, vs_random=score_vs_random,
                                vs_best=score_vs_best, accepted=accepted)
        if accepted:
            best, best_rating, best_vs_random = challenger, rating, score_vs_random
            rated_nets.append((challenger, rating))
    return best, best_rating, best_vs_random


def train_model():
    """Run the whole pipeline and return the best net."""
    progress.start(NUM_GENERATIONS, total_steps=2 + 3 * NUM_GENERATIONS, match_games=MATCH_GAMES)
    best, first_X, first_y = train_first_net()
    best, rating, vs_random = run_self_play_generations(best, first_X, first_y)
    print(f"FINAL: best net rating {rating:.0f} (random player = 0), wins {vs_random:.0%} vs random")
    progress.finish(f"Best net rating: {rating:.0f} Elo. It beat a random player {vs_random:.0%} of the time.")
    return best
