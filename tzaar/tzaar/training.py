"""The training pipeline: random games first, then generations of self-play."""
import numpy as np

from .games import generate_training_data, match_score, play_match
from .network import Net
from .players import make_net_player, random_player

NUM_GENERATIONS = 5
RANDOM_GAMES = 1500
SELF_PLAY_GAMES = 800
MATCH_GAMES = 100
SELF_PLAY_EPSILON = 0.2       # share of random moves in self-play, for variety
BATCHES_TO_KEEP = 3           # how many recent batches of data to train on
MIN_SCORE_VS_BEST = 0.55      # challenger must beat the current best by this much


def train_first_net():
    """Generation 0: learn from random-vs-random games."""
    print(f"STEP 1: {RANDOM_GAMES:,} random-vs-random games")
    X, y = generate_training_data(random_player, random_player, RANDOM_GAMES)
    print(f"   {len(X)} positions recorded")

    print("STEP 2: train generation 0 on random data")
    net = Net()
    net.train(X, y)
    print("   vs random (win/loss):", play_match(make_net_player(net), random_player, MATCH_GAMES))
    return net, X, y


def run_self_play_generations(best, first_X, first_y):
    """Each generation: best net plays itself, a fresh net trains on the data,
    and it replaces `best` only if it beats it (and isn't worse vs random)."""
    print("STEP 3: self-play generations")
    print("   (scores are win rates over the match games: 0.50 = even, higher = better)")
    best_vs_random = match_score(*play_match(make_net_player(best), random_player, MATCH_GAMES))
    baseline = best                          # generation 0, a fixed yardstick for progress
    print(f"   gen 0: vs random {best_vs_random:.2f}")
    data_X, data_y = [first_X], [first_y]    # replay buffer: keep older data too

    for generation in range(1, NUM_GENERATIONS + 1):
        explorer = make_net_player(best, epsilon=SELF_PLAY_EPSILON)
        X, y = generate_training_data(explorer, explorer, SELF_PLAY_GAMES)
        data_X = (data_X + [X])[-BATCHES_TO_KEEP:]
        data_y = (data_y + [y])[-BATCHES_TO_KEEP:]

        challenger = Net()
        challenger.train(np.concatenate(data_X), np.concatenate(data_y))
        score_vs_best = match_score(*play_match(make_net_player(challenger),
                                                make_net_player(best, epsilon=0.05), MATCH_GAMES))
        # random is a fixed yardstick: the challenger must not get worse against it
        score_vs_random = match_score(*play_match(make_net_player(challenger), random_player, MATCH_GAMES))
        score_vs_gen0 = match_score(*play_match(make_net_player(challenger),
                                                make_net_player(baseline), MATCH_GAMES))
        accepted = (score_vs_best >= MIN_SCORE_VS_BEST
                    and score_vs_random >= best_vs_random - 0.01)
        print(f"   gen {generation}: vs best {score_vs_best:.2f} | vs gen 0 {score_vs_gen0:.2f} | "
              f"vs random {score_vs_random:.2f}  -> {'ACCEPTED' if accepted else 'rejected'}")
        if accepted:
            best, best_vs_random = challenger, score_vs_random
    return best


def train_model():
    """Run the whole pipeline and return the best net."""
    best, first_X, first_y = train_first_net()
    best = run_self_play_generations(best, first_X, first_y)
    print("FINAL: best net vs random (win/loss):",
          play_match(make_net_player(best), random_player, MATCH_GAMES))
    return best
