"""Playing games, recording them as training data, and measuring strength."""
import multiprocessing

import numpy as np

from .encoding import encode_state
from .game import new_game
from .rng import random_generator

# Playing games is most of the time spent in training, and games are independent of each other, so they are played in
# parallel. 6 of a 16-thread machine, so that two runs can train side by side (12 if only one is running) and
# there is still room for the dashboard and anything else running at the same time.
WORKERS = 6                   # processes that play games at the same time

_job = None                   # the work for the current batch of games; set before the workers are forked

def _run_chunk(args):
    """Runs in a worker: play one slice of the games with its own random stream."""
    game_numbers, seed = args
    random_generator.bit_generator.state = np.random.default_rng(seed).bit_generator.state

    return _job(game_numbers)

def _in_parallel(work_function, num_games):
    """Split games 0..num_games-1 over the workers and return each worker's result from job(game_numbers).
    The workers are forked, so they inherit the players (which are functions and can't be sent between processes)."""

    global _job
    _job = work_function
    chunks = [chunk for chunk in np.array_split(np.arange(num_games), WORKERS) if len(chunk)]
    seeds = random_generator.integers(2 ** 32, size=len(chunks))

    with multiprocessing.get_context("fork").Pool(len(chunks)) as pool:
        return pool.map(_run_chunk, list(zip(chunks, seeds)))

def play_game(white_player, black_player, record=True):
    """Returns (positions, labels, winner). Each position is recorded from the
    side of the player who just acted; label = +1 if they went on to win, else -1."""

    state, history = new_game(), []

    while state.result == 0:
        actor = state.to_move
        player = white_player if actor == 1 else black_player
        state.play(player(state))

        if record:
            history.append((encode_state(state, actor), actor))

    positions = [position for position, _ in history]
    labels = [1.0 if actor == state.result else -1.0 for _, actor in history]

    return positions, labels, state.result

def generate_training_data(player, opponents, num_games, weights=None):
    """Play many games and return (inputs, targets) training arrays. Each game's opponent is drawn at
    random from `opponents`, with probabilities `weights` (equal if None). Alternates who is White."""

    choices = random_generator.choice(len(opponents), size=num_games, p=weights)

    def play_some(game_numbers):
        game_inputs, game_targets = [], []

        for game_number in game_numbers:
            opponent = opponents[choices[game_number]]
            white, black = (player, opponent) if game_number % 2 == 0 else (opponent, player)
            positions, labels, _ = play_game(white, black)
            game_inputs += positions
            game_targets += labels

        return game_inputs, game_targets

    inputs, targets = [], []

    for part_inputs, part_targets in _in_parallel(play_some, num_games):
        inputs += part_inputs
        targets += part_targets

    return np.array(inputs), np.array(targets, dtype=np.float32)

def play_match(player_a, player_b, num_games):
    """player_a's (wins, losses) over num_games, alternating who is White."""

    def play_some(game_numbers):
        wins = 0

        for game_number in game_numbers:
            if game_number % 2 == 0:
                _, _, winner = play_game(player_a, player_b, record=False)
                wins += winner == 1
            else:
                _, _, winner = play_game(player_b, player_a, record=False)
                wins += winner == -1

        return int(wins)

    wins = sum(_in_parallel(play_some, num_games))

    return wins, num_games - wins

def match_score(wins, losses):
    """Share of games won, 0 to 1 (Tzaar has no draws)."""

    return wins / (wins + losses)
