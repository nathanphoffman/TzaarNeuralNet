"""Playing games, recording them as training data, and measuring strength."""
import numpy as np

from .encoding import encode_state
from .game import new_game


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


def generate_training_data(player_a, player_b, num_games):
    """Play many games and return (X, y) training arrays. Alternates who is White."""
    X, y = [], []
    for i in range(num_games):
        white, black = (player_a, player_b) if i % 2 == 0 else (player_b, player_a)
        positions, labels, _ = play_game(white, black)
        X += positions
        y += labels
    return np.array(X), np.array(y, dtype=np.float32)


def play_match(player_a, player_b, num_games):
    """player_a's (wins, losses) over num_games, alternating who is White."""
    wins = 0
    for i in range(num_games):
        if i % 2 == 0:
            _, _, winner = play_game(player_a, player_b, record=False)
            wins += winner == 1
        else:
            _, _, winner = play_game(player_b, player_a, record=False)
            wins += winner == -1
    return wins, num_games - wins


def match_score(wins, losses):
    """Share of games won, 0 to 1 (Tzaar has no draws)."""
    return wins / (wins + losses)
