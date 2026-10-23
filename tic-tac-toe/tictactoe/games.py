"""Playing games, recording them as training data, and measuring strength."""
import numpy as np

from .game import encode_board, find_winner, legal_moves

def play_game(x_player, o_player):
    """Returns (positions, labels, winner). Each position is recorded from the
    side of the player who just moved; label = +1 if they won, -1 lost, 0 draw."""

    board, player, history = [0] * 9, 1, []

    while True:
        mover = x_player if player == 1 else o_player
        board[mover(board, player)] = player
        history.append((board.copy(), player))
        winner = find_winner(board)

        if winner != 0 or not legal_moves(board):
            break

        player = -player

    positions = [encode_board(b, p) for b, p in history]
    labels = [0.0 if winner == 0 else (1.0 if p == winner else -1.0) for _, p in history]

    return positions, labels, winner

def generate_training_data(player_a, player_b, num_games):
    """Play many games and return (X, y) training arrays. Alternates who goes first."""
    X, y = [], []

    for i in range(num_games):
        x_player, o_player = (player_a, player_b) if i % 2 == 0 else (player_b, player_a)
        positions, labels, _ = play_game(x_player, o_player)
        X += positions
        y += labels

    return np.array(X), np.array(y, dtype=np.float32)

def play_match(player_a, player_b, num_games=1000):
    """player_a's (wins, draws, losses) over num_games, alternating who goes first."""
    wins = draws = losses = 0

    for i in range(num_games):
        if i % 2 == 0:
            _, _, result = play_game(player_a, player_b)
        else:
            _, _, result = play_game(player_b, player_a)
            result = -result

        wins += result == 1
        draws += result == 0
        losses += result == -1

    return wins, draws, losses

def match_score(wins, draws, losses):
    """0 to 1: a win counts 1, a draw counts 0.5."""

    return (wins + 0.5 * draws) / (wins + draws + losses)
