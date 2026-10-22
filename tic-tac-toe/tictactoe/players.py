"""Players are functions: (board, player) -> the square to play."""
import numpy as np

from .game import encode_board, legal_moves
from .rng import rng


def random_player(board, player):
    return rng.choice(legal_moves(board))


def make_net_player(net, epsilon=0.0):
    """Try every legal move, ask the net how good the result is, pick the best.
    `epsilon` is the chance of playing a random move instead (exploration)."""
    def play(board, player):
        moves = legal_moves(board)
        if rng.random() < epsilon:
            return rng.choice(moves)
        positions_after = []
        for move in moves:
            next_board = board.copy()
            next_board[move] = player
            positions_after.append(encode_board(next_board, player))
        return moves[int(np.argmax(net.predict(np.array(positions_after))))]
    return play
