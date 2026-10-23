"""The rules of tic-tac-toe. Board = list of 9 ints: 1 = X, -1 = O, 0 = empty."""
import numpy as np

WIN_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
             (0, 3, 6), (1, 4, 7), (2, 5, 8),
             (0, 4, 8), (2, 4, 6)]

def find_winner(board):
    """Return 1 if X won, -1 if O won, 0 if nobody has won."""

    for a, b, c in WIN_LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]

    return 0

def legal_moves(board):
    return [i for i in range(9) if board[i] == 0]

def encode_board(board, player):
    """Board seen from `player`'s side: 27 numbers (mine / theirs / empty per cell)."""
    board = np.array(board) * player

    return np.concatenate([(board == 1), (board == -1), (board == 0)]).astype(np.float32)
