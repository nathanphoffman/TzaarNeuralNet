"""Players are functions: state -> the move to play."""
import numpy as np

from .encoding import encode_batch
from .game import PASS
from .rng import rng


def random_player(state):
    moves = state.legal_moves()
    return moves[rng.integers(len(moves))]


def _positions_after(state, moves):
    """Arrays (owner, kind, height) of shape (M, 60): the board after each candidate move."""
    me, count = state.to_move, len(moves)
    owner = np.tile(np.array(state.owner, dtype=np.int16), (count, 1))
    kind = np.tile(np.array(state.kind, dtype=np.int16), (count, 1))
    height = np.tile(np.array(state.height, dtype=np.int16), (count, 1))

    src = np.array([m[0] for m in moves])
    dst = np.array([m[1] for m in moves])
    rows = np.nonzero(src >= 0)[0]                  # everything except a pass
    src, dst = src[rows], dst[rows]
    is_stack = owner[rows, dst] == me
    new_height = height[rows, src] + np.where(is_stack, height[rows, dst], 0)
    kind[rows, dst] = kind[rows, src]
    owner[rows, dst] = me
    height[rows, dst] = new_height
    owner[rows, src] = 0
    height[rows, src] = 0
    return owner, kind, height


def _lacks_a_kind(owner, kind, side):
    """For each position: does `side` have no stacks left of some kind?"""
    has_kind = np.stack([((owner == side) & (kind == k)).any(axis=1) for k in range(3)])
    return ~has_kind.all(axis=0)


def score_moves(net, state, moves):
    """Net's score for each move, from the mover's point of view.
    Moves that end the game on the spot get +2 (win) or -2 (lose)."""
    me = state.to_move
    owner, kind, height = _positions_after(state, moves)
    moves_again = state.phase == 0 and not state.is_first_turn
    scores = net.predict(encode_batch(owner, kind, height, me, np.full(len(moves), moves_again)))
    wins = _lacks_a_kind(owner, kind, -me)
    losses = _lacks_a_kind(owner, kind, me) & ~wins
    return np.where(wins, 2.0, np.where(losses, -2.0, scores))


def make_net_player(net, epsilon=0.0):
    """Try every legal move, ask the net how good the result is, pick the best.
    `epsilon` is the chance of playing a random move instead (exploration)."""
    def play(state):
        moves = state.legal_moves()
        if rng.random() < epsilon:
            return moves[rng.integers(len(moves))]
        return moves[int(np.argmax(score_moves(net, state, moves)))]
    return play
