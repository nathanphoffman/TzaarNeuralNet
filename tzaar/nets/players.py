"""Players are functions: state -> the move to play."""
import functools

import numpy as np

from tzaar.rules.encoding import encode_batch
from tzaar.rules.game import NUM_KINDS
from tzaar.rules.rng import random_generator

def random_player(state):
    moves = state.legal_moves()

    return moves[random_generator.integers(len(moves))]

def _positions_after(state, moves):
    """Arrays (owner, kind, height) of shape (M, 60): the board after each candidate move."""
    mover, count = state.to_move, len(moves)
    owner = np.tile(np.array(state.owner, dtype=np.int16), (count, 1))
    kind = np.tile(np.array(state.kind, dtype=np.int16), (count, 1))
    height = np.tile(np.array(state.height, dtype=np.int16), (count, 1))

    source = np.array([move[0] for move in moves])
    destination = np.array([move[1] for move in moves])
    rows = np.nonzero(source >= 0)[0]                  # everything except a pass
    source, destination = source[rows], destination[rows]
    is_stack = owner[rows, destination] == mover
    new_height = height[rows, source] + np.where(is_stack, height[rows, destination], 0)
    kind[rows, destination] = kind[rows, source]
    owner[rows, destination] = mover
    height[rows, destination] = new_height
    owner[rows, source] = 0
    height[rows, source] = 0

    return owner, kind, height

def _lacks_a_kind(owner, kind, side):
    """For each position: does `side` have no stacks left of some kind?"""
    has_kind = np.stack([((owner == side) & (kind == piece_kind)).any(axis=1) for piece_kind in range(NUM_KINDS)])

    return ~has_kind.all(axis=0)

def score_moves(network, state, moves):
    """Net's score for each move, from the mover's point of view.
    Moves that end the game on the spot get +2 (win) or -2 (lose)."""

    mover = state.to_move
    owner, kind, height = _positions_after(state, moves)
    moves_again = state.phase == 0 and not state.is_first_turn
    scores = network.predict(encode_batch(owner, kind, height, mover, np.full(len(moves), moves_again)))
    wins = _lacks_a_kind(owner, kind, -mover)
    losses = _lacks_a_kind(owner, kind, mover) & ~wins

    return np.where(wins, 2.0, np.where(losses, -2.0, scores))

def pick_move(network, epsilon, state):
    """Try every legal move, ask the net how good the result is, pick the best.
    `epsilon` is the chance of playing a random move instead (exploration)."""

    moves = state.legal_moves()

    if random_generator.random() < epsilon:
        return moves[random_generator.integers(len(moves))]

    return moves[int(np.argmax(score_moves(network, state, moves)))]

def make_net_player(network, epsilon=0.0):
    """A player (a function from a state to a move) that plays by the net."""

    return functools.partial(pick_move, network, epsilon)
