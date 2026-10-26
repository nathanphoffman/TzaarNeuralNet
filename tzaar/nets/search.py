"""Lookahead: try the net's best few moves, then the opponent's best few replies, and so on, and pick the move
whose worst case is best. Depth is counted in whole turns (a turn is one or two actions by the same player)."""
import functools

import numpy as np

from tzaar.nets.players import score_moves

def _value(network, state, turns_left, keep, side):
    """How good `state` is for `side`: the best result for `side` when it is to move, the worst when the opponent is.
    Only the `keep` moves the net likes best (for whoever is moving) are tried."""

    moves, mover = state.legal_moves(), state.to_move
    scores = score_moves(network, state, moves)
    perspective = mover * side                     # +1 if the mover is `side`, else -1
    results = []

    for index in np.argsort(-scores, kind="stable")[:keep]:
        child = state.copy()
        child.play(moves[index])
        turn_over = child.to_move != mover

        if child.result != 0 or (turn_over and turns_left == 1):
            results.append(scores[index] * perspective)       # the net's score is from the mover's side
        else:
            results.append(_value(network, child, turns_left - (1 if turn_over else 0), keep, side))

    return max(results) if mover == side else min(results)

def search_move(network, turns, keep, state):
    """The move with the best worst case, looking `turns` whole turns ahead and trying only the `keep` best-scoring
    moves at every step. With turns=1 on a one-action turn this is the same as pick_move with no exploration."""

    moves, side = state.legal_moves(), state.to_move
    scores = score_moves(network, state, moves)
    best_move, best_value = None, -np.inf

    for index in np.argsort(-scores, kind="stable")[:keep]:
        child = state.copy()
        child.play(moves[index])
        turn_over = child.to_move != side

        if child.result != 0 or (turn_over and turns == 1):
            value = scores[index]
        else:
            value = _value(network, child, turns - (1 if turn_over else 0), keep, side)

        if value > best_value:
            best_move, best_value = moves[index], value

    return best_move

def make_search_player(network, turns=2, keep=5):
    """A player (a function from a state to a move) that looks `turns` turns ahead."""

    return functools.partial(search_move, network, turns, keep)
