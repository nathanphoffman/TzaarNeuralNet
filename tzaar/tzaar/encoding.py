"""Turn positions into numbers the network can read.

Inputs, in order (797 in all):
  0-359    presence: for each of 6 piece slots (my/their Tzaar, Tzaara, Tott) x 60 points, 1 if it is there
  360-719  height: the same layout, stack height / 5
  720      flag: 1 if the same player moves again (the second action of a turn)
  721-726  pieces left: my/their Tzaar, Tzaara, Tott, as a share of how many each side starts with
  727-732  danger: 1 if a side has exactly one stack of that type left (losing the last one loses the game)
  733-792  threat: for each of the 60 points, 1 if my stack there could be captured by the opponent next move
  793-794  captures available: how many captures the opponent / I could make if it were their / my move (share of 20)
  795-796  stuck: 1 if the opponent / I would have no capture at all (a side with no capture on its turn loses)
"""

import numpy as np

from .board import NUM_POINTS, RAY_TABLE
from .game import PIECES_PER_KIND

NUM_FEATURES = NUM_POINTS * 6 * 2 + 1 + 6 + 6 + NUM_POINTS + 2 + 2
CAPTURES_SCALE = 20
SIGNALS_START = NUM_POINTS * 6 * 2 + 1       # inputs from here on (pieces left, danger, threat) came after the first nets
START_COUNTS = np.tile(np.array(PIECES_PER_KIND, dtype=np.float32), 2)       # for [my 3 kinds, their 3 kinds]

def piece_counts(owner, kind, actor):
    """(M, 6): stacks left of [my Tzaar, Tzaara, Tott, their Tzaar, Tzaara, Tott]. A stack counts as its top piece."""

    # One row per position: how many pieces of each kind each side has (this side's three kinds, then the other's).
    return np.stack([((owner == side) & (kind == piece_kind)).sum(axis=1)
                     for side in (actor, -actor) for piece_kind in range(3)], axis=1)

def captures(owner, height, attacker):
    """Returns (victims, moves): victims is (M, 60), 1 where one of the other side's stacks could be captured if
    `attacker` were to move now, and moves is (M,), how many captures `attacker` could make.
    A stack can be captured by an enemy stack at least as tall that reaches it along a straight line
    with nothing in between (the first piece on the line is the one it can hit)."""

    count = owner.shape[0]
    occupied = np.concatenate([owner != 0, np.zeros((count, 1), dtype=bool)], axis=1)    # extra column = padding
    owner_ext = np.concatenate([owner, np.zeros((count, 1), dtype=owner.dtype)], axis=1)
    height_ext = np.concatenate([height, np.zeros((count, 1), dtype=height.dtype)], axis=1)

    position, source = np.nonzero(owner == attacker)             # only the attacker's stacks can attack
    rays = RAY_TABLE[source]                                     # (n, direction, step): points along each line
    ray_occupied = occupied[position[:, None, None], rays]
    has_target = ray_occupied.any(axis=2)                        # (n, direction)
    first = ray_occupied.argmax(axis=2)                          # first occupied step on each line
    target = np.take_along_axis(rays, first[..., None], axis=2)[..., 0]
    can_hit = (has_target & (owner_ext[position[:, None], target] == -attacker)
               & (height_ext[position[:, None], target] <= height[position, source][:, None]))

    hit_position, hit_direction = np.nonzero(can_hit)
    threat = np.zeros((count, NUM_POINTS + 1), dtype=np.float32)
    threat[position[hit_position], target[hit_position, hit_direction]] = 1
    moves = np.bincount(position, weights=can_hit.sum(axis=1), minlength=count)

    return threat[:, :NUM_POINTS], moves

def threatened_stacks(owner, height, actor):
    """(M, 60): 1 where one of actor's stacks could be captured if the opponent were to move now."""

    return captures(owner, height, -actor)[0]

def encode_batch(owner, kind, height, actor, actor_moves_again):
    """Positions seen from `actor`'s side.
    owner/kind/height: arrays of shape (M, 60). actor_moves_again: array of M bools."""

    count = owner.shape[0]
    rows, points = np.nonzero(owner != 0)
    channel = (kind + 3 * (owner != actor))[rows, points]       # 0-2 mine, 3-5 theirs

    presence = np.zeros((count, 6, NUM_POINTS), dtype=np.float32)
    heights = np.zeros((count, 6, NUM_POINTS), dtype=np.float32)
    presence[rows, channel, points] = 1
    heights[rows, channel, points] = height[rows, points] / 5

    flag = np.asarray(actor_moves_again, dtype=np.float32).reshape(count, 1)
    counts = piece_counts(owner, kind, actor)
    threat, their_captures = captures(owner, height, -actor)
    _, my_captures = captures(owner, height, actor)
    available = np.stack([their_captures, my_captures], axis=1)

    return np.concatenate([presence.reshape(count, -1), heights.reshape(count, -1), flag,
                           (counts / START_COUNTS).astype(np.float32), (counts == 1).astype(np.float32),
                           threat, np.minimum(available / CAPTURES_SCALE, 1).astype(np.float32),
                           (available == 0).astype(np.float32)], axis=1)

def encode_state(state, actor):
    """One position, seen from `actor`'s side (usually the player who just moved)."""

    return encode_batch(np.array([state.owner]), np.array([state.kind]), np.array([state.height]),
                        actor, [state.to_move == actor])[0]
