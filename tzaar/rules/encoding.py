"""Turn positions into numbers the network can read.

Inputs, in order (879 in all):
  0-359    presence: for each of 6 piece slots (my/their Tzaar, Tzaara, Tott) x 60 points, 1 if it is there
  360-719  height: the same layout, stack height / 5
  720      flag: 1 if the same player moves again (the second action of a turn)
  721-726  pieces left: my/their Tzaar, Tzaara, Tott, as a share of how many each side starts with
  727-732  danger: 1 if a side has exactly one stack of that type left (losing the last one loses the game)
  733-792  threat: for each of the 60 points, 1 if my stack there could be captured by the opponent next move
  793-794  captures available: how many captures the opponent / I could make if it were their / my move (share of 20)
  795-796  stuck: 1 if the opponent / I would have no capture at all (a side with no capture on its turn loses)
  797-856  attack: for each of the 60 points, 1 if the opponent's stack there could be captured by me if it were my move
  857-862  last stack hit: for my / their Tzaar, Tzaara, Tott, 1 if that is the side's last stack of the kind and it could be captured
  863-864  safe captures: how many captures I / the opponent could make that the other side could not answer by capturing
           the capturing stack back (share of 20)
  865-870  pieces: for my / their Tzaar, Tzaara, Tott, the pieces in stacks showing that kind (as a share of 30)
  871-876  tallest: the height of the tallest stack of each of those six (stack height / 5, at most 2)
  877-878  stacks: how many stacking moves I / the opponent could make (share of 40)
"""

import numpy as np

from tzaar.rules.board import NUM_POINTS, RAY_TABLE
from tzaar.rules.game import NUM_KINDS, PIECES_PER_KIND

NUM_FEATURES = NUM_POINTS * 6 * 2 + 1 + 6 + 6 + NUM_POINTS + 2 + 2 + NUM_POINTS + 6 + 2 + 6 + 6 + 2
STACKS_SCALE = 40
PIECES_PER_SIDE = 30
CAPTURES_SCALE = 20
NUM_SLOTS = 2 * NUM_KINDS      # piece slots: my three kinds, then theirs
HEIGHT_SCALE = 5               # stack heights are divided by this to keep the inputs near 0 to 1
SIGNALS_START = NUM_POINTS * 6 * 2 + 1       # inputs from here on (pieces left, danger, threat, and the later ones) came after the first nets
START_COUNTS = np.tile(np.array(PIECES_PER_KIND, dtype=np.float32), 2)       # for [my 3 kinds, their 3 kinds]

def piece_counts(owner, kind, actor):
    """(M, 6): stacks left of [my Tzaar, Tzaara, Tott, their Tzaar, Tzaara, Tott]. A stack counts as its top piece."""

    # One row per position: how many pieces of each kind each side has (this side's three kinds, then the other's).
    return np.stack([((owner == side) & (kind == piece_kind)).sum(axis=1)
                     for side in (actor, -actor) for piece_kind in range(NUM_KINDS)], axis=1)

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

def _stack_lines(owner, height, side):
    """Where each of `side`'s stacks can go in each of the 6 directions: the first piece on the line is the only
    point it can land on. Returns (occupied, owner, height, position, source, target, can_hit, can_stack) with the
    first three padded by one extra empty point (see captures); the rest has one row per stack of `side`
    and one column per direction."""

    count = owner.shape[0]
    occupied = np.concatenate([owner != 0, np.zeros((count, 1), dtype=bool)], axis=1)
    owner_ext = np.concatenate([owner, np.zeros((count, 1), dtype=owner.dtype)], axis=1)
    height_ext = np.concatenate([height, np.zeros((count, 1), dtype=height.dtype)], axis=1)

    position, source = np.nonzero(owner == side)
    rays = RAY_TABLE[source]
    ray_occupied = occupied[position[:, None, None], rays]
    has_target = ray_occupied.any(axis=2)
    first = ray_occupied.argmax(axis=2)
    target = np.take_along_axis(rays, first[..., None], axis=2)[..., 0]
    target_owner = owner_ext[position[:, None], target]
    can_hit = has_target & (target_owner == -side) & (height_ext[position[:, None], target] <= height[position, source][:, None])
    can_stack = has_target & (target_owner == side)

    return occupied, owner_ext, height_ext, position, source, target, can_hit, can_stack

def _safe_captures(lines, height, side, count):
    """(M,): how many captures `side` could make after which the capturing stack, now standing on the captured point
    with its own height and its old point empty, can not be captured back."""

    occupied, owner_ext, height_ext, position, source, target, can_hit, _ = lines
    hit_stack, hit_direction = np.nonzero(can_hit)
    where, origin = position[hit_stack], source[hit_stack]
    landing = target[hit_stack, hit_direction]
    landed_height = height[where, origin]

    rays = RAY_TABLE[landing]                                                    # the lines out of the point it landed on
    ray_occupied = occupied[where[:, None, None], rays] & (rays != origin[:, None, None])      # the old point is empty now
    has_neighbour = ray_occupied.any(axis=2)
    nearest = np.take_along_axis(rays, ray_occupied.argmax(axis=2)[..., None], axis=2)[..., 0]
    exposed = (has_neighbour & (owner_ext[where[:, None], nearest] == -side)
               & (height_ext[where[:, None], nearest] >= landed_height[:, None])).any(axis=1)

    return np.bincount(where, weights=~exposed, minlength=count)

def extra_signals(owner, kind, height, actor, attack):
    """The inputs from 'attack' on (see the top of the file). `attack` is the first result of captures(..., actor)
    and `threat` the same for the opponent; both are (M, 60)."""

    count = owner.shape[0]
    threat = captures(owner, height, -actor)[0]
    sides = (actor, -actor)
    victims = (threat, attack)                                   # the points where each side's stacks could be captured
    last_hit, pieces, tallest, safe, stacks = [], [], [], [], []

    for side, hit in zip(sides, victims):
        lines = _stack_lines(owner, height, side)
        safe.append(_safe_captures(lines, height, side, count))
        stacks.append(np.bincount(lines[3], weights=lines[7].sum(axis=1), minlength=count))

        for piece_kind in range(NUM_KINDS):
            mine = (owner == side) & (kind == piece_kind)
            last_hit.append((mine.sum(axis=1) == 1) & (mine & (hit > 0)).any(axis=1))
            pieces.append((height * mine).sum(axis=1) / PIECES_PER_SIDE)
            tallest.append(np.minimum((height * mine).max(axis=1) / HEIGHT_SCALE, 2))

    safe = np.stack(safe, axis=1)
    stacks = np.stack(stacks, axis=1)

    return np.concatenate([attack, np.stack(last_hit, axis=1).astype(np.float32),
                           np.minimum(safe / CAPTURES_SCALE, 1).astype(np.float32),
                           np.stack(pieces, axis=1).astype(np.float32), np.stack(tallest, axis=1).astype(np.float32),
                           np.minimum(stacks / STACKS_SCALE, 1).astype(np.float32)], axis=1)

def threatened_stacks(owner, height, actor):
    """(M, 60): 1 where one of actor's stacks could be captured if the opponent were to move now."""

    return captures(owner, height, -actor)[0]

def encode_batch(owner, kind, height, actor, actor_moves_again):
    """Positions seen from `actor`'s side.
    owner/kind/height: arrays of shape (M, 60). actor_moves_again: array of M bools."""

    count = owner.shape[0]
    rows, points = np.nonzero(owner != 0)
    channel = (kind + NUM_KINDS * (owner != actor))[rows, points]       # 0-2 mine, 3-5 theirs

    presence = np.zeros((count, NUM_SLOTS, NUM_POINTS), dtype=np.float32)
    heights = np.zeros((count, NUM_SLOTS, NUM_POINTS), dtype=np.float32)
    presence[rows, channel, points] = 1
    heights[rows, channel, points] = height[rows, points] / HEIGHT_SCALE

    flag = np.asarray(actor_moves_again, dtype=np.float32).reshape(count, 1)
    counts = piece_counts(owner, kind, actor)
    threat, their_captures = captures(owner, height, -actor)
    attack, my_captures = captures(owner, height, actor)
    available = np.stack([their_captures, my_captures], axis=1)

    return np.concatenate([presence.reshape(count, -1), heights.reshape(count, -1), flag,
                           (counts / START_COUNTS).astype(np.float32), (counts == 1).astype(np.float32),
                           threat, np.minimum(available / CAPTURES_SCALE, 1).astype(np.float32),
                           (available == 0).astype(np.float32), extra_signals(owner, kind, height, actor, attack)], axis=1)
