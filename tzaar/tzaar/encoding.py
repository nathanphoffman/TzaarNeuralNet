"""Turn positions into numbers the network can read."""
import numpy as np

from .board import NUM_POINTS

NUM_FEATURES = NUM_POINTS * 6 * 2 + 1   # per point: 6 (side x kind) presence + 6 heights, + 1 flag


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
    return np.concatenate([presence.reshape(count, -1), heights.reshape(count, -1), flag], axis=1)


def encode_state(state, actor):
    """One position, seen from `actor`'s side (usually the player who just moved)."""
    return encode_batch(np.array([state.owner]), np.array([state.kind]), np.array([state.height]),
                        actor, [state.to_move == actor])[0]
