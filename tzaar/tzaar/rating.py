"""Elo-style ratings. The random player is fixed at 0; every +100 is roughly a 64% win rate."""
import math

MIN_SCORE, MAX_SCORE = 0.02, 0.98     # clamp so a 100% / 0% match can't give an infinite gap


def elo_gap(score):
    """How many rating points better than the opponent a net is, given its win rate (0 to 1)."""
    score = min(max(score, MIN_SCORE), MAX_SCORE)
    return 400 * math.log10(score / (1 - score))


def rate_net(results):
    """results: list of (opponent_rating, win_rate). Average of what each opponent implies."""
    return sum(rating + elo_gap(score) for rating, score in results) / len(results)
