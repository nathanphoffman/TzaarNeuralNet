"""Tests for the extra input signals, checked against the rules code on random positions."""
import unittest

import numpy as np

from tzaar.nets.players import random_player
from tzaar.rules.encoding import CAPTURES_SCALE, HEIGHT_SCALE, NUM_FEATURES, PIECES_PER_SIDE, STACKS_SCALE, encode_batch
from tzaar.rules.game import NUM_KINDS, PASS, State, new_game

ATTACK, LAST_HIT, SAFE, PIECES, TALLEST, STACKS = 797, 857, 863, 865, 871, 877

def random_positions(count, seed=0):
    """Positions from random games, skipping any that are over."""
    np.random.seed(seed)
    states, state = [], new_game()

    while len(states) < count:
        if state.result != 0:
            state = new_game()

        state.play(random_player(state))
        states.append(state.copy())

    return [state for state in states if state.result == 0]

def encode(state, actor):
    return encode_batch(np.array([state.owner]), np.array([state.kind]), np.array([state.height]), actor, [False])[0]

def moves_for(state, side, phase):
    copy = State(list(state.owner), list(state.kind), list(state.height), side, phase, False)

    return copy, copy._generate_moves()

class ExtraSignalTests(unittest.TestCase):
    def test_the_input_count_matches_the_layout(self):
        self.assertEqual(NUM_FEATURES, STACKS + 2)
        self.assertEqual(len(encode(new_game(), 1)), NUM_FEATURES)

    def test_attack_map_is_the_points_i_could_capture(self):
        for state in random_positions(40):
            actor = state.to_move
            _, moves = moves_for(state, actor, 0)
            expected = np.zeros(60)

            for _, destination in moves:
                expected[destination] = 1

            np.testing.assert_array_equal(encode(state, actor)[ATTACK:ATTACK + 60], expected)

    def test_safe_captures_and_stacking_moves_match_playing_them_out(self):
        for state in random_positions(40, seed=1):
            actor = state.to_move
            row = encode(state, actor)

            for index, side in enumerate((actor, -actor)):
                _, with_stacks = moves_for(state, side, 1)
                with_stacks = [move for move in with_stacks if move != PASS]
                captures = [move for move in with_stacks if state.owner[move[1]] == -side]
                stacks = [move for move in with_stacks if state.owner[move[1]] == side]
                safe = 0

                for source, destination in captures:
                    after, _ = moves_for(state, side, 1)
                    after._move_stack(source, destination)
                    answers = State(after.owner, after.kind, after.height, -side, 0, False)._generate_moves()
                    safe += not any(answer[1] == destination for answer in answers)

                self.assertAlmostEqual(row[SAFE + index], min(safe / CAPTURES_SCALE, 1), places=5)
                self.assertAlmostEqual(row[STACKS + index], min(len(stacks) / STACKS_SCALE, 1), places=5)

    def test_last_stack_alert_pieces_and_tallest(self):
        for state in random_positions(40, seed=2):
            actor = state.to_move
            row = encode(state, actor)
            threatened = {side: {destination for _, destination in moves_for(state, -side, 0)[1]} for side in (actor, -actor)}
            slot = 0

            for side in (actor, -actor):
                for kind in range(NUM_KINDS):
                    points = [point for point in range(60) if state.owner[point] == side and state.kind[point] == kind]
                    heights = [state.height[point] for point in points]
                    self.assertEqual(row[LAST_HIT + slot], float(len(points) == 1 and points[0] in threatened[side]))
                    self.assertAlmostEqual(row[PIECES + slot], sum(heights) / PIECES_PER_SIDE, places=5)
                    self.assertAlmostEqual(row[TALLEST + slot], min(max(heights, default=0) / HEIGHT_SCALE, 2), places=5)
                    slot += 1

    def test_the_older_inputs_are_unchanged_in_meaning(self):
        state = new_game()
        row = encode(state, 1)
        self.assertEqual(row[720], 0)                       # the first-action flag
        np.testing.assert_allclose(row[721:727], 1.0)       # all pieces left at the start

if __name__ == "__main__":
    unittest.main()
