"""Tests for the game rules, the players and the Elo maths. Run: uv run python -m unittest discover tests"""
import unittest

from tzaar.rules.game import State, new_game
from tzaar.nets.players import random_player
from tzaar.nets.rating import elo_gap, rate_net
from tzaar.nets.games import play_game

class EloTests(unittest.TestCase):
    def test_even_score_is_no_gap(self):
        self.assertEqual(elo_gap(0.5), 0)

    def test_sixty_four_percent_is_about_one_hundred_elo(self):
        self.assertAlmostEqual(elo_gap(0.64), 100, delta=1)

    def test_perfect_score_is_clamped_not_infinite(self):
        self.assertLess(elo_gap(1.0), 1000)
        self.assertGreater(elo_gap(0.0), -1000)

    def test_rating_averages_what_each_opponent_implies(self):
        self.assertAlmostEqual(rate_net([(100, 0.5), (300, 0.5)]), 200)

class GameTests(unittest.TestCase):
    def test_new_game_has_thirty_pieces_per_side(self):
        state = new_game()
        self.assertEqual(state.owner.count(1), 30)
        self.assertEqual(state.owner.count(-1), 30)

    def test_first_action_can_only_capture(self):
        state = new_game()

        for source, destination in state.legal_moves():
            self.assertEqual(state.owner[destination], -state.to_move)

    def test_state_survives_a_round_trip_through_a_dict(self):
        state = new_game()
        copy = State.from_dict(state.to_dict())
        self.assertEqual(copy.to_dict(), state.to_dict())

    def test_a_random_game_ends_with_a_winner(self):
        _, _, winner = play_game(random_player, random_player, record=False)
        self.assertIn(winner, (1, -1))

    def test_recorded_game_has_one_label_per_position(self):
        positions, labels, _ = play_game(random_player, random_player)
        self.assertEqual(len(positions), len(labels))
        self.assertTrue(set(labels) <= {1.0, -1.0})

class GameCountTests(unittest.TestCase):
    def test_finished_games_are_reported_while_a_batch_is_played(self):
        from tzaar.nets import games

        reports = []
        original = games.progress_callback, games.PROGRESS_SECONDS
        games.progress_callback, games.PROGRESS_SECONDS = lambda done, total: reports.append((done, total)), 0.05

        try:
            games.win_rate(random_player, random_player, 24)
        finally:
            games.progress_callback, games.PROGRESS_SECONDS = original

        self.assertTrue(all(total == 24 for _, total in reports))
        self.assertTrue(all(0 <= done <= 24 for done, _ in reports))
