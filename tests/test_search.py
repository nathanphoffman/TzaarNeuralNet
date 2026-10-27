"""Tests for the lookahead player. Run: uv run python -m unittest discover tests"""
import unittest

from tzaar.nets.games import play_game
from tzaar.nets.network import Network
from tzaar.nets.players import pick_move, random_player
from tzaar.nets.search import make_search_player, make_value_labeler, search_move, search_value
from tzaar.rules.game import new_game

class SearchTests(unittest.TestCase):
    def setUp(self):
        self.network = Network(hidden=8)

    def test_one_turn_on_a_one_action_turn_matches_the_plain_net_player(self):
        for _ in range(5):
            state = new_game()      # White's first turn is a single action
            self.assertEqual(search_move(self.network, 1, 1000, state), pick_move(self.network, 0.0, state))

    def test_it_plays_legal_moves_at_every_depth(self):
        for turns in (1, 2):
            state = new_game()

            for _ in range(6):
                move = search_move(self.network, turns, 3, state)
                self.assertIn(move, state.legal_moves())
                state.play(move)

    def test_a_search_player_finishes_a_game_against_random(self):
        _, _, winner = play_game(make_search_player(self.network, turns=2, keep=2), random_player, record=False)
        self.assertIn(winner, (1, -1))

class SearchLabelTests(unittest.TestCase):
    def setUp(self):
        self.network = Network(hidden=8)

    def test_value_stays_in_the_nets_range_and_is_for_the_side_asked(self):
        state = new_game()
        state.play(state.legal_moves()[0])
        value = search_value(self.network, state, 1, 3, state.to_move)
        self.assertTrue(-1.0 <= value <= 1.0)
        self.assertAlmostEqual(search_value(self.network, state, 1, 3, -state.to_move) + value, 0.0, places=5)

    def test_labeler_labels_about_its_share_of_the_calls(self):
        state = new_game()
        state.play(state.legal_moves()[0])
        never = make_value_labeler(self.network, 1, 2, 0.0)
        always = make_value_labeler(self.network, 1, 2, 1.0)
        self.assertIsNone(never(state, state.to_move))
        self.assertIsNotNone(always(state, state.to_move))

    def test_labeled_positions_mix_the_result_and_the_search_value(self):
        positions, labels, winner = play_game(random_player, random_player, labeler=lambda state, side: 0.0, label_weight=0.5)
        self.assertEqual(len(positions), len(labels))
        self.assertTrue(all(abs(label) in (0.5, 1.0) for label in labels))     # unlabeled final position keeps +-1
        self.assertGreater(sum(abs(label) == 0.5 for label in labels), 0)

    def test_no_labeler_means_plain_win_loss_labels(self):
        _, labels, _ = play_game(random_player, random_player)
        self.assertTrue(all(abs(label) == 1.0 for label in labels))

if __name__ == "__main__":
    unittest.main()
