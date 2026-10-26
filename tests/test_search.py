"""Tests for the lookahead player. Run: uv run python -m unittest discover tests"""
import unittest

from tzaar.nets.games import play_game
from tzaar.nets.network import Network
from tzaar.nets.players import pick_move, random_player
from tzaar.nets.search import make_search_player, search_move
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

if __name__ == "__main__":
    unittest.main()
