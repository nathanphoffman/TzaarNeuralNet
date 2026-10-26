"""Tests for per-model settings and the training policy."""
import json
import unittest

from tests.helpers import TemporaryModels
from tzaar.learning import policy, settings
from tzaar.nets import model_store
from tzaar.nets.network import Network
from tzaar.rules.game import new_game
from tzaar.runs.config import apply_config, load_config

class ConfigTests(TemporaryModels):
    def test_missing_config_means_default_settings(self):
        self.assertEqual(load_config("first"), {})

    def test_config_changes_a_training_setting(self):
        self.addCleanup(setattr, settings, "MIN_SCORE_VS_BEST", settings.MIN_SCORE_VS_BEST)
        apply_config({"MIN_SCORE_VS_BEST": 0.6})
        self.assertEqual(settings.MIN_SCORE_VS_BEST, 0.6)

    def test_unknown_setting_is_an_error(self):
        with self.assertRaises(ValueError):
            apply_config({"NOT_A_SETTING": 1})

    def test_config_file_is_read(self):
        model_store.model_dir("first").mkdir()
        (model_store.model_dir("first") / "config.json").write_text(json.dumps({"seed": 3}))
        self.assertEqual(load_config("first"), {"seed": 3})

class TrainingHelperTests(unittest.TestCase):
    def test_self_play_best_net_looks_ahead_only_when_asked(self):
        self.addCleanup(setattr, settings, "SELF_PLAY_SEARCH_TURNS", settings.SELF_PLAY_SEARCH_TURNS)
        rated = [(0, Network(hidden=8), 100.0)]
        plain, players, _, _ = policy.self_play_opponents(rated, [])
        self.assertEqual(plain.func.__name__, "pick_move")
        settings.SELF_PLAY_SEARCH_TURNS = 1
        explorer, players, _, _ = policy.self_play_opponents(rated, [])
        self.assertEqual(explorer.func.__name__, "search_move")
        self.assertIs(players[0], explorer)
        state = new_game()
        self.assertIn(explorer(state), state.legal_moves())

    def test_games_grow_each_generation_up_to_the_cap(self):
        self.assertEqual(policy.games_for(0, 1000), settings.START_GAMES)
        self.assertEqual(policy.games_for(10 ** 6, 1000), 1000)

    def test_stronger_nets_get_fewer_epochs_but_at_least_one(self):
        self.assertGreater(policy.epochs_for(0), policy.epochs_for(500))
        self.assertEqual(policy.epochs_for(10 ** 6), 1)

    def test_clear_loss_to_random_is_never_borderline(self):
        self.assertFalse(policy.is_borderline(0.55, 1000, 900, 0.5))
