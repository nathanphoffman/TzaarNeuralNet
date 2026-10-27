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

    def test_opponents_look_ahead_too_only_when_asked(self):
        for name in ("SELF_PLAY_SEARCH_TURNS", "SELF_PLAY_OPPONENTS_SEARCH"):
            self.addCleanup(setattr, settings, name, getattr(settings, name))

        rated = [(0, Network(hidden=8), 100.0), (1, Network(hidden=8), 110.0)]
        settings.SELF_PLAY_SEARCH_TURNS = 1
        _, players, _, _ = policy.self_play_opponents(rated, [])
        self.assertEqual([player.func.__name__ for player in players], ["search_move", "pick_move"])
        settings.SELF_PLAY_OPPONENTS_SEARCH = True
        _, players, _, _ = policy.self_play_opponents(rated, [])
        self.assertEqual([player.func.__name__ for player in players], ["search_move", "search_move"])

    def test_rolling_donor_is_off_by_default(self):
        rolling = policy.RollingDonor()
        rolling.reject(1, "net", 0.4)
        self.assertEqual(rolling.donors_with([]), [])
        self.assertEqual(policy.lasting_donors_to_keep(), settings.DONORS_TO_KEEP)

    def test_rolling_donor_is_the_best_of_the_window_and_replaced_when_old(self):
        self.addCleanup(setattr, settings, "RECENT_DONOR_WINDOW", settings.RECENT_DONOR_WINDOW)
        settings.RECENT_DONOR_WINDOW = 3
        self.assertEqual(policy.lasting_donors_to_keep(), settings.DONORS_TO_KEEP - 1)
        rolling = policy.RollingDonor()
        rolling.reject(1, "a", 0.45)
        rolling.reject(2, "b", 0.40)                  # worse and not old yet: keeps a
        self.assertEqual(rolling.entry[1], "a")
        rolling.reject(3, "c", 0.47)                  # better: takes the slot
        self.assertEqual(rolling.entry[1], "c")
        rolling.reject(4, "d", 0.30)
        self.assertEqual(rolling.entry[1], "c")
        rolling.reject(6, "e", 0.30)                  # c is 3 generations old: replaced even though e is worse
        self.assertEqual(rolling.entry[1], "e")
        rolling.accept()
        self.assertIsNone(rolling.entry)

    def test_rolling_donor_is_not_counted_twice(self):
        self.addCleanup(setattr, settings, "RECENT_DONOR_WINDOW", settings.RECENT_DONOR_WINDOW)
        settings.RECENT_DONOR_WINDOW = 5
        rolling = policy.RollingDonor()
        rolling.reject(7, "net", 0.48)
        self.assertEqual(len(rolling.donors_with([(7, "net", 0.48)])), 1)
        self.assertEqual(len(rolling.donors_with([(5, "old", 0.49)])), 2)

    def test_baselines_get_their_share_of_self_play_games(self):
        self.addCleanup(setattr, settings, "BASELINE_SHARE", settings.BASELINE_SHARE)
        rated = [(0, Network(hidden=8), 100.0)]
        _, plain, plain_weights, _ = policy.self_play_opponents(rated, [])
        settings.BASELINE_SHARE = 0.2
        _, players, weights, labels = policy.self_play_opponents(rated, [])
        count = len(players) - len(plain)
        self.assertGreater(count, 0)
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertAlmostEqual(weights[len(plain):].sum(), 0.2)
        self.assertEqual(len(labels), len(players))

    def test_games_grow_each_generation_up_to_the_cap(self):
        self.assertEqual(policy.games_for(0, 1000), settings.START_GAMES)
        self.assertEqual(policy.games_for(10 ** 6, 1000), 1000)

    def test_stronger_nets_get_fewer_epochs_but_at_least_one(self):
        self.assertGreater(policy.epochs_for(0), policy.epochs_for(500))
        self.assertEqual(policy.epochs_for(10 ** 6), 1)

    def test_clear_loss_to_random_is_never_borderline(self):
        self.assertFalse(policy.is_borderline(0.55, 1000, 900, 0.5))
