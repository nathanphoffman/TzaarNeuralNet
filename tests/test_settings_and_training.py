"""Tests for per-model settings and the training helpers."""
import json
import unittest

from tests.helpers import TemporaryModels
from tzaar import model_store, training
from tzaar.config import apply_config, load_config

class ConfigTests(TemporaryModels):
    def test_missing_config_means_default_settings(self):
        self.assertEqual(load_config("first"), {})

    def test_config_changes_a_training_setting(self):
        original = training.MIN_SCORE_VS_BEST

        try:
            apply_config({"MIN_SCORE_VS_BEST": 0.6})
            self.assertEqual(training.MIN_SCORE_VS_BEST, 0.6)
        finally:
            training.MIN_SCORE_VS_BEST = original

    def test_unknown_setting_is_an_error(self):
        with self.assertRaises(ValueError):
            apply_config({"NOT_A_SETTING": 1})

    def test_config_file_is_read(self):
        model_store.model_dir("first").mkdir()
        (model_store.model_dir("first") / "config.json").write_text(json.dumps({"seed": 3}))
        self.assertEqual(load_config("first"), {"seed": 3})

class TrainingHelperTests(unittest.TestCase):
    def test_games_grow_each_generation_up_to_the_cap(self):
        self.assertEqual(training.games_for(0, 1000), training.START_GAMES)
        self.assertEqual(training.games_for(10 ** 6, 1000), 1000)

    def test_stronger_nets_get_fewer_epochs_but_at_least_one(self):
        self.assertGreater(training.epochs_for(0), training.epochs_for(500))
        self.assertEqual(training.epochs_for(10 ** 6), 1)

    def test_clear_loss_to_random_is_never_borderline(self):
        self.assertFalse(training.is_borderline(0.55, 1000, 900, 0.5))
