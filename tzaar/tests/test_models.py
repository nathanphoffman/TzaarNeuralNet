"""Tests for saving, loading and listing models, per-model settings and the training helpers.
They use a temporary models folder, never the real one."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tzaar import model_store, training
from tzaar.checkpoint import GENERATION_KEY, NETS_KEY, clear_checkpoint, load_checkpoint, save_checkpoint
from tzaar.config import apply_config, load_config
from tzaar.encoding import NUM_FEATURES
from tzaar.network import Network
from tzaar.server import NetPlayers, available_nets

class TemporaryModels(unittest.TestCase):
    """Points the model store at an empty temporary folder for the length of each test."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.original_root = model_store.MODELS_ROOT
        model_store.MODELS_ROOT = Path(self.folder.name)
        model_store.use_model("first")

    def tearDown(self):
        model_store.MODELS_ROOT = self.original_root
        self.folder.cleanup()

def history_row(generation, rating=100.0):
    return {GENERATION_KEY: generation, "rating": rating, "vs_random": 0.9, "vs_best": None, "accepted": True,
            "near_miss": False, "games": 10, "retest": None, "time": "now"}

class NetworkStoreTests(TemporaryModels):
    def test_saved_net_scores_positions_the_same_after_loading(self):
        network = Network(hidden=8)
        model_store.save_net(network)
        loaded = model_store.load_net()
        positions = np.random.default_rng(0).random((3, NUM_FEATURES)).astype(np.float32)
        np.testing.assert_allclose(network.predict(positions), loaded.predict(positions), rtol=1e-5)

    def test_older_net_with_fewer_inputs_is_padded_with_zeros(self):
        network = Network(hidden=8)
        model_store.model_dir().mkdir()
        np.savez(model_store.model_path(), W1=network.W1[:-5], b1=network.b1, W2=network.W2, b2=network.b2)
        loaded = model_store.load_net()
        self.assertEqual(loaded.W1.shape[0], NUM_FEATURES)
        self.assertFalse(loaded.W1[-5:].any())

    def test_copy_does_not_share_weights(self):
        network = Network(hidden=8)
        copy = network.copy()
        copy.W1 += 1
        self.assertFalse(np.array_equal(network.W1, copy.W1))

    def test_widened_net_scores_positions_the_same(self):
        network = Network(hidden=8)
        positions = np.random.default_rng(1).random((3, NUM_FEATURES)).astype(np.float32)
        np.testing.assert_allclose(network.predict(positions), network.widened(16).predict(positions), rtol=1e-5)

    def test_models_are_listed_by_folder(self):
        model_store.model_dir("second").mkdir()
        model_store.model_dir("first").mkdir()
        self.assertEqual(model_store.list_models(), ["first", "second"])

def save_two_nets():
    history = [history_row(0), history_row(1, 200.0)]
    rated_nets = [(0, Network(hidden=8), 100.0), (1, Network(hidden=8), 200.0)]
    save_checkpoint(history, rated_nets, [])

    return history

class CheckpointTests(TemporaryModels):

    def test_nothing_to_resume_in_a_new_model(self):
        self.assertIsNone(load_checkpoint())

    def test_checkpoint_resumes_with_the_same_history_and_nets(self):
        history = save_two_nets()
        saved_history, rated_nets, near_misses, _, donors = load_checkpoint()
        self.assertEqual(saved_history, history)
        self.assertEqual([(generation, rating) for generation, _, rating in rated_nets], [(0, 100.0), (1, 200.0)])
        self.assertEqual((near_misses, donors), ([], []))

    def test_clearing_a_checkpoint_starts_over(self):
        save_two_nets()
        clear_checkpoint()
        self.assertIsNone(load_checkpoint())

class ServerTests(TemporaryModels):
    def test_only_trained_models_can_be_played(self):
        model_store.model_dir("untrained").mkdir()
        save_two_nets()
        models = available_nets()
        self.assertEqual([model["name"] for model in models], ["first"])
        self.assertEqual([net[GENERATION_KEY] for net in models[0][NETS_KEY]], [0, 1])

    def test_unknown_net_is_refused(self):
        with self.assertRaises(ValueError):
            NetPlayers().player_for("first", 99)

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
