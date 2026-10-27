"""Tests for resuming training from a checkpoint and for listing the nets you can play."""

import numpy as np

from tests.helpers import TemporaryModels, save_two_nets
from tzaar.nets import model_store
from tzaar.learning.checkpoint import GENERATION_KEY, NETS_KEY, clear_checkpoint, load_batches, load_checkpoint, save_batch
from tzaar.rules.encoding import NUM_FEATURES
from web.server import NetPlayers, available_nets

class CheckpointTests(TemporaryModels):

    def test_nothing_to_resume_in_a_new_model(self):
        self.assertIsNone(load_checkpoint())

    def test_checkpoint_resumes_with_saved_history_nets(self):
        history = save_two_nets()
        saved_history, rated_nets, near_misses, _, donors = load_checkpoint()
        self.assertEqual(saved_history, history)
        self.assertEqual([(generation, rating) for generation, _, rating in rated_nets], [(0, 100.0), (1, 200.0)])
        self.assertEqual((near_misses, donors), ([], []))

    def test_clearing_a_checkpoint_starts_over(self):
        save_two_nets()
        clear_checkpoint()
        self.assertIsNone(load_checkpoint())

class SavedBatchTests(TemporaryModels):
    def batch(self, value):
        return np.full((3, NUM_FEATURES), value, dtype=np.float32), np.full(3, value, dtype=np.float32)

    def test_a_saved_batch_comes_back_for_the_same_best_net(self):
        save_batch(self.batch(1.0), best_generation=7, keep=2)
        (inputs, targets), = load_batches(7, 1)
        self.assertEqual(inputs.shape, (3, NUM_FEATURES))
        self.assertEqual(targets.tolist(), [1.0, 1.0, 1.0])

    def test_batches_from_an_older_best_net_are_not_reused(self):
        save_batch(self.batch(1.0), best_generation=7, keep=2)
        self.assertEqual(load_batches(8, 1), [])

    def test_only_the_newest_batches_are_kept_and_the_newest_comes_last(self):
        for value in (1.0, 2.0, 3.0):
            save_batch(self.batch(value), best_generation=7, keep=2)

        self.assertEqual([targets[0] for _, targets in load_batches(7, 5)], [2.0, 3.0])

    def test_a_restart_reuses_the_saved_batch_instead_of_playing_games(self):
        from unittest import mock
        from tzaar.learning import training
        from tzaar.nets.network import Network

        rated_nets = [(7, Network(hidden=8), 100.0)]
        save_batch(self.batch(5.0), best_generation=7, keep=2)

        with mock.patch.object(training, "generate_training_data", side_effect=AssertionError("played games")):
            (inputs, targets), = training.rebuild_batches(rated_nets, [], 8)

        self.assertEqual(targets[0], 5.0)

    def test_clearing_the_checkpoint_removes_saved_batches(self):
        save_batch(self.batch(1.0), best_generation=7, keep=2)
        clear_checkpoint()
        self.assertEqual(load_batches(7, 1), [])

class ServerTests(TemporaryModels):
    def test_only_trained_models_can_be_played(self):
        model_store.model_dir("untrained").mkdir()
        save_two_nets()
        models = available_nets()
        self.assertEqual([model["name"] for model in models], ["first"])
        self.assertEqual([entry[GENERATION_KEY] for entry in models[0][NETS_KEY]], [0, 1])

    def test_unknown_net_is_refused(self):
        with self.assertRaises(ValueError):
            NetPlayers().player_for("first", 99)
