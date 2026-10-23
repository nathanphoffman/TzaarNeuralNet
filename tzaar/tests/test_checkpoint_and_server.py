"""Tests for resuming training from a checkpoint and for listing the nets you can play."""

from tests.helpers import TemporaryModels, save_two_nets
from tzaar import model_store
from tzaar.checkpoint import GENERATION_KEY, NETS_KEY, clear_checkpoint, load_checkpoint
from tzaar.server import NetPlayers, available_nets

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
