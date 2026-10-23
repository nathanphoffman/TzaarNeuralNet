"""Tests for saving, loading and listing models."""

import numpy as np

from tests.helpers import TemporaryModels
from tzaar.nets import model_store
from tzaar.rules.encoding import NUM_FEATURES
from tzaar.nets.network import Network

class NetworkStoreTests(TemporaryModels):
    def test_saved_net_scores_positions_the_same_after_loading(self):
        network = Network(hidden=8)
        model_store.save_net(network, model_store.model_path())
        loaded = model_store.load_net(model_store.model_path())
        positions = np.random.default_rng(0).random((3, NUM_FEATURES)).astype(np.float32)
        np.testing.assert_allclose(network.predict(positions), loaded.predict(positions), rtol=1e-5)

    def test_older_net_with_fewer_inputs_is_padded_with_zeros(self):
        network = Network(hidden=8)
        model_store.model_dir().mkdir()
        np.savez(model_store.model_path(), W1=network.W1[:-5], b1=network.b1, W2=network.W2, b2=network.b2)
        loaded = model_store.load_net(model_store.model_path())
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
