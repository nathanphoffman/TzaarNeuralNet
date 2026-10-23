"""Shared by the tests: a temporary models folder, so no test touches the real one."""
import tempfile
import unittest
from pathlib import Path

from tzaar.nets import model_store
from tzaar.learning.checkpoint import GENERATION_KEY, save_checkpoint
from tzaar.nets.network import Network

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

def save_two_nets():
    history = [history_row(0), history_row(1, 200.0)]
    rated_nets = [(0, Network(hidden=8), 100.0), (1, Network(hidden=8), 200.0)]
    save_checkpoint(history, rated_nets, [])

    return history
