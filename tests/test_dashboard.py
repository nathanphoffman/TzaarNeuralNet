"""Tests for starting and stopping training from the dashboard (no real training is started)."""
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

from tests.helpers import TemporaryModels
from tzaar.nets import model_store
from web import dashboard

class FakeProcess:
    def __init__(self):
        self.signals = []

    def poll(self):
        return None

    def wait(self):
        pass

    def send_signal(self, signal_number):
        self.signals.append(signal_number)

class FakeOutput:
    def join(self):
        pass

class ModelInfoTests(TemporaryModels):
    def test_info_has_the_note_and_the_settings_but_not_the_note_keys_in_settings(self):
        model_store.model_dir("first").mkdir()
        (model_store.model_dir("first") / "config.json").write_text(
            json.dumps({"label": "A label", "description": "Why", "seed": 3, "SEARCH_LABEL_SHARE": 0.3}))
        info = dashboard.model_info("first")
        self.assertEqual((info["label"], info["description"]), ("A label", "Why"))
        self.assertEqual(info["settings"], {"seed": 3, "SEARCH_LABEL_SHARE": 0.3})

    def test_a_model_without_a_config_has_empty_info(self):
        model_store.model_dir("second").mkdir()
        self.assertEqual(dashboard.model_info("second"), {"label": None, "description": None, "settings": {}})

    def test_progress_carries_the_info(self):
        model_store.model_dir("first").mkdir()
        self.assertIn("info", dashboard.model_progress("first"))

    def test_the_note_keys_are_not_settings_for_training(self):
        from tzaar.runs.config import apply_config
        apply_config({"label": "x", "description": "y"})

class TrainerTests(TemporaryModels):
    def setUp(self):
        super().setUp()
        model_store.model_dir("first").mkdir()
        self.trainer = dashboard.Trainer()
        self.started = []

        def fake_start_child(name, generations, fresh, workers=None):
            self.started.append((name, generations, fresh, workers))

            return FakeProcess(), FakeOutput()

        patcher = mock.patch.object(dashboard, "start_child", fake_start_child)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_start_runs_the_model_without_wiping_its_progress(self):
        self.trainer.start("first", 5)
        self.assertEqual(self.started, [("first", 5, False, None)])

    def test_the_number_of_workers_is_passed_on(self):
        self.trainer.start("first", None, 3)
        self.assertEqual(self.started, [("first", None, False, 3)])

    def test_a_model_already_training_is_not_started_twice(self):
        self.trainer.start("first")

        with self.assertRaises(ValueError):
            self.trainer.start("first")

        self.assertEqual(len(self.started), 1)

    def test_unknown_model_and_bad_generations_are_refused(self):
        with self.assertRaises(ValueError):
            self.trainer.start("nope")

        with self.assertRaises(ValueError):
            self.trainer.start("first", 0)

        for workers in (0, dashboard.MAX_WORKERS + 1):
            with self.assertRaises(ValueError):
                self.trainer.start("first", None, workers)

        self.assertEqual(self.started, [])

    def test_stop_interrupts_a_run_that_has_only_just_started(self):
        self.trainer.start("first")
        self.trainer.stop("first")
        self.assertEqual(len(self.trainer._children["first"].signals), 1)

    def test_stop_refuses_a_model_that_is_not_training(self):
        with self.assertRaises(ValueError):
            self.trainer.stop("first")

class RequestTests(TemporaryModels):
    def test_a_post_without_the_header_is_refused(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/train",
                                         data=json.dumps({"model": "first"}).encode(), method="POST")

        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)

        self.assertEqual(caught.exception.code, 400)

if __name__ == "__main__":
    unittest.main()
