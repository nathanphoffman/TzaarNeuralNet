"""Training progress, shared between the training code and the dashboard page.

The dashboard runs in a different process from the training (one process per model), so when a file is
attached with attach() the progress is also written to it after every change, and the dashboard reads it."""

import json
import os
import threading
import time
RUNNING_KEY = "running"        # keys of the progress snapshot that the dashboard also reads
ELAPSED_KEY = "elapsed"

class Progress:
    def __init__(self):
        self._lock = threading.Lock()
        self._path = None
        self.start(0, 0, 0)

    def attach(self, path):
        """Also write the progress to this file after every change."""
        self._path = path

    def _save(self):
        """Call with the lock held."""

        if self._path is None:
            return

        snapshot = dict(self._snapshot(), pid=os.getpid(), written_at=time.time())
        temp = self._path.with_suffix(".tmp")
        temp.write_text(json.dumps(snapshot))
        temp.replace(self._path)

    def start(self, total_generations, total_steps, match_games, rows=(), setup_steps=0, steps_per_generation=3):
        """total_generations / total_steps are None when training runs until it is stopped; the
        progress bar then shows how far through the current generation we are."""

        with self._lock:
            self.total_generations = total_generations
            self.total_steps = total_steps
            self.setup_steps = setup_steps
            self.steps_per_generation = steps_per_generation
            self.match_games = match_games

            self.steps_started = 0

            self.status = "Starting..."
            self.generations = list(rows)    # one row per generation, generation 0 first
            self.summary = None
            self.running = True

            self.started_at = time.time()
            self.finished_at = None
            self._save()

    def step(self, status):
        """Call at the start of each step; the step before it counts as done."""

        with self._lock:
            self.steps_started += 1
            self.status = status
            self._save()

    def set_status(self, status):
        """Change the status text without counting a new step."""

        with self._lock:
            self.status = status
            self._save()

    def add_generation(self, result_row):
        with self._lock:
            self.generations.append(result_row)
            self._save()

    def finish(self, summary):
        with self._lock:
            self.running = False
            self.summary = summary
            self.status = "Done"
            self.finished_at = time.time()
            self._save()

    def snapshot(self):
        with self._lock:
            return self._snapshot()

    def _snapshot(self):
        """The progress as a dict for the dashboard. The bar is by steps done; for a run with no end it shows the
        steps done in the current generation instead."""

        end_time = self.finished_at or time.time()

        if not self.running:
            fraction = 1.0
        elif self.total_steps is None:
            done_in_generation = max(0, self.steps_started - 1 - self.setup_steps) % self.steps_per_generation
            fraction = done_in_generation / self.steps_per_generation
        else:
            fraction = max(0, self.steps_started - 1) / max(1, self.total_steps)

        return {"status": self.status, RUNNING_KEY: self.running, "summary": self.summary,
                ELAPSED_KEY: end_time - self.started_at, "fraction": fraction,
                "total_generations": self.total_generations, "match_games": self.match_games,
                "generations": list(self.generations)}

progress = Progress()
