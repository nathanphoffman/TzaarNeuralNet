"""Training progress, shared between the training code and the dashboard page."""
import threading
import time


class Progress:
    def __init__(self):
        self._lock = threading.Lock()
        self.start(0, 0, 0)

    def start(self, total_generations, total_steps, match_games, rows=()):
        with self._lock:
            self.total_generations = total_generations
            self.total_steps = total_steps
            self.match_games = match_games
            self.steps_started = 0
            self.status = "Starting..."
            self.generations = list(rows)    # one row per generation, generation 0 first
            self.summary = None
            self.running = True
            self.started_at = time.time()
            self.finished_at = None

    def step(self, status):
        """Call at the start of each step; the step before it counts as done."""
        with self._lock:
            self.steps_started += 1
            self.status = status

    def add_generation(self, row):
        with self._lock:
            self.generations.append(row)

    def finish(self, summary):
        with self._lock:
            self.running = False
            self.summary = summary
            self.status = "Done"
            self.finished_at = time.time()

    def snapshot(self):
        with self._lock:
            end = self.finished_at or time.time()
            fraction = 1.0 if not self.running else max(0, self.steps_started - 1) / max(1, self.total_steps)
            return {"status": self.status, "running": self.running, "summary": self.summary,
                    "elapsed": end - self.started_at, "fraction": fraction,
                    "total_generations": self.total_generations, "match_games": self.match_games,
                    "generations": list(self.generations)}


progress = Progress()
