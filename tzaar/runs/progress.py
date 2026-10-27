"""Training progress, shared between the training code and the dashboard page.

The dashboard runs in a different process from the training (one process per model), so when a file is
attached with attach() the progress is also written to it after every change, and the dashboard reads it."""

import json
import os
import threading
import time
DEFAULT_STEPS_PER_GENERATION = 3      # self-play, training, testing (see training.py)
RUNNING_KEY = "running"        # keys of the progress snapshot that the dashboard also reads
ELAPSED_KEY = "elapsed"
STEP_CAP = 0.95                # a step that runs longer than last time waits at 95% of its share until it finishes

def fraction_now(snapshot, now):
    """How far along the bar is. Once every step of a generation has been timed once, the bar is weighted by how
    long each step took last time and moves smoothly inside the current step. Until then each step counts the same."""

    if not snapshot[RUNNING_KEY]:
        return 1.0

    steps_per_generation, durations = snapshot["steps_per_generation"], snapshot["step_durations"]
    done_steps = max(0, snapshot["steps_started"] - 1)
    into_generations = done_steps - snapshot["setup_steps"]          # steps done since the self-play generations began
    total_steps = snapshot["total_steps"]
    timed = into_generations >= 0 and snapshot["step_started_at"] is not None and None not in durations
    games_total = snapshot.get("games_total")
    games_share = snapshot["games_done"] / games_total if games_total else 0.0      # of the batch being played now

    if into_generations < 0:                   # setting up: random games, or rebuilding the training data, are most of it
        return min(1.0, (done_steps + games_share) / max(1, snapshot["setup_steps"] if total_steps is None else total_steps))

    if timed:
        position, finished = into_generations % steps_per_generation, into_generations // steps_per_generation
        into_step = min(max(0.0, now - snapshot["step_started_at"]), STEP_CAP * durations[position])
        generation_fraction = (sum(durations[:position]) + into_step) / sum(durations)
    else:
        position, finished = max(0, into_generations) % steps_per_generation, max(0, into_generations) // steps_per_generation
        generation_fraction = (position + (games_share if position == 0 else 0.0)) / steps_per_generation      # step 0 is one batch of games

    if total_steps is None:
        return generation_fraction             # no end in sight: the bar shows the current generation

    if not timed:
        return (done_steps + (games_share if position == 0 else 0.0)) / max(1, total_steps)

    generations_to_run = (total_steps - snapshot["setup_steps"]) / steps_per_generation

    return min(1.0, (finished + generation_fraction) / max(1.0, generations_to_run))

class Progress:
    def __init__(self):
        self._lock = threading.Lock()
        self._path = None
        self._seed = None            # step lengths from the previous run of this model, a first guess for the bar
        self.start(0, 0, 0)

    def attach(self, path):
        """Also write the progress to this file after every change. The step lengths in the file from the last
        run are kept as a first guess for the bar, until this run has timed its own."""
        self._path = path

        try:
            previous = json.loads(path.read_text()).get("step_durations")
        except (OSError, ValueError):
            return

        if isinstance(previous, list) and all(isinstance(length, (int, float)) for length in previous):
            self._seed = previous

    def _save(self):
        """Call with the lock held."""

        if self._path is None:
            return

        snapshot = dict(self._snapshot(), pid=os.getpid(), written_at=time.time())
        temp = self._path.with_suffix(".tmp")
        temp.write_text(json.dumps(snapshot))
        temp.replace(self._path)

    def start(self, total_generations, total_steps, match_games, rows=(), setup_steps=0, steps_per_generation=DEFAULT_STEPS_PER_GENERATION):
        """total_generations / total_steps are None when training runs until it is stopped; the
        progress bar then shows how far through the current generation we are."""

        with self._lock:
            self.total_generations = total_generations
            self.total_steps = total_steps
            self.setup_steps = setup_steps
            self.steps_per_generation = steps_per_generation
            self.match_games = match_games

            self.steps_started = 0
            self.step_started_at = None
            seeded = self._seed is not None and len(self._seed) == steps_per_generation
            self.step_durations = list(self._seed) if seeded else [None] * steps_per_generation    # how long each step took last time
            self.games_done = self.games_total = None       # of the batch of games being played now, if one is
            self.detail = None                              # a line about this generation, e.g. who the best net plays against

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
            now = time.time()
            into_generations = self.steps_started - 1 - self.setup_steps    # the step that is ending, counted from the first generation

            if self.step_started_at is not None and into_generations >= 0:
                self.step_durations[into_generations % self.steps_per_generation] = now - self.step_started_at

            self.steps_started += 1
            self.step_started_at = now
            self.status = status
            self.games_done = self.games_total = None
            self._save()

    def set_detail(self, text):
        """A line of extra information for the dashboard; it stays until the next one."""

        with self._lock:
            self.detail = text
            self._save()

    def set_games(self, done, total):
        """How many games of the batch being played are finished. Called about once a second while games are played."""

        with self._lock:
            self.games_done, self.games_total = done, total
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
        """The progress as a dict for the dashboard (see fraction_now for the bar)."""

        now = time.time()
        end_time = self.finished_at or now

        snapshot = {"status": self.status, RUNNING_KEY: self.running, "summary": self.summary,
                    ELAPSED_KEY: end_time - self.started_at,
                    "total_generations": self.total_generations, "match_games": self.match_games,
                    "generations": list(self.generations),
                    # what fraction_now needs, so the dashboard can recompute the bar between saves
                    "steps_started": self.steps_started, "total_steps": self.total_steps, "setup_steps": self.setup_steps,
                    "steps_per_generation": self.steps_per_generation, "step_started_at": self.step_started_at,
                    "step_durations": list(self.step_durations), "games_done": self.games_done,
                    "games_total": self.games_total, "detail": self.detail}

        snapshot["fraction"] = fraction_now(snapshot, now)

        return snapshot

progress = Progress()
