"""A live training dashboard: serves web/dashboard.html plus every model's progress as JSON.

Each model trains in its own process and writes its progress to models/<name>/progress.json
(see progress.py); this reads those files, so it works from any process."""

import json
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tzaar.learning.checkpoint import HISTORY_KEY
from tzaar.learning.checkpoint_paths import history_path
from tzaar.nets.model_store import list_models, model_dir
from tzaar.runs.progress import ELAPSED_KEY, RUNNING_KEY, bars_now
from tzaar.runs.runner import start_child
from web.web_response import HTML_TYPE, HTTP_BAD_REQUEST, HTTP_NOT_FOUND, send, send_json

DASHBOARD_HTML = Path(__file__).resolve().parent / "pages" / "dashboard.html"
PORT = 8002
HOST = "127.0.0.1"                # this machine only: the page can start and stop training
MAX_WORKERS = 64
REQUEST_HEADER = "X-Tzaar"        # a page on another site can't send this without our permission, so it can't start training

def _alive(process_id):
    """Signal 0 only checks that the process exists. Training processes are ours, so no other error can happen."""

    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False

    return True

def model_progress(name):
    """The latest progress of a model's training run. If no run has written one (or it was stopped), the
    saved history is shown instead."""

    progress_path = model_dir(name) / "progress.json"
    snapshot = json.loads(progress_path.read_text()) if progress_path.exists() else None

    if snapshot and snapshot[RUNNING_KEY]:
        if _alive(snapshot["pid"]):
            now = time.time()
            snapshot[ELAPSED_KEY] += now - snapshot["written_at"]     # the file is only written between steps
            snapshot["step_fraction"], snapshot["fraction"] = bars_now(snapshot, now)

            return snapshot

        snapshot.update(running=False, status="Stopped", summary="Training is not running. Run train.py to continue.")

        return snapshot

    if snapshot:
        return snapshot

    rows = json.loads(history_path(name).read_text())[HISTORY_KEY] if history_path(name).exists() else []

    return {"status": "Not running", RUNNING_KEY: False, "summary": "Training is not running.", ELAPSED_KEY: 0,
            "fraction": 1.0, "step_fraction": 1.0, "total_generations": None, "match_games": None, "generations": rows}

class Trainer:
    """Starts and stops training runs for the page. A run started here is a child process, the same as with train.py."""

    def __init__(self):
        self._lock = threading.Lock()
        self._children = {}         # model name -> the process this dashboard started for it

    def _known(self, name):
        if name not in list_models():
            raise ValueError("unknown model")

    def is_running(self, name):
        child = self._children.get(name)

        return (child is not None and child.poll() is None) or bool(model_progress(name)[RUNNING_KEY])

    def start(self, name, generations=None, workers=None):
        """Train a model for `generations` more generations (until stopped if None). It continues from its checkpoint.
        `workers` is the number of game-playing processes (the model's config.json decides if None)."""

        self._known(name)

        if generations is not None and generations < 1:
            raise ValueError("generations must be at least 1")

        if workers is not None and not 1 <= workers <= MAX_WORKERS:
            raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")

        with self._lock:                # two quick clicks must not start two runs
            if self.is_running(name):
                raise ValueError(f"{name} is already training")

            process, output = start_child(name, generations, False, workers)
            self._children[name] = process

            # collect the process when it ends, so it doesn't stay behind as a zombie that still looks alive
            threading.Thread(target=lambda: (output.join(), process.wait()), daemon=True).start()

    def stop(self, name):
        """Ask a training run to stop, the same as Ctrl+C: it saves after its current step and exits."""

        self._known(name)
        progress = model_progress(name)

        if not progress[RUNNING_KEY]:
            child = self._children.get(name)

            if child is None or child.poll() is not None:
                raise ValueError(f"{name} is not training")

            return child.send_signal(signal.SIGINT)        # started a moment ago, hasn't written its progress yet

        os.kill(progress["pid"], signal.SIGINT)

trainer = Trainer()

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """/api/train {model, generations, workers} starts a run, /api/stop {model} stops it."""

        length = int(self.headers.get("Content-Length", 0))

        try:
            if self.headers.get(REQUEST_HEADER) != "1":
                raise ValueError("missing header")

            request = json.loads(self.rfile.read(length) or b"{}")

            if self.path == "/api/train":
                trainer.start(request["model"], request.get("generations"), request.get("workers"))
            elif self.path == "/api/stop":
                trainer.stop(request["model"])
            else:
                raise ValueError("unknown path")

            send_json(self, {"ok": True}, no_store=True)
        except (KeyError, ValueError) as error:
            send_json(self, {"error": str(error)}, status=HTTP_BAD_REQUEST, no_store=True)

    def do_GET(self):
        """/api/models lists the models, /api/progress?model=<name> gives one model's progress, anything else is the page."""
        parsed_url = urlparse(self.path)

        if parsed_url.path == "/api/models":
            send_json(self, list_models(), no_store=True)
        elif parsed_url.path == "/api/progress":
            name = parse_qs(parsed_url.query).get("model", [""])[0]

            if name not in list_models():
                return send_json(self, {"error": "unknown model"}, status=HTTP_NOT_FOUND, no_store=True)

            send_json(self, model_progress(name), no_store=True)
        else:
            send(self, HTML_TYPE, DASHBOARD_HTML.read_bytes(), no_store=True)

    def log_message(self, *args):
        pass        # the page polls every second or so; don't flood the training output

def start_dashboard():
    """Serve the dashboard from a background thread so training can keep running."""

    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError:
        print(f"Dashboard not started: port {PORT} is already in use.")

        return

    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Training dashboard: http://localhost:{PORT}")
