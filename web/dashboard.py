"""A live training dashboard: serves web/dashboard.html plus every model's progress as JSON.

Each model trains in its own process and writes its progress to models/<name>/progress.json
(see progress.py); this reads those files, so it works from any process."""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tzaar.learning.checkpoint import HISTORY_KEY
from tzaar.learning.checkpoint_paths import history_path
from tzaar.nets.model_store import list_models, model_dir
from tzaar.runs.progress import ELAPSED_KEY, RUNNING_KEY
from web.web_response import HTML_TYPE, HTTP_NOT_FOUND, send, send_json

DASHBOARD_HTML = Path(__file__).resolve().parent / "pages" / "dashboard.html"
PORT = 8002

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
            snapshot[ELAPSED_KEY] += time.time() - snapshot["written_at"]     # the file is only written between steps

            return snapshot

        snapshot.update(running=False, status="Stopped", summary="Training is not running. Run train.py to continue.")

        return snapshot

    if snapshot:
        return snapshot

    rows = json.loads(history_path(name).read_text())[HISTORY_KEY] if history_path(name).exists() else []

    return {"status": "Not running", RUNNING_KEY: False, "summary": "Training is not running.", ELAPSED_KEY: 0,
            "fraction": 1.0, "total_generations": None, "match_games": None, "generations": rows}

class Handler(BaseHTTPRequestHandler):
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
        server = ThreadingHTTPServer(("", PORT), Handler)
    except OSError:
        print(f"Dashboard not started: port {PORT} is already in use.")

        return

    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Training dashboard: http://localhost:{PORT}")
