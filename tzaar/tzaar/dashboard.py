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

from .checkpoint import history_path
from .model_store import list_models, model_dir

DASHBOARD_HTML = Path(__file__).resolve().parent.parent / "web" / "dashboard.html"
PORT = 8002


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def model_progress(name):
    """The latest progress of a model's training run. If no run has written one (or it was stopped), the
    saved history is shown instead."""
    try:
        snapshot = json.loads((model_dir(name) / "progress.json").read_text())
    except (OSError, ValueError):
        snapshot = None
    if snapshot and snapshot["running"]:
        if _alive(snapshot["pid"]):
            snapshot["elapsed"] += time.time() - snapshot["written_at"]     # the file is only written between steps
            return snapshot
        snapshot.update(running=False, status="Stopped", summary="Training is not running. Run train.py to continue.")
        return snapshot
    if snapshot:
        return snapshot
    try:
        rows = json.loads(history_path(name).read_text())["history"]
    except (OSError, ValueError, KeyError):
        rows = []
    return {"status": "Not running", "running": False, "summary": "Training is not running.", "elapsed": 0,
            "fraction": 1.0, "total_generations": None, "match_games": None, "generations": rows}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/models":
            self._send("application/json", json.dumps(list_models()).encode())
        elif url.path == "/api/progress":
            name = parse_qs(url.query).get("model", [""])[0]
            if name not in list_models():
                return self._send("application/json", b'{"error": "unknown model"}', 404)
            self._send("application/json", json.dumps(model_progress(name)).encode())
        else:
            self._send("text/html", DASHBOARD_HTML.read_bytes())

    def _send(self, content_type, body, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

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
