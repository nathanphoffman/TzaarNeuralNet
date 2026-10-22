"""A live training dashboard: serves web/dashboard.html plus the progress as JSON."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .progress import progress

DASHBOARD_HTML = Path(__file__).resolve().parent.parent / "web" / "dashboard.html"
PORT = 8002


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/progress":
            self._send("application/json", json.dumps(progress.snapshot()).encode())
        else:
            self._send("text/html", DASHBOARD_HTML.read_bytes())

    def _send(self, content_type, body):
        self.send_response(200)
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
