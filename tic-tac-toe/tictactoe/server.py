"""A tiny web server so you can play against the trained net in a browser."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

from .game import legal_moves
from .model_store import MODEL_PATH, load_net, save_net
from .players import make_net_player
from .training import train_model

INDEX_HTML = Path(__file__).resolve().parent.parent / "web" / "index.html"
PORT = 8000


def get_net():
    """Load the saved net, or train and save one if there isn't one yet."""
    if MODEL_PATH.exists():
        print("Loaded", MODEL_PATH)
        return load_net()
    net = train_model()
    save_net(net)
    return net


def make_handler(net_player):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._send(200, "text/html", INDEX_HTML.read_bytes())

        def do_POST(self):
            """Body: {"board": [9 ints], "ai": 1 or -1}. Replies with {"move": square}."""
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            board, ai_side = request["board"], request["ai"]
            move = int(net_player(np.array(board), ai_side)) if legal_moves(board) else -1
            self._send(200, "application/json", json.dumps({"move": move}).encode())

        def _send(self, status, content_type, body):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(body)
    return Handler


def run_server():
    net_player = make_net_player(get_net())
    print(f"Open http://localhost:{PORT}")
    HTTPServer(("", PORT), make_handler(net_player)).serve_forever()
