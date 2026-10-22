"""A tiny web server so you can play against the trained net in a browser.

API (all POST, JSON in and out; the browser sends the whole game state each time):
  /api/new               -> a fresh random game
  /api/move {state, move}-> your move applied
  /api/ai   {state}      -> the net picks and plays a move
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .board import POINTS
from .game import State, new_game
from .model_store import MODEL_PATH, load_net, save_net
from .players import make_net_player
from .training import train_model

INDEX_HTML = Path(__file__).resolve().parent.parent / "web" / "index.html"
PORT = 8001


def get_net():
    """Load the saved net, or train and save one if there isn't one yet."""
    if MODEL_PATH.exists():
        print("Loaded", MODEL_PATH)
        return load_net()
    net = train_model()
    save_net(net)
    return net


def game_response(state, last_move=None):
    return {"state": state.to_dict(), "points": POINTS, "last_move": last_move,
            "legal_moves": [list(move) for move in state.legal_moves()]}


def make_handler(net_player):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._send(200, "text/html", INDEX_HTML.read_bytes())

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
            try:
                self._send_json(200, self._handle_api(request))
            except (KeyError, ValueError) as error:
                self._send_json(400, {"error": str(error)})

        def _handle_api(self, request):
            if self.path == "/api/new":
                return game_response(new_game())
            state = State.from_dict(request["state"])
            if self.path == "/api/move":
                move = tuple(request["move"])
                if move not in state.legal_moves():
                    raise ValueError("illegal move")
            elif self.path == "/api/ai":
                move = net_player(state)
            else:
                raise ValueError("unknown path")
            state.play(move)
            return game_response(state, last_move=list(move))

        def _send_json(self, status, data):
            self._send(status, "application/json", json.dumps(data).encode())

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
