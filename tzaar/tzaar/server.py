"""A tiny web server so you can play against the trained net in a browser.

API (all POST, JSON in and out; the browser sends the whole game state each time):
  /api/new               -> a fresh random game
  /api/move {state, move}-> your move applied
  /api/ai   {state, net} -> the net {model, generation} picks and plays a move
GET /api/info            -> the models you can play and their accepted nets, with ratings
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .board import POINTS
from .checkpoint import history_path
from .game import State, new_game
from .model_store import list_models, load_net, model_dir
from .players import make_net_player

INDEX_HTML = Path(__file__).resolve().parent.parent / "web" / "index.html"
PORT = 8001


def available_nets():
    """Every model and its accepted nets, from the training histories: [{name, nets: [{generation, rating}]}],
    each model's nets oldest first. Models with no accepted nets yet are left out."""
    models = []
    for name in list_models():
        try:
            nets = json.loads(history_path(name).read_text())["nets"]
        except (OSError, ValueError, KeyError):
            continue
        if nets:
            models.append({"name": name, "nets": [{"generation": n["generation"], "rating": round(n["rating"])}
                                                  for n in nets]})
    return models


class NetPlayers:
    """The players for the nets you can pick, loaded the first time they are used. The list of nets is read
    again for each new one, so nets accepted while the server is up can be played too."""

    def __init__(self):
        self.players = {}

    def get(self, name, generation):
        if (name, generation) not in self.players:
            if not any(m["name"] == name and any(n["generation"] == generation for n in m["nets"])
                       for m in available_nets()):
                raise ValueError("unknown net")
            path = model_dir(name) / "checkpoint" / f"net_gen{generation:04d}.npz"
            self.players[(name, generation)] = make_net_player(load_net(path))
        return self.players[(name, generation)]


def game_response(state, last_move=None):
    return {"state": state.to_dict(), "points": POINTS, "last_move": last_move,
            "legal_moves": [list(move) for move in state.legal_moves()]}


def make_handler(net_players):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/info":
                return self._send_json(200, available_nets())
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
                net = request["net"]
                move = net_players.get(net["model"], int(net["generation"]))(state)
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
    models = available_nets()
    if not models:
        raise SystemExit("No trained nets yet. Run: uv run train.py")
    print("Playable:", ", ".join(f"{m['name']} ({len(m['nets'])} nets)" for m in models))
    print(f"Open http://localhost:{PORT}")
    HTTPServer(("", PORT), make_handler(NetPlayers())).serve_forever()
