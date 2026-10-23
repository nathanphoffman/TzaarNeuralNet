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

from tzaar.rules.board import POINTS
from tzaar.learning.checkpoint import GENERATION_KEY, NETS_KEY, RATING_KEY
from tzaar.learning.checkpoint_paths import history_path, net_path
from tzaar.rules.game import State, new_game
from tzaar.nets.model_store import list_models, load_net
from tzaar.nets.players import make_net_player
from web.web_response import HTML_TYPE, HTTP_BAD_REQUEST, send, send_json

INDEX_HTML = Path(__file__).resolve().parent / "pages" / "index.html"
PORT = 8001

def accepted_nets(name):
    """A model's accepted nets as [{generation, rating}], oldest first (none if it has not been trained yet)."""

    if not history_path(name).exists():
        return []

    nets = json.loads(history_path(name).read_text())[NETS_KEY]

    return [{GENERATION_KEY: net[GENERATION_KEY], RATING_KEY: round(net[RATING_KEY])} for net in nets]

def available_nets():
    """Every model that has accepted nets, from the training histories: [{name, nets: [{generation, rating}]}].
    Models that have none yet are left out."""

    models = [{"name": name, NETS_KEY: accepted_nets(name)} for name in list_models()]

    return [model for model in models if model[NETS_KEY]]

class NetPlayers:
    """The players for the nets you can pick, loaded the first time they are used. The list of nets is read
    again for each new one, so nets accepted while the server is up can be played too."""

    def __init__(self):
        self.players = {}

    def player_for(self, name, generation):
        """The player for one accepted net of a model; a net that is not in the model's history is refused."""

        if (name, generation) not in self.players:
            if generation not in [net[GENERATION_KEY] for net in accepted_nets(name)]:
                raise ValueError("unknown net")

            self.players[(name, generation)] = make_net_player(load_net(net_path("net", generation, name)))

        return self.players[(name, generation)]

def game_response(state, last_move=None):
    return {"state": state.to_dict(), "points": POINTS, "last_move": last_move,
            "legal_moves": [list(move) for move in state.legal_moves()]}

class Handler(BaseHTTPRequestHandler):
    net_players = NetPlayers()      # shared by every request

    def do_GET(self):
        if self.path == "/api/info":
            return send_json(self, available_nets())

        send(self, HTML_TYPE, INDEX_HTML.read_bytes())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")

        try:
            send_json(self, self._handle_api(request))
        except (KeyError, ValueError) as error:
            send_json(self, {"error": str(error)}, status=HTTP_BAD_REQUEST)      # a bad request: tell the page what was wrong

    def _handle_api(self, request):
        """Run one of the /api/ calls from the module docstring and return the new game state."""

        if self.path == "/api/new":
            return game_response(new_game())

        state = State.from_dict(request["state"])

        if self.path == "/api/move":
            move = tuple(request["move"])

            if move not in state.legal_moves():
                raise ValueError("illegal move")
        elif self.path == "/api/ai":
            network = request["net"]
            move = self.net_players.player_for(network["model"], int(network["generation"]))(state)
        else:
            raise ValueError("unknown path")

        state.play(move)

        return game_response(state, last_move=list(move))

def run_server():
    models = available_nets()

    if not models:
        raise SystemExit("No trained nets yet. Run: uv run train.py")

    print("Playable:", ", ".join(f"{model['name']} ({len(model[NETS_KEY])} nets)" for model in models))
    print(f"Open http://localhost:{PORT}")
    HTTPServer(("", PORT), Handler).serve_forever()
