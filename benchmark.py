"""Rate every player against every other, tournament style. Run: uv run benchmark.py

Players: random, the nets in baselines/, and the best net of every model in models/. Each pair plays
--games games (alternating who is White), then one rating per player is fitted to all the results at once,
with random fixed at 0. Prints a table and saves it to benchmark_results.json (or --output).
Only reads the models, never writes to them.
"""

import argparse
import itertools
import json
import os
from pathlib import Path

# one math thread per process, set before numpy loads: otherwise every game worker starts a thread per core
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np

from tzaar.nets import games as game_play
from tzaar.nets.games import win_rate
from tzaar.nets.model_store import MODELS_ROOT, list_models, load_net
from tzaar.nets.players import make_net_player, random_player
from tzaar.nets.rating import ELO_SCALE

BASELINES = Path(__file__).resolve().parent / "baselines"
DEFAULT_GAMES = 400
DEFAULT_WORKERS = 2
FIT_STEPS = 2000
FIT_STEP_SIZE = 20.0       # rating points moved per step, scaled by how far off the results are

def find_players():
    """(name, player) for random, every baseline net, and every model's best net."""

    players = [("random", random_player)]

    for path in sorted(BASELINES.glob("*.npz")):
        players.append((f"baseline {path.stem}", make_net_player(load_net(path))))

    for name in list_models():
        best = MODELS_ROOT / name / "model.npz"

        if best.exists():
            players.append((f"{name} best", make_net_player(load_net(best))))

    return players

def fit_ratings(count, results):
    """Ratings (player 0 fixed at 0) that best explain results = [(a, b, a's win rate, games)].
    Half a game's win and loss is added to every pair, so a 100% result can't push a rating to infinity."""

    ratings = np.zeros(count)

    for _ in range(FIT_STEPS):
        step = np.zeros(count)

        for a, b, score, games in results:
            expected = 1 / (1 + 10 ** (-(ratings[a] - ratings[b]) / ELO_SCALE))
            error = ((score * games + 0.5) / (games + 1) - expected) * games / (games + 1)
            step[a] += error
            step[b] -= error

        ratings += FIT_STEP_SIZE * step / max(1, count - 1)
        ratings -= ratings[0]

    return ratings

def run_tournament(players, games):
    results, total = [], len(players) * (len(players) - 1) // 2

    for number, (a, b) in enumerate(itertools.combinations(range(len(players)), 2), start=1):
        score = win_rate(players[a][1], players[b][1], games)
        results.append((a, b, score, games))
        print(f"[{number}/{total}] {players[a][0]} vs {players[b][0]}: {score:.1%}", flush=True)

    return results

def main():
    parser = argparse.ArgumentParser(description="Round-robin rating of the baselines and every model's best net.")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="games per pair")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="processes playing games at once")
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    game_play.WORKERS = args.workers
    players = find_players()
    names = [name for name, _ in players]
    print(f"{len(players)} players, {args.games} games per pair")
    results = run_tournament(players, args.games)
    ratings = fit_ratings(len(players), results)

    print("\nRating (random = 0)")

    for index in np.argsort(-ratings):
        print(f"{ratings[index]:8.0f}  {names[index]}")

    Path(args.output).write_text(json.dumps({
        "games_per_pair": args.games,
        "ratings": {names[i]: round(float(ratings[i]), 1) for i in range(len(players))},
        "results": [{"a": names[a], "b": names[b], "a_win_rate": score, "games": games} for a, b, score, games in results],
    }, indent=2))
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
