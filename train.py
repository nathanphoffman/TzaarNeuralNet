"""Train the nets. Run: uv run train.py

Trains every model in models/ side by side (one process each, so they know nothing about each other), or just
the ones you name: uv run train.py --models evolution-1. Keeps running generations until you stop it with
Ctrl+C. Progress is saved after every generation, so running it again carries on from the last finished one.
Also starts a live dashboard at http://localhost:8002 (skip it with --no-dashboard).
"""

import argparse
import threading

from web.dashboard import start_dashboard
from tzaar.nets.model_store import list_models
from tzaar.runs.runner import run_model, run_models

def find_problem(args, names, available):
    """What is wrong with the command line, or None if it is fine."""

    unknown = [name for name in names if name not in available]

    if unknown:
        return f"unknown model(s) {unknown}; models are folders in models/: {available}"

    if not names:
        return "no models yet: create a folder like models/evolution-1 (it can start empty) and run again"

    if args.fresh and not args.models:
        return "--fresh deletes saved progress; say which models with --models"

    return None

def keep_dashboard_up():
    print("The dashboard stays up so you can look at the results. Press Ctrl+C to quit.")

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("Dashboard closed.")

def train_all(args, names):
    if not args.no_dashboard:
        start_dashboard()

    print("Training:", ", ".join(names))

    if not run_models(names, args.generations, args.fresh):
        print("\nStopped. Everything up to the last finished generation is saved; run again to continue.")

        return

    if not args.no_dashboard:
        keep_dashboard_up()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="which models to train (default: all of them)")
    parser.add_argument("--generations", type=int, default=None,
                        help="stop after this many more generations (default: keep going until Ctrl+C)")

    parser.add_argument("--fresh", action="store_true",
                        help="throw away saved progress and start over (needs --models, so no model is wiped by accident)")

    parser.add_argument("--no-dashboard", action="store_true", help="don't start the live dashboard")
    parser.add_argument("--child", help=argparse.SUPPRESS)       # used internally: train this one model in this process
    args = parser.parse_args()

    if args.child:
        run_model(args.child, args.generations, args.fresh)
    else:
        available = list_models()
        names = args.models or available
        problem = find_problem(args, names, available)

        if problem:
            parser.error(problem)

        train_all(args, names)
