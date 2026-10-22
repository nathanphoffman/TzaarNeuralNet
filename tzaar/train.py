"""Train the net. Run: uv run train.py

Progress is saved after every generation, so you can stop (Ctrl+C) and run it again to
keep going. Also starts a live dashboard at http://localhost:8002 (skip it with --no-dashboard).
"""
import argparse
import threading

from tzaar.dashboard import start_dashboard
from tzaar.training import NUM_GENERATIONS, train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=NUM_GENERATIONS,
                        help="how many more generations to run (default %(default)s)")
    parser.add_argument("--fresh", action="store_true", help="throw away saved progress and start over")
    parser.add_argument("--no-dashboard", action="store_true", help="don't start the live dashboard")
    args = parser.parse_args()

    if not args.no_dashboard:
        start_dashboard()
    train_model(args.generations, args.fresh)
    print("Done. The best net is saved in models/model.npz")

    if not args.no_dashboard:
        print("The dashboard stays up so you can look at the results. Press Ctrl+C to quit.")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
