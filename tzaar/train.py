"""Train the net. Run: uv run train.py

Keeps running generations until you stop it with Ctrl+C. Progress is saved after every
generation, so running it again carries on from the last finished one.
Also starts a live dashboard at http://localhost:8002 (skip it with --no-dashboard).
"""
import argparse
import threading

from tzaar.dashboard import start_dashboard
from tzaar.training import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=None,
                        help="stop after this many more generations (default: keep going until Ctrl+C)")
    parser.add_argument("--fresh", action="store_true", help="throw away saved progress and start over")
    parser.add_argument("--no-dashboard", action="store_true", help="don't start the live dashboard")
    args = parser.parse_args()

    if not args.no_dashboard:
        start_dashboard()
    try:
        train_model(args.generations, args.fresh)
    except KeyboardInterrupt:
        print("\nStopped. Everything up to the last finished generation is saved; run again to continue.")
        raise SystemExit(0)
    print("Done. The best net is saved in models/model.npz")

    if not args.no_dashboard:
        print("The dashboard stays up so you can look at the results. Press Ctrl+C to quit.")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
