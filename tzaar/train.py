"""Train the net and save it. Run: uv run train.py

Also starts a live dashboard at http://localhost:8002 (skip it with --no-dashboard).
"""
import argparse
import threading

from tzaar.dashboard import start_dashboard
from tzaar.model_store import save_net
from tzaar.training import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-dashboard", action="store_true", help="don't start the live dashboard")
    args = parser.parse_args()

    if not args.no_dashboard:
        start_dashboard()
    save_net(train_model())
    print("Model saved.")

    if not args.no_dashboard:
        print("The dashboard stays up so you can look at the results. Press Ctrl+C to quit.")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
