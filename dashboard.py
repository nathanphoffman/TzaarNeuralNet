"""The training dashboard on its own. Run: uv run dashboard.py

Opens http://localhost:8002, where you pick a model and press Start training or Stop. Nothing trains until you do.
(train.py starts this same dashboard too, and starts training for the models you name or all of them.)
"""

import threading

from web.dashboard import start_dashboard

if __name__ == "__main__":
    start_dashboard()

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("Dashboard closed.")
