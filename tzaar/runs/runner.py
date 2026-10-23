"""Training several models side by side: one process per model, so they share nothing.

train.py starts a child process for each model (run_model below runs inside it) and prefixes each line
of its output with the model's name."""

import subprocess
import sys
import threading
from pathlib import Path

from tzaar.runs.config import apply_config, load_config
from tzaar.nets.model_store import model_dir, use_model
from tzaar.runs.progress import progress
from tzaar.learning.training import train_model

TRAIN_SCRIPT = Path(__file__).resolve().parent.parent.parent / "train.py"

def run_model(name, generations, fresh):
    """Runs inside a model's own process."""
    use_model(name)
    apply_config(load_config(name))
    progress.attach(model_dir() / "progress.json")

    try:
        train_model(generations, fresh)
    except KeyboardInterrupt:
        print("Stopped. Everything up to the last finished generation is saved; run again to continue.")

        raise SystemExit(0)

    print("Done. The best net is saved in", model_dir() / "model.npz")

def _copy_output(process, name):
    for line in process.stdout:
        print(f"[{name}] {line}", end="", flush=True)

def start_child(name, generations, fresh):
    """Start the training process of one model. Returns it with the thread that copies its output."""

    command = [sys.executable, "-u", str(TRAIN_SCRIPT), "--child", name]

    if generations is not None:
        command += ["--generations", str(generations)]

    if fresh:
        command.append("--fresh")

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    thread = threading.Thread(target=_copy_output, args=(process, name), daemon=True)
    thread.start()

    return process, thread

def wait_for(children):
    for process, thread in children:
        process.wait()
        thread.join()

def run_models(names, generations, fresh):
    """Start a process per model and wait until all of them have finished (or Ctrl+C stopped them)."""

    children = [start_child(name, generations, fresh) for name in names]

    try:
        wait_for(children)
    except KeyboardInterrupt:       # Ctrl+C also reaches the children (same terminal); let them save and exit
        wait_for(children)

        raise
