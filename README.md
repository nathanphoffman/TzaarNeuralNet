# TzaarNet

Small neural networks that learn board games through self-play, each with a web page to play against the result. Pure Python + numpy, run with [uv](https://docs.astral.sh/uv/).

| Project | What it is | Web page |
|---|---|---|
| [`tic-tac-toe/`](tic-tac-toe) | The simple starter version, used to get the training loop right | http://localhost:8000 |
| [`tzaar/`](tzaar) | The real goal: [Tzaar](https://en.wikipedia.org/wiki/TZAAR) | http://localhost:8001 |

## Run a project

```
cd tic-tac-toe     # or: cd tzaar
uv run train.py    # train a model (saved to models/model.npz)
uv run serve.py    # play against it in your browser
```

`serve.py` trains a model automatically if none is saved yet. See each project's README for details.

## How the training works

1. Play random-vs-random games and train a first net on who won.
2. Repeat for several generations: the best net plays itself, a fresh net trains on those games, and it replaces the best net only if it beats it (and isn't worse against a random player).

The projects are independent and share no code.
