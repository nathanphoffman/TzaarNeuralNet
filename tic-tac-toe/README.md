# Tic-Tac-Toe Self-Play

A small neural network that learns tic-tac-toe, plus a web page to play against it. Only needs `numpy` (uv installs it for you).

## Run

From this folder:

```
uv run train.py    # train a model and save it to models/model.npz
uv run serve.py    # open http://localhost:8000 and play
```

`serve.py` trains a model automatically if `models/model.npz` doesn't exist. Delete that file to retrain from scratch.

## How training works

1. Play 20,000 random-vs-random games and record each position with who won.
2. Train generation 0 on that data.
3. For 5 generations: the best net plays itself (with 20% random moves), a fresh net trains on the last 3 batches of data, and it replaces the best net only if it beats it and doesn't get worse against a random player.

Settings are at the top of `tictactoe/training.py`.

## Layout

```
train.py, serve.py     entry points
tictactoe/
  game.py              rules: winner, legal moves, board encoding
  network.py           the neural network
  players.py           random player and net player
  games.py             play games, make training data, run matches
  training.py          the generation loop
  model_store.py       save / load the model
  server.py            the web server
web/index.html         the game board page
models/                saved models (git-ignored)
```
