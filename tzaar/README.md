# Tzaar Self-Play

A small neural network that learns [Tzaar](https://en.wikipedia.org/wiki/TZAAR), plus a web page to play against it. Same idea as `../tic-tac-toe`. Only needs `numpy` (uv installs it for you).

## Run

From this folder:

```
uv run train.py    # train a model and save it to models/model.npz
uv run serve.py    # open http://localhost:8001 and play
```

`serve.py` trains a model automatically if `models/model.npz` doesn't exist. Delete that file to retrain from scratch.

## Rules used

- 60 points on a hexagonal board; the centre is a gap that nothing can land on or jump over.
- Each side has 6 Tzaars, 9 Tzaaras, 15 Totts.
- A turn is two actions: a capture, then a capture, a stack, or a pass. White's first turn is one capture only.
- A piece slides in a straight line over empty points and must land on the first piece it meets. Capture = enemy stack no taller than yours. Stack = merge onto your own stack.
- You lose if you can't capture at the start of your turn, or you lose your last stack of one type.
- **Setup is random** each game (the rules allow any arrangement), not the fixed layout from the printed rulebook.

## How training works

1. Play random-vs-random games and record each position with who won.
2. Train generation 0 on that data.
3. For each generation: the best net plays itself (with 20% random moves), a fresh net trains on the last 3 batches of data, and it replaces the best net only if it beats it and doesn't get worse against a random player.

The net looks one action ahead: it tries every legal action, scores the resulting position, and plays the best. That makes it a basic player, not a strong one. Settings are at the top of `tzaar/training.py`.

## Layout

```
train.py, serve.py     entry points
tzaar/
  board.py             the 60 points and the lines between them
  game.py              rules: State, legal moves, win conditions, new_game
  encoding.py          positions -> numbers for the network
  network.py           the neural network
  players.py           random player and net player
  games.py             play games, make training data, run matches
  training.py          the generation loop
  model_store.py       save / load the model
  server.py            the web server
web/index.html         the game board page
models/                saved models (git-ignored)
```
