# Tzaar Self-Play

A small neural network that learns [Tzaar](https://en.wikipedia.org/wiki/TZAAR), plus a web page to play against it. Same idea as `../tic-tac-toe`. Only needs `numpy` (uv installs it for you).

## Run

From this folder:

```
uv run train.py    # train a model and save it to models/model.npz
uv run serve.py    # open http://localhost:8001 and play
```

`serve.py` trains a model automatically if `models/model.npz` doesn't exist. Delete that file to retrain from scratch.

## Stopping and resuming

Progress is saved after every generation in `models/checkpoint/` (every accepted net plus the history). Stop with Ctrl+C any time and run `uv run train.py` again to carry on from the last finished generation. Each run does 5 more generations by default.

```
uv run train.py --generations 20   # run 20 more generations
uv run train.py --fresh            # throw away saved progress and start over
```

The training data isn't saved. On resume it is rebuilt from a few fresh self-play games by the best net, which takes about a minute.

## Training dashboard

`uv run train.py` also starts a live dashboard at http://localhost:8002 with a progress bar, a win-rate chart per generation, and a table of the numbers. It stays up after training finishes so you can look at the results; press Ctrl+C to quit. Use `uv run train.py --no-dashboard` to skip it.

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
3. For each generation: the best net plays self-play games against itself, older accepted nets and near-miss nets (see below), a new net starts as a copy of the best net and trains a bit more on the last 3 batches of games, and it replaces the best net only if it passes the acceptance checks.

Every new net gets an Elo-style rating (random player = 0, +100 is about a 64% win rate). It is the average of what its head-to-head results against the last 3 accepted nets imply. The random player is left out of that average on purpose (it is nearly maxed out and very noisy). A rating that stops climbing means the setup has hit its ceiling.

**A new net is accepted only if all of these hold** (settings at the top of `tzaar/training.py`):
1. It wins at least 52% of games against the best net.
2. Its rating is at least 5 Elo above the best net's.
3. It still beats a random player at least 90% of the time (a safety floor).

**Re-test:** a net that fails narrowly (its rating clears the bar but it won at least 48% against the best net, or it won at least 52% but its rating is up to 20 Elo short) plays one extra 1,600-game match against the best net before it is rejected, because a normal test is only accurate to about 2.5 points. The re-test alone decides: 52% or more accepts it, 50% to 52% keeps it as a near miss, anything lower rejects it. The first test's lucky or unlucky games are not pooled in. A net accepted after a re-test gets the rating "best net's rating + the gap that re-test win rate implies", since that match is more reliable than the three-opponent average.

**Near misses:** a rejected net that still won at least 50% against the best net is saved (`models/checkpoint/miss_gen*.npz`) and joins the pool of self-play opponents, because it is nearly as strong as the best net. Near misses are never used as rating references. In the training log they are marked with `*`.

Checkpoints from before this rule load fine: on the first resume the best net is re-rated once from head-to-head games only, so it can be compared with new ratings. Ratings in the history before that point include the random player, so the rating chart has a small step there.

**What the net sees (797 inputs):** for each of the 60 points, which piece is there and how tall the stack is (720 numbers), a flag for "the same player moves again", how many stacks of each type each side has left (6 numbers, as a share of the starting count), 6 "only one left" danger flags, and a 60-point threat map marking which of your stacks the opponent could capture next move, how many captures each side could make (2 numbers), and 2 "no capture available" flags (a side that cannot capture on its turn loses). Nets saved before the newest inputs existed still load: the missing weights are filled with zeros, so they play exactly as before and can learn to use the new inputs. The code sets numpy to one thread (in `tzaar/__init__.py`), because on a many-core machine its default threading makes these small matrix multiplications 10 to 25 times slower.

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
  progress.py          training progress, shared with the dashboard
  dashboard.py         the training dashboard server
  server.py            the game web server
web/index.html         the game board page
web/dashboard.html     the training dashboard page
models/                saved models (git-ignored)
```
