# Changes that are not proven yet

These changes were made on 2026-10-23 and have not been shown to help. If the scores against the best net stay
well below 0.50, they are the first suspects. Keep this file up to date: when a change is shown to work, move it
to the README; when it is backed out, delete its section.

## What to compare against

- Before the second layer: best net generation 213, rating 772. Rejected challengers scored about 0.49 to 0.51
  against the best net.
- After the second layer: generation 229 was accepted at 781 (0.53 vs the old best), then challengers dropped to
  0.36 to 0.47 against the best net.
- Rejected nets are normally accurate to about +-2 points per test, so look at the trend over 10 or more
  generations, not one result.

## 1. Second hidden layer (`network.py`, `training.py`, `model_store.py`)

- The net was inputs (797) -> 256 hidden -> 1 output. It now also has a second hidden layer of 64 neurons beside
  the direct path:
  `hidden2 = tanh(hidden @ V + c)` and `output = tanh(hidden @ W2 + hidden2 @ U + b2)`.
- `U` starts at zero, so a net that is given the layer scores every position exactly as it did before.
- `Net.deepened()` adds the layer. `training.py` calls it on resume if the best net has none.
- `HIDDEN2 = 64` sets the size.
- `SECOND_LAYER_RATE = 0.1`: the layer learns at a tenth of the normal learning rate, so it cannot unsettle a net
  that is already tuned. The first attempt used the full rate and the challengers dropped to about 0.38.
- `model_store.py` saves and loads V, c and U. Older nets without them load fine and still work as opponents.
- Donors (`apply_donors` in `training.py`, `PROVEN_HIDDEN = 64`): new challengers get the average of the donors'
  weights for the layer, for the hidden neurons beyond the first 64, and for the newest inputs. The first 64
  neurons always come from the best net.

## 2. More training data and fewer epochs (`training.py`)

- `BATCHES_TO_KEEP`: 5 -> 8 -> 2. Each new net trains on the games from the last 2 generations (it was 3, then 5,
  then 8). It is only 2 because each batch is now huge, see `MAX_SELF_PLAY_GAMES`.
- Epochs: was a fixed `WARM_START_EPOCHS` (10 -> 6 -> 2 -> 1), now `epochs_for(rating)` in `training.py`:
  `max(1, round(6 - rating / 100))` using the best net's rating, so 6 at rating 0, 3 at 300, and 1 from about 500
  up (the current run is at 1). Each new net makes fewer passes over that data, so it drifts less from the
  best net it started from. Each position is labeled with the result of the whole game, which is noisy, so many
  passes mostly fit the noise.
- `MAX_SELF_PLAY_GAMES`: 800 -> 1000 -> 5000. The best net plays up to 5 times as many self-play games per
  generation. A position is about 3.2 KB, so a 5000-game batch is roughly 670 MB, and the machine has 12 GB, which
  is why only 2 batches are kept. If it runs out of memory, lower one of the two. (The test matches stay at
  `MAX_MATCH_GAMES = 600`.)
- These were changed together, so if the scores improve we won't know which one did it.
- Raising the batches and the games makes each restart play more games to refill the batches. Fewer epochs
  makes the training step faster.

## Idea, not built yet: reduced starting positions

The idea: some self-play games start with fewer pieces (for example only 1 or 2 stacks of a kind per side), to
force the net to learn the signals that decide a game, such as the "only one left" danger flags and the capture
inputs. In normal games those situations are rare, so the net sees few examples of them.

- Use only a small share of the self-play games, about 10 to 20%, and keep the rest normal. The net's job is to
  play normal games, and too many odd starts would spend its capacity on positions that rarely happen.
- Every start must give each side at least one stack of every kind, or the game is over at once.
- Testing and ratings stay on normal starts only, so Elo keeps its meaning.
- A cheaper alternative: take real positions from the late part of normal games and use those as the starting
  points. They are realistic few-piece positions, and not artificial ones.
- Where it would go: `new_game()` in `game.py` (a way to build a smaller starting setup) and
  `generate_training_data()` in `games.py` (use it for a share of the games).
- Wait until the current run stops improving before trying it, and change nothing else at the same time.

## How to back out

- Models from before the second layer: the `bafa982` commit ("checkpointing models", gen 228) or `14ad200`
  ("new high hit at 772 ELO"). Restore with `git checkout <commit> -- models`, then `git clean -fd models`.
- Code: the layer and donor changes are in `0011bc0`. To turn the layer off without removing the code, make
  `Net.__init__` and `Net.deepened()` skip `add_second_layer`, and remove the "Added a second hidden layer" step in
  `train_model`.
- The batches and epochs are single numbers at the top of `training.py`.

## How to judge them

- Watch the "vs best" column on the dashboard for 10 to 20 generations. Back near 0.50 means they are fine.
  Still around 0.40 to 0.45 means the layer, or the extra data and fewer epochs, is hurting.
- To test the layer on its own, play a net with the layer against a copy with its `U` weights set to zero. Clearly
  above 50% means the layer is helping.
