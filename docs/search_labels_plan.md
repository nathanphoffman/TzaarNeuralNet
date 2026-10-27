# Plan: train the net on the search's scores

Status: **built** (settings `SEARCH_LABEL_*`, off by default). Config for the experiment: `models/evolution-7/config.json`.

## Why

Lookahead makes a net much stronger when it plays (+160 to +250 Elo), but putting it into self-play did not make the
trained nets stronger:

| Run | What it did | Result |
|---|---|---|
| evolution-1 | no search | best plain net, 881 on the benchmark; the one everything else is compared to |
| evolution-3 | best net looks ahead, opponents play plainly | rejected 9 generations in a row |
| evolution-5 (from evo-4) | same, plus 20% of games vs the baselines | climbed fast, stalled at 724 |
| evolution-6 | both sides look ahead | running, no result yet |

In all of these a position is still labeled with who won the whole game (+1 or -1), a noisy label. Search only
changes which games get played. The benchmark rates the plain net without search, so the search's strength never
reaches the thing we measure.

## The idea

Label a position with **what the search says about it** instead of (or mixed with) the game result. The net learns to
give the answer the search would give, without searching.

- The games are played with plain nets (fast), as in evolution-1.
- For a sample of the recorded positions, run the lookahead from that position and use its score as the target.
- Targets are clipped to the net's output range (-1 to 1). The search returns +2/-2 for a certain win/loss.

## Design

1. **New function** in `tzaar/nets/search.py`: `search_value(network, state, turns, keep, side)`, the same
   minimax as `_value` (already there), returning the score for `side`. No new search logic.
2. **Keep the state with each position.** `play_game` today records only the encoded board. It also has to keep the
   `State` (or a copy) for the sampled positions, because the search needs the real board, not the encoding.
   `side` is the actor, as for the label today.
3. **Sample, don't label everything.** One search is about 0.05 s, and a 1000-game batch has roughly 100,000
   positions. Label a fraction (a setting; evolution-7 uses 30%). The other positions keep the win/loss label.
4. **Mix the target:** `target = (1 - w) * game_result + w * search_value`. `w` is a setting; start at 0.5, try 1.0.
5. **Settings** in `settings.py`, all off by default so nothing else changes: `SEARCH_LABEL_SHARE = 0`,
   `SEARCH_LABEL_WEIGHT = 0.5`, `SEARCH_LABEL_TURNS = 2`, `SEARCH_LABEL_KEEP = 3` for the depth of the labeling search (separate from the self-play search).
6. **Labeling runs in the worker pool** (`games._in_parallel`), at the end of each worker's games, and counts
   toward the progress bar like games do.
7. **Saved batches** already store (inputs, targets), so a restart reuses labeled batches. Tag them so a batch
   labeled with other settings is not reused.

## Testing it

- **Unit tests:** `search_value` at one turn matches the best of `score_moves`; a won position gives +1 after
  clipping; the mix gives the expected numbers; with the share at 0 the data is identical to today.
- **Quick check before a long run:** label 500 positions both ways and see how often the sign of the search's score
  agrees with the game result. If they almost always agree, the labels add little.
- **Experiment:** `evolution-7`, a copy of evo-1 at generation 432 (no saved batches), plain self-play, labels as
  above. Run it for 50 generations and compare against evo-1's own run from the same point.
- **Judging it:** the benchmark ladder, plain net, same as for every other model. Internal ratings are only for
  watching progress.

## Risks

- **Slow labeling.** 10% of 100,000 positions is 10,000 searches, about 8 minutes on one core. OK at 12 workers; a
  larger share needs a cheaper search (1 turn).
- **The search is only as good as the net it uses.** It improves a bad net's play by a lot, but its scores are still
  built from that net's own guesses. This works because the search sees more than the net does (as in AlphaZero),
  not because it is perfect.
- **Positions near the end of a game** have an exact answer; the search finds it easily, so those labels should be
  good. Early positions are where it is least reliable, and that is where win/loss labels are weakest too.
- **It might still stall like the others.** If so, the next ideas are a deeper or wider search for the labels, and
  weighted-random move choice in self-play (see `minimax_plan.md`).

## Quick check (done)

2,252 positions from evo-1 self-play: the search's score matched the game result's sign 56.4% of the time (correlation 0.23); the plain net's own guess 52.8% (0.10). In the second half of games 65% against 59%. So the search values carry more information than the plain net's guess. Batches are not tagged with the label settings; they are per model, so don't change them mid-run.
