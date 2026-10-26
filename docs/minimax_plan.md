# Plan: lookahead search (minimax)

Status: **plan agreed, nothing built yet.**

## Where we are

`pick_move` in `players.py` looks one action ahead: it tries every legal move, scores the board after each with the net, and plays the best. The benchmark (`benchmark_results.json`, round robin of the baselines and both best nets) is the "before" number:

| Player | Rating (random = 0) |
|---|---|
| evolution-1 best (gen 432) | 886 |
| evolution-2 best (gen 253) | 807 |

## The idea

For each candidate move, also try the opponent's replies and assume they pick the one that is worst for us. Pick the move whose worst case is best. Deeper = more of this.

## Decisions made

- **Depth is counted in whole turns.** A turn is two actions by the same player, so depth 2 = my turn, then the opponent's turn.
- **Only look at the net's top few moves (K) at each action.** Measured on 100 random games: about 38 legal moves on the first action of a turn (max 89) and about 78 on the second (max 157). That is roughly 3,000 move pairs per turn, and about 9 million for two turns, which is far too many. Keeping the top K by the net's score per action gives K^4 boards for two turns (K=5 is 625, K=8 is 4,096).
- **Start at 2 turns.** Try 3 only if the speed allows.
- **K and depth are settings.** We measure strength and time per move for different values before picking defaults.
- **Playing is deterministic:** always the top K moves, always play the best result. No randomness in the benchmark, so a result means what it says.

## Things specific to Tzaar

- **A turn is two actions by the same player**, so the search can't flip max/min every step. At every node it looks at `state.to_move` and maximises for the searcher's side, minimises for the other.
- **Game over inside the search.** `State.play` already sets `result`; a won position is +2 and a lost one is -2, the same as `score_moves` does now.

## Design

1. **Leaf value comes from the existing net.** `score_moves(network, state, moves)` already scores a whole list of moves in one batch. That same call also picks the top K moves at each step, and at the last level it gives the leaf values. Values are from the mover's point of view, so flip the sign when the other side moved.
2. **New file** `tzaar/nets/search.py` with `search_move(network, state, turns, keep)` and `make_search_player(network, turns, keep)` next to `make_net_player`. It is a normal player (state -> move), so `win_rate`, the benchmark and the web server can use it unchanged.
3. **First version: plain minimax over the top K moves, no pruning.** Easy to check: with K large enough and one action of depth it must pick the same moves as `pick_move`.
4. **Then alpha-beta and batching** (score the leaves of many nodes in one `predict` call) only if the speed needs it.

## Measuring it

- Add `--search-turns N` and `--keep K` to `benchmark.py` so the best nets also enter the ladder with search on.
- Main test: the same net with search vs without it, 400 games. Start with 1 turn and a small K as a sanity check, then 2 turns at several K values, then 3.
- Record time per move for each setting. Don't build anything for training before we know the speed.
- Check how often the move the search plays was outside the net's top K at the first step (this tells us if the top-K cut is throwing away good moves).

## Later, only if search helps (not part of the first build)

- **Weighted-random move choice for training.** In self-play, pick among the search's best moves with probability based on their scores, sharpened so good moves are much more likely and weak ones rare (for example a move rated well below the best gets chosen well under 1% of the time). This replaces the plain random-move `epsilon` with more realistic variety. Off by default, training only.
- **Weighted-random candidate choice.** Pick which moves to search by score instead of always the top K, so a move the net underrates is sometimes looked at. Test after the basic version.
- **Training use of search:** the search player plays the self-play games with the same win/loss labels as now, or the net is trained to predict the search result.

## Risks

- The net was trained to rank moves one step ahead. Its scores may not be accurate enough to chain together, so search could help less than hoped.
- Speed: even with the top-K cut, each move costs hundreds to thousands of net evaluations.
- The search player picks the same move every time from the same position. Starting positions are random, so games still differ.
