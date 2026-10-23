# Plan: better acceptance rule for new nets

Status: **built** (acceptance rule, head-to-head-only rating, one-time re-rating of the best net, and near-miss nets as self-play opponents). The "Later ideas" section at the bottom is still to do.

## The problem

A new net is only accepted if it scores at least 55% against the current best **and** its win rate vs random is within 1 point of the best's. Two issues showed up around gens 16 to 25:

- **The vs random check is noisy at the top of the scale.** At about 95%, 1 point is roughly 30 Elo, but with 400 games the result is only accurate to about 1.2 points (roughly 37 Elo). So it rejects good nets by chance (gen 16: won 60% vs best, rejected on vs random).
- **The rating is inflated by the random leg.** A 96% win rate vs random works out to about +575 Elo on its own, and it is averaged in with the head-to-head results. Nets that were about even with the best (50 to 52%) still got higher ratings (gens 22, 24, 25), because the random leg pulled the average up.

A flat percentage margin is fine for head-to-head matches (55% is about +35 Elo at any level). It is not fine for vs random.

## The rule

A new net is accepted only if **both** hold:

1. **Beats the best net head to head:** at least 52% (about +14 Elo).
2. **Higher rating than the best net:** the rating is the average of its results against the last 3 accepted nets, **with the random leg removed**. It must beat the best net's rating by a small margin (about 5 Elo).

Optional safety floor instead of the old vs random comparison: still above about 90% vs random.

Margins are slim on purpose. A net that is only equal to the best does little harm, while rejecting a real improvement loses progress.

## Re-test of borderline nets (also built)

- **Trigger:** the net is rejected, still beats random at least 90%, and either (a) its rating clears the bar and it won at least 48% vs the best net, or (b) it won at least 52% vs the best net but its rating is short by up to 20 Elo.
- **Re-test:** 1,600 games against the best net. Only this result decides; the first 400 games are not pooled in (that would let lucky nets in).
- **Outcome:** 52% or more accepts, 50% to 52% rejects but keeps it as a near miss, under 50% rejects.
- **Rating after an accept:** a net accepted after a re-test is rated as the old best's rating plus the Elo gap from the re-test win rate (not the noisy three-opponent average), so the best net's rating can't drift down.
- **Cost:** about a minute, and only for the few generations that qualify.
- Settings: `RETEST_GAMES`, `RETEST_MIN_SCORE`, `RETEST_RATING_SLACK` in `settings.py`.

## Things to handle

- **Re-rate the best net once on resume.** Its stored rating includes the random leg, so comparing it with new ratings would make check 2 too hard. On the first resume after the change, play the current best against the same last-3 opponents and store the new rating.
- **The two checks overlap a little,** because the best net is one of the 3 opponents in the rating. They are not two fully separate votes.
- **Old history rows** keep their old ratings (with the random leg). The dashboard chart will have a small step where the rating basis changes. Either note it in the README or mark the change point on the chart.
- **Settings** go in `settings.py`: `MIN_SCORE_VS_BEST = 0.52`, `MIN_RATING_GAIN = 5`, optional `MIN_SCORE_VS_RANDOM = 0.90`.

## Steps

1. Change `rate_net` usage in `training.py` so the rating uses only the head-to-head results (drop the `(0, score_vs_random)` entry).
2. Replace the accept condition with the two checks above.
3. Add the one-time re-rating of the best net when resuming.
4. Keep logging vs random in the history and dashboard (it is still useful to see), but not in the rating.
5. Check on a copy of the project, not the live `models/` folder, then compare a few generations against the old rule.

## Later ideas (bigger payoffs, not part of this change)

- **Lookahead search:** for each action, try the opponent's replies and assume the worst. Likely the biggest improvement.
- **Extra inputs:** pieces left per type per side, captures available, tallest stack.
- **Bigger net** (more hidden units).
- **Better labels:** weight positions near the end of a game more, or train on the search result.
- **Head-to-head script:** compare any two saved nets over about 1,000 games, as a check that the rating is really climbing.
- **Game server:** reload the best net on "New game", and let you pick an older net to play.
