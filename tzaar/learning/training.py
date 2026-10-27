"""The training pipeline: random games first, then generations of self-play.

Training can be stopped and resumed: after every generation the nets and the history are
saved (see checkpoint.py), and the next run carries on from there.
"""

import itertools

from tzaar.learning import settings
from tzaar.learning.checkpoint import (GENERATION_KEY, RATING_BASIS, clear_checkpoint, load_batches, load_checkpoint, save_batch,
                                       save_checkpoint)
from tzaar.learning.generation import (add_row, file_challenger, play_self_play_games, report_verdict, retest_borderline,
                                       score_vs_partner, test_challenger, train_challenger)

from tzaar.nets.games import generate_training_data, win_rate
from tzaar.learning.phases import Phases
from tzaar.learning.policy import RollingDonor, games_for, self_play_opponents
from tzaar.nets.network import Network
from tzaar.nets.players import make_net_player, random_player
from tzaar.nets.rating import elo_gap, rate_net
from tzaar.runs.progress import progress

def start_new_run():
    """Generation 0: learn from random-vs-random games, then rate that net.
    Returns (history, rated_nets, near_misses, batches) ready for the self-play generations."""

    print(f"STEP 1: {settings.RANDOM_GAMES:,} random-vs-random games")
    progress.step("Playing random-vs-random games")
    inputs, targets = generate_training_data(random_player, [random_player], settings.RANDOM_GAMES)
    print(f"   {len(inputs)} positions recorded")

    print("STEP 2: train generation 0 on random data")
    progress.step("Training generation 0")
    network = Network()
    network.train(inputs, targets)

    # Every later rating is measured from this one (random = 0), so a small error here would shift all of them;
    # that is why this match is the biggest.
    vs_random = win_rate(make_net_player(network), random_player, settings.MAX_MATCH_GAMES)
    rating = elo_gap(vs_random)
    print(f"   gen 0: rating {rating:.0f} | vs random {vs_random:.2f}")
    history, rated_nets = [], [(0, network, rating)]
    add_row(history, 0, rating, vs_random, settings.MAX_MATCH_GAMES)

    return history, rated_nets, [], [(inputs, targets)]

def rebuild_batches(rated_nets, near_misses, next_generation):
    """The batches of games to start from (one fewer than we keep; the first generation adds the last one): the saved
    batches the current best net played, and fresh self-play games for whatever is missing."""

    best_generation = rated_nets[-1][0]
    saved = load_batches(best_generation, settings.BATCHES_TO_KEEP - 1)
    missing = settings.BATCHES_TO_KEEP - 1 - len(saved)

    if not missing:
        progress.step("Loading the saved training data")
        print(f"Reusing {len(saved)} saved batch(es) of games played by gen {best_generation}")

        return saved

    progress.step("Rebuilding training data from the best net")
    explorer, opponents, weights, _ = self_play_opponents(rated_nets, near_misses)
    games = games_for(next_generation, settings.MAX_SELF_PLAY_GAMES)
    fresh = []

    for _ in range(missing):
        fresh.append(generate_training_data(explorer, opponents, games, weights))
        save_batch(fresh[-1], best_generation, settings.BATCHES_TO_KEEP)

    return saved + fresh

def rebase_best_rating(rated_nets, games):
    """Checkpoints from before the new rating rule rated nets with the random player averaged in,
    which inflated the numbers. Re-rate the best net from head-to-head games only, so it can be
    compared fairly with the ratings new nets get."""

    generation, best, old_rating = rated_nets[-1]
    player = make_net_player(best)
    results = [(rating, win_rate(player, make_net_player(network), games))
               for _, network, rating in rated_nets[:-1][-settings.OPPONENTS_BACK:]]

    new_rating = rate_net(results)
    print(f"Re-rated the best net (gen {generation}) from head-to-head games only: "
          f"{old_rating:.0f} -> {new_rating:.0f}")

    rated_nets[-1] = (generation, best, new_rating)

def run_self_play_generations(generations, history, rated_nets, near_misses, batches, donors=()):
    """Each generation: the best net plays older nets (and itself), a new net trains on the data,
    and it replaces the best net only if it passes the acceptance checks.
    Returns (best net, its rating)."""

    print("STEP 3: self-play generations")
    print("   (rating = Elo, random player = 0, every +100 is about a 64% win rate; "
          "later generations play more games per test; * = near-miss net used as an opponent)")

    first = history[-1][GENERATION_KEY] + 1
    numbers = itertools.count(first) if generations is None else range(first, first + generations)

    rolling = RollingDonor()
    phases = Phases()
    start_row = next((row for row in history if row[GENERATION_KEY] == rated_nets[-1][0]), None)

    if start_row is not None and start_row.get("vs_partner") is None:         # how far the net we start from is from the partner
        start_row["vs_partner"] = score_vs_partner(rated_nets[-1][1])

    for generation in numbers:
        phases.start_generation()
        match_games = games_for(generation, settings.MAX_MATCH_GAMES)
        batches = play_self_play_games(generation, rated_nets, near_misses, batches)
        challenger = train_challenger(generation, rated_nets, rolling.donors_with(donors), batches)
        verdict = test_challenger(generation, challenger, rated_nets, match_games)

        if verdict.borderline:
            retest_borderline(generation, challenger, rated_nets, verdict)

        report_verdict(generation, verdict)
        vs_partner = score_vs_partner(challenger) if verdict.accepted else None
        add_row(history, generation, verdict.rating, verdict.score_vs_random, match_games, vs_best=verdict.score_vs_best,
                accepted=verdict.accepted, near_miss=verdict.near_miss, retest=verdict.retest, vs_partner=vs_partner)

        donors = file_challenger(generation, challenger, verdict, rated_nets, near_misses, donors)

        if verdict.accepted:
            rolling.accept()
        else:
            rolling.reject(generation, challenger, verdict.score_vs_best)

        phases.end_generation(generation, challenger, verdict, near_misses)
        save_checkpoint(history, rated_nets, near_misses, donors)

    _, best, best_rating = rated_nets[-1]

    return best, best_rating

def upgrade_best_net(rated_nets):
    """Widen the best net to the current hidden size and add the second layer if it has none. Nothing else
    changes: such a net plays exactly like the original, and older narrower nets stay as they are and still
    serve as opponents."""

    generation, best, rating = rated_nets[-1]

    wider = best.W1.shape[1] < settings.HIDDEN_NEURONS
    second_wider = best.V is not None and best.V.shape[1] < settings.HIDDEN2_NEURONS

    if wider or second_wider:
        rated_nets[-1] = (generation, best.widened(), rating)

        if wider:
            print(f"Widened the best net (gen {generation}) from {best.W1.shape[1]} to {settings.HIDDEN_NEURONS} hidden neurons")

        if second_wider:
            print(f"Widened the best net's second layer (gen {generation}) from {best.V.shape[1]} to {settings.HIDDEN2_NEURONS} neurons")

    if rated_nets[-1][1].V is None:             # a net saved before the second layer existed: add it (scores don't change)
        generation, best, rating = rated_nets[-1]
        rated_nets[-1] = (generation, best.deepened(), rating)
        print(f"Added a second hidden layer of {settings.HIDDEN2_NEURONS} neurons to the best net (gen {generation})")

def start_fresh(generations):
    """No checkpoint: generation 0 from random games. Returns what the self-play generations need."""

    open_ended = generations is None
    progress.start(None if open_ended else generations,
                   None if open_ended else 2 + settings.STEPS_PER_GENERATION * generations,
                   settings.MAX_MATCH_GAMES, setup_steps=2, steps_per_generation=settings.STEPS_PER_GENERATION)

    history, rated_nets, near_misses, batches = start_new_run()
    save_checkpoint(history, rated_nets, near_misses)

    return history, rated_nets, near_misses, batches, []

def resume(checkpoint, generations):
    """Carry on from a saved checkpoint. Returns what the self-play generations need."""

    history, rated_nets, near_misses, basis, donors = checkpoint
    open_ended = generations is None
    upgrade_best_net(rated_nets)
    donors = [(donor_generation, network.widened(), score) for donor_generation, network, score in donors]
    last = history[-1][GENERATION_KEY]
    print(f"Resuming after generation {last}. Best net so far: generation {rated_nets[-1][0]}, "
          f"rating {rated_nets[-1][2]:.0f}; {len(near_misses)} near-miss nets saved")

    rebase = basis < RATING_BASIS and len(rated_nets) > 1
    setup_steps = 1 + rebase
    progress.start(None if open_ended else last + generations,
                   None if open_ended else setup_steps + settings.STEPS_PER_GENERATION * generations,
                   settings.MAX_MATCH_GAMES, rows=history, setup_steps=setup_steps,
                   steps_per_generation=settings.STEPS_PER_GENERATION)

    if rebase:
        progress.step("Re-rating the best net (head-to-head games only)")
        rebase_best_rating(rated_nets, games_for(last + 1, settings.MAX_MATCH_GAMES))

    batches = rebuild_batches(rated_nets, near_misses, last + 1)
    save_checkpoint(history, rated_nets, near_misses, donors)       # records the new rating basis

    return history, rated_nets, near_misses, batches, donors

def train_model(generations=None, fresh=False):
    """Run `generations` more generations, or keep going until stopped if it is None.
    Continues from a saved checkpoint if there is one, unless `fresh`. Returns the best net."""

    if fresh:
        clear_checkpoint()

    checkpoint = load_checkpoint()
    history, rated_nets, near_misses, batches, donors = (start_fresh(generations) if checkpoint is None
                                                         else resume(checkpoint, generations))

    best, rating = run_self_play_generations(generations, history, rated_nets, near_misses, batches, donors)
    vs_random = next(entry["vs_random"] for entry in history if entry[GENERATION_KEY] == rated_nets[-1][0])
    print(f"FINAL: best net rating {rating:.0f} (random player = 0), wins {vs_random:.0%} vs random")
    progress.finish(f"Best net rating: {rating:.0f} Elo. It beat a random player {vs_random:.0%} of the time.")

    return best
