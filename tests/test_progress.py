"""Tests for the progress bar."""
import unittest
from unittest import mock

from tzaar.runs import progress as progress_module
from tzaar.runs.progress import Progress, fraction_now

def snapshot(**changes):
    values = {"games_done": None, "games_total": None, "running": True, "steps_started": 1, "total_steps": None, "setup_steps": 0, "steps_per_generation": 3,
              "step_started_at": 100.0, "step_durations": [None, None, None]}
    values.update(changes)

    return values

class FractionTests(unittest.TestCase):
    def test_a_finished_run_is_full(self):
        self.assertEqual(fraction_now(snapshot(running=False), 0), 1.0)

    def test_untimed_steps_count_the_same(self):
        self.assertAlmostEqual(fraction_now(snapshot(steps_started=2), 105), 1 / 3)

    def test_timed_steps_are_weighted_by_how_long_they_took(self):
        durations = [10.0, 20.0, 70.0]
        # on the second step (index 1), 10 seconds in: (10 + 10) / 100
        self.assertAlmostEqual(fraction_now(snapshot(steps_started=2, step_durations=durations), 110), 0.2)

    def test_a_step_that_overruns_waits_below_its_share(self):
        durations = [10.0, 20.0, 70.0]
        value = fraction_now(snapshot(steps_started=2, step_durations=durations), 100 + 1000)
        self.assertAlmostEqual(value, (10 + 0.95 * 20) / 100)

    def test_a_fixed_run_counts_finished_generations(self):
        durations = [10.0, 20.0, 70.0]
        # 2 setup steps, 10 generations; 2 setup + 3 steps done = on the 6th step: generation 1, first step, no time in yet
        value = fraction_now(snapshot(steps_started=6, setup_steps=2, total_steps=32, step_durations=durations), 100)
        self.assertAlmostEqual(value, 1 / 10)

    def test_setup_steps_use_the_plain_step_count(self):
        self.assertAlmostEqual(fraction_now(snapshot(steps_started=2, setup_steps=2, total_steps=32), 100), 1 / 32)

class GamesFractionTests(unittest.TestCase):
    def test_setup_follows_the_games_played(self):
        value = fraction_now(snapshot(setup_steps=1, steps_started=1, games_done=500, games_total=1000), 100)
        self.assertAlmostEqual(value, 0.5)

    def test_the_first_generations_self_play_follows_the_games_played(self):
        value = fraction_now(snapshot(setup_steps=1, steps_started=2, games_done=500, games_total=1000), 100)
        self.assertAlmostEqual(value, 0.5 / 3)

    def test_games_in_later_steps_do_not_move_an_untimed_bar(self):
        value = fraction_now(snapshot(setup_steps=0, steps_started=3, games_done=500, games_total=1000), 100)
        self.assertAlmostEqual(value, 2 / 3)

    def test_a_snapshot_without_game_counts_still_works(self):
        self.assertAlmostEqual(fraction_now(snapshot(steps_started=2), 100), 1 / 3)

class ProgressTimingTests(unittest.TestCase):
    def test_step_lengths_are_learned_from_the_previous_generation(self):
        clock = [1000.0]

        with mock.patch.object(progress_module.time, "time", lambda: clock[0]):
            progress = Progress()
            progress.start(None, None, 100, setup_steps=1)
            progress.step("setup")
            clock[0] += 3                         # the setup step is not part of a generation, so it isn't timed

            for length, name in ((5.0, "self-play"), (10.0, "training"), (15.0, "testing")):
                progress.step(name)
                clock[0] += length

            self.assertEqual(progress.snapshot()["step_durations"], [5.0, 10.0, None])      # testing is still going
            progress.step("next generation")
            self.assertEqual(progress.snapshot()["step_durations"], [5.0, 10.0, 15.0])

    def test_the_detail_line_stays_until_it_is_replaced(self):
        progress = Progress()
        progress.start(None, None, 100)
        progress.set_detail("opponents: gen 5 80%")
        progress.step("training")
        self.assertEqual(progress.snapshot()["detail"], "opponents: gen 5 80%")
        progress.start(None, None, 100)
        self.assertIsNone(progress.snapshot()["detail"])

    def test_game_counts_are_cleared_when_the_next_step_starts(self):
        progress = Progress()
        progress.start(None, None, 100)
        progress.step("games")
        progress.set_games(40, 100)
        self.assertEqual((progress.snapshot()["games_done"], progress.snapshot()["games_total"]), (40, 100))
        progress.step("training")
        self.assertIsNone(progress.snapshot()["games_total"])

if __name__ == "__main__":
    unittest.main()
