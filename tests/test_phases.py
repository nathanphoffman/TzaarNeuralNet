"""Tests for the optional plain / smart phases."""
import unittest
from types import SimpleNamespace

from tzaar.learning import settings
from tzaar.learning.phases import Phases

def verdict(accepted=False, score=0.45):
    return SimpleNamespace(accepted=accepted, score_vs_best=score, rating=100.0)

class PhaseTests(unittest.TestCase):
    def setUp(self):
        for name in ("SMART_PHASE_GENERATIONS", "SMART_PHASE_SETTINGS", "SELF_PLAY_SEARCH_TURNS", "CURRENT_PHASE"):
            self.addCleanup(setattr, settings, name, getattr(settings, name))

        settings.SMART_PHASE_GENERATIONS = 3
        settings.SMART_PHASE_SETTINGS = {"SELF_PLAY_SEARCH_TURNS": 2}

    def test_off_by_default_changes_nothing(self):
        settings.SMART_PHASE_GENERATIONS = 0
        phases, near_misses = Phases(), []
        phases.start_generation()
        phases.end_generation(1, "net", verdict(accepted=True), near_misses)
        self.assertFalse(phases.smart)
        self.assertEqual(settings.SELF_PLAY_SEARCH_TURNS, 0)
        self.assertEqual(near_misses, [])

    def test_an_accepted_net_switches_phase_and_the_smart_settings_apply_only_in_the_smart_phase(self):
        phases, near_misses = Phases(), []
        phases.start_generation()
        self.assertEqual((settings.SELF_PLAY_SEARCH_TURNS, settings.CURRENT_PHASE), (0, "plain"))
        phases.end_generation(1, "a", verdict(accepted=True), near_misses)
        phases.start_generation()
        self.assertEqual((settings.SELF_PLAY_SEARCH_TURNS, settings.CURRENT_PHASE), (2, "smart"))
        phases.end_generation(2, "b", verdict(accepted=True), near_misses)
        phases.start_generation()
        self.assertEqual((settings.SELF_PLAY_SEARCH_TURNS, settings.CURRENT_PHASE), (0, "plain"))

    def test_plain_rejects_never_switch(self):
        phases, near_misses = Phases(), []

        for generation in range(1, 10):
            phases.end_generation(generation, "net", verdict(), near_misses)

        self.assertFalse(phases.smart)

    def test_a_smart_phase_gives_up_after_the_cap_and_keeps_its_best_reject_as_an_opponent(self):
        phases, near_misses = Phases(), []
        phases.end_generation(1, "a", verdict(accepted=True), near_misses)         # now smart
        phases.end_generation(2, "low", verdict(score=0.40), near_misses)
        phases.end_generation(3, "best", verdict(score=0.48), near_misses)
        self.assertTrue(phases.smart)
        phases.end_generation(4, "mid", verdict(score=0.44), near_misses)           # third smart generation: cap
        self.assertFalse(phases.smart)
        self.assertEqual([(generation, net) for generation, net, _ in near_misses], [(3, "best")])

    def test_a_best_reject_that_is_already_an_opponent_is_not_added_twice(self):
        phases, near_misses = Phases(), [(2, "net", 100.0)]
        phases.end_generation(1, "a", verdict(accepted=True), near_misses)
        phases.end_generation(2, "net", verdict(score=0.49), near_misses)
        phases.end_generation(3, "x", verdict(score=0.40), near_misses)
        phases.end_generation(4, "y", verdict(score=0.40), near_misses)
        self.assertEqual(len(near_misses), 1)

    def test_an_unknown_override_is_an_error(self):
        settings.SMART_PHASE_SETTINGS = {"NOT_A_SETTING": 1}

        with self.assertRaises(ValueError):
            Phases()

if __name__ == "__main__":
    unittest.main()
