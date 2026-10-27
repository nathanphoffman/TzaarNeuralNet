"""Optional alternating phases: plain generations (the model's normal settings) and smart ones (the same settings with
SMART_PHASE_SETTINGS applied, e.g. lookahead in self-play). Off unless SMART_PHASE_GENERATIONS is set.

The model starts in the plain phase. An accepted net ends the phase it was found in, and the other phase takes over from
the new best net. A smart phase that finds nothing in SMART_PHASE_GENERATIONS generations hands back anyway, and its best
rejected net becomes an extra self-play opponent (a near-miss) for the plain phase after it."""

from tzaar.learning import settings

class Phases:
    def __init__(self):
        self.smart = False
        self.generations_in_smart = 0
        self.best_reject = None         # (generation, net, rating, score vs best) of the best rejected net in this smart phase
        self._normal = {}               # a setting's normal value, kept while the smart phase has changed it

        for name in settings.SMART_PHASE_SETTINGS:
            if not hasattr(settings, name):
                raise ValueError(f"unknown setting in SMART_PHASE_SETTINGS: {name}")

    @property
    def enabled(self):
        return settings.SMART_PHASE_GENERATIONS > 0

    def start_generation(self):
        """Put the settings of the current phase in place."""

        if not self.enabled:
            return

        for name, value in settings.SMART_PHASE_SETTINGS.items():
            self._normal.setdefault(name, getattr(settings, name))
            setattr(settings, name, value if self.smart else self._normal[name])

        settings.CURRENT_PHASE = "smart" if self.smart else "plain"

    def end_generation(self, generation, challenger, verdict, near_misses):
        """Switch phase if this generation's result says so. Adds to `near_misses` when a smart phase hands back empty-handed."""

        if not self.enabled:
            return

        if verdict.accepted:
            self._switch(f"gen {generation} was accepted")
        elif self.smart:
            self.generations_in_smart += 1

            if self.best_reject is None or verdict.score_vs_best > self.best_reject[3]:
                self.best_reject = (generation, challenger, verdict.rating, verdict.score_vs_best)

            if self.generations_in_smart >= settings.SMART_PHASE_GENERATIONS:
                kept, network, rating, _ = self.best_reject

                if all(known != kept for known, _, _ in near_misses):
                    near_misses.append((kept, network, rating))

                self._switch(f"{self.generations_in_smart} generations found nothing; gen {kept} is kept as an opponent")

    def _switch(self, reason):
        self.smart = not self.smart
        self.generations_in_smart = 0
        self.best_reject = None
        print(f"   phase: now {'smart' if self.smart else 'plain'} ({reason})")
