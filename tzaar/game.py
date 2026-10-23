"""The rules of Tzaar.

Each side has 6 Tzaars, 9 Tzaaras and 15 Totts. A turn is two actions:
  1. a capture (required)
  2. a capture, a stack, or a pass (optional)
White's very first turn is only the one capture.
A piece moves in a straight line over empty points and must land on the first
piece it meets:
  - capture: land on an enemy stack no taller than yours; it is removed
  - stack:   land on your own stack; they merge (height adds up, top piece is yours)
You lose if you can't capture at the start of your turn, or you have no stacks
left of one of the three types.
"""

from .board import NUM_POINTS, RAYS
from .rng import random_generator

TZAAR, TZAARA, TOTT = 0, 1, 2
KIND_NAMES = ["Tzaar", "Tzaara", "Tott"]
PIECES_PER_KIND = [6, 9, 15]
PASS = (-1, -1)          # a move is (from_point, to_point)

class State:
    """owner: 1 = White, -1 = Black, 0 = empty. result: 0 = ongoing, else the winner."""

    def __init__(self, owner, kind, height, to_move=1, phase=0, is_first_turn=True):
        self.owner, self.kind, self.height = owner, kind, height
        self.to_move = to_move
        self.phase = phase                  # 0 = first action (must capture), 1 = second action
        self.is_first_turn = is_first_turn
        self.result = 0
        self._legal = None

    def copy(self):
        state = State(self.owner[:], self.kind[:], self.height[:],
                      self.to_move, self.phase, self.is_first_turn)

        state.result = self.result

        return state

    def to_dict(self):
        return {"owner": self.owner, "kind": self.kind, "height": self.height,
                "to_move": self.to_move, "phase": self.phase,
                "is_first_turn": self.is_first_turn, "result": self.result}

    @classmethod
    def from_dict(cls, data):
        state = cls(data["owner"], data["kind"], data["height"],
                    data["to_move"], data["phase"], data["is_first_turn"])

        state.result = data["result"]

        return state

    def legal_moves(self):
        if self.result != 0:
            return []

        if self._legal is None:
            self._legal = self._generate_moves()

        return self._legal

    def _generate_moves(self):
        mover, moves = self.to_move, []

        for source in range(NUM_POINTS):
            if self.owner[source] != mover:
                continue

            for ray_points in RAYS[source]:
                for destination in ray_points:
                    dst_owner = self.owner[destination]

                    if dst_owner == 0:
                        continue                    # keep sliding over empty points

                    if dst_owner == -mover:
                        if self.height[destination] <= self.height[source]:
                            moves.append((source, destination))        # capture
                    elif self.phase == 1:
                        moves.append((source, destination))            # stack (second action only)

                    break                                   # can't go past a piece

        if self.phase == 1:
            moves.append(PASS)

        return moves

    def play(self, move):
        """Apply a move (assumed legal) and work out whose turn it is next."""
        mover = self.to_move

        if move != PASS:
            source, destination = move

            if self.owner[destination] == mover:                       # stack
                self.height[destination] += self.height[source]
            else:                                           # capture
                self.owner[destination] = mover
                self.height[destination] = self.height[source]

            self.kind[destination] = self.kind[source]
            self.owner[source], self.height[source] = 0, 0

        self._legal = None

        if self._lacks_a_kind(-mover):
            self.result = mover
        elif self._lacks_a_kind(mover):
            self.result = -mover
        elif self.phase == 0 and not self.is_first_turn:
            self.phase = 1                                  # same player, second action
        else:
            self.to_move, self.phase, self.is_first_turn = -mover, 0, False

            if not self.legal_moves():                      # can't capture = loses
                self.result = mover

    def _lacks_a_kind(self, side):
        kinds_left = {self.kind[point] for point in range(NUM_POINTS) if self.owner[point] == side}

        return len(kinds_left) < 3

def new_game():
    """A fresh game with all 60 pieces placed at random (the rules allow any setup)."""

    # One (side, kind) entry for every piece: each side gets PIECES_PER_KIND[kind] pieces of each kind.
    pieces = [(side, kind)
              for side in (1, -1)
              for kind, count in enumerate(PIECES_PER_KIND)
              for _ in range(count)]

    order = random_generator.permutation(len(pieces))
    owner = [pieces[piece][0] for piece in order]
    kind = [pieces[piece][1] for piece in order]

    return State(owner, kind, [1] * NUM_POINTS)
