"""Board geometry: a hexagon with 5 points per side, minus the centre = 60 points.

Points are numbered 0..59. Pieces move along the 6 straight lines through a point.
The centre of the board is a gap: nothing can land on it or jump over it.
"""
import numpy as np

RADIUS = 4
DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

# (q, r) hex coordinates of every point; the centre (0, 0) is left out.
POINTS = [(q, r)
          for q in range(-RADIUS, RADIUS + 1)
          for r in range(-RADIUS, RADIUS + 1)
          if abs(q + r) <= RADIUS and (q, r) != (0, 0)]
NUM_POINTS = len(POINTS)
POINT_INDEX = {point: i for i, point in enumerate(POINTS)}


def _build_rays():
    """rays[i] = for each of the 6 directions, the points reachable in that direction, nearest first."""
    rays = []
    for q, r in POINTS:
        point_rays = []
        for dq, dr in DIRECTIONS:
            ray, (nq, nr) = [], (q + dq, r + dr)
            while (nq, nr) in POINT_INDEX:      # stops at the board edge or the centre gap
                ray.append(POINT_INDEX[(nq, nr)])
                nq, nr = nq + dq, nr + dr
            point_rays.append(ray)
        rays.append(point_rays)
    return rays


RAYS = _build_rays()

# The same rays as one padded array indexed [point, direction, step], so many positions can be checked at once.
# Unused steps hold NUM_POINTS: a made-up extra point that is always empty.
MAX_RAY = max(len(ray) for point_rays in RAYS for ray in point_rays)
RAY_TABLE = np.full((NUM_POINTS, len(DIRECTIONS), MAX_RAY), NUM_POINTS, dtype=np.int64)
for _point, _point_rays in enumerate(RAYS):
    for _direction, _ray in enumerate(_point_rays):
        RAY_TABLE[_point, _direction, :len(_ray)] = _ray
