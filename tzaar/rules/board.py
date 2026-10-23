"""Board geometry: a hexagon with 5 points per side, minus the centre = 60 points.

Points are numbered 0..59. Pieces move along the 6 straight lines through a point.
The centre of the board is a gap: nothing can land on it or jump over it.
"""

import numpy as np

RADIUS = 4
DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

# (q, r) hex coordinates of every point; the centre (0, 0) is left out.
# Every (column, row) pair in the hexagon of the given radius, minus the empty centre point.
POINTS = [(column, row_number)
          for column in range(-RADIUS, RADIUS + 1)
          for row_number in range(-RADIUS, RADIUS + 1)
          if abs(column + row_number) <= RADIUS and (column, row_number) != (0, 0)]

NUM_POINTS = len(POINTS)
POINT_INDEX = {point: index for index, point in enumerate(POINTS)}

def _build_rays():
    """rays[i] = for each of the 6 directions, the points reachable in that direction, nearest first."""
    rays = []

    for column, row_number in POINTS:
        point_rays = []

        for column_step, row_step in DIRECTIONS:
            ray_points, (next_column, next_row) = [], (column + column_step, row_number + row_step)

            while (next_column, next_row) in POINT_INDEX:      # stops at the board edge or the centre gap
                ray_points.append(POINT_INDEX[(next_column, next_row)])
                next_column, next_row = next_column + column_step, next_row + row_step

            point_rays.append(ray_points)

        rays.append(point_rays)

    return rays

RAYS = _build_rays()

# The same rays as one padded array indexed [point, direction, step], so many positions can be checked at once.
# Unused steps hold NUM_POINTS: a made-up extra point that is always empty.
# The longest line of points that any point has in any direction (loops over every point, then every line from it).
MAX_RAY = max(len(ray_points) for point_rays in RAYS for ray_points in point_rays)
RAY_TABLE = np.full((NUM_POINTS, len(DIRECTIONS), MAX_RAY), NUM_POINTS, dtype=np.int64)

for _point, _point_rays in enumerate(RAYS):
    for _direction, _ray in enumerate(_point_rays):
        RAY_TABLE[_point, _direction, :len(_ray)] = _ray
