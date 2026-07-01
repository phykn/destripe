ALL_DIRECTIONS = (0, 1, 2, 3, 4)
PARALLEL_OFFSETS = {
    0: (1, 0),
    1: (2, 1),
    2: (1, 1),
    3: (2, -1),
    4: (1, -1),
}
CROSS_OFFSETS = {
    0: (0, 1),
    1: (1, -2),
    2: (1, -1),
    3: (1, 2),
    4: (1, 1),
}

MU1_MIN = 0.10
MU1_MAX = 0.50
MU2_MIN = 0.0017
MU2_MAX = 0.017
EPS = 1e-9
