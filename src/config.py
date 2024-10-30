ENABLE_JIT = False
ENABLE_AOT = False

ENABLE_TAICHI = True

ENABLE_QUADTREES = False


SAVE_PNG_FRAMES = True
MAX_SEARCH_POINTS = 500
MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION = 30
SHOW_UNIQUE_SOLUTIONS_AND_EXIT = False
MIN_FRAMES_BETWEEN_NEW_SOLUTIONS = 5

MAX_ITERS = 1000
EPSILON = 0.0001
BLACKOUT_ITERS = 1000
BLENDING_DECAY_FACTOR = 4.0
DEFAULT_COLOURS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
           'tab:pink', 'tab:olive', 'tab:cyan', 'tab:gray', 'tab:brown']
FRAME_COUNT_PADDING = '{:06d}'

NUMBA_TARGET = 'CPU' # setting cuda is not doing anything yet

import os
from pathlib import Path
BUILD_DIR = Path(os.path.realpath(__file__)).parent/'build'