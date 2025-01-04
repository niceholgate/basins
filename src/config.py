# Whether to use numba acceleration (compiles solver to C library ahead-of-time).
ENABLE_NUMBA = True

# Whether to use taichi acceleration.
# This is a WIP. It works, but it doesn't yet leverage quadtrees, and GPU isn't faster.
# If True, overrides numba and quadtrees acceleration.
ENABLE_TAICHI = False
ENABLE_GPU = True

# Whether to accelerate the frame generation using quadtree partitioning of the region.
# If enough randomly sampled pixels in a quadtree subregion are all the same colour, the remaining pixels are filled in
# without needing to perform Newton's method for each one.
ENABLE_QUADTREES = True

# Whether to save a .png file of the image (for animations, only the first frame is saved).
SAVE_PNG_FRAME = True

# Parameters relating to pre-frame search for unique solutions.
MAX_SEARCH_POINTS = 500
MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION = 30
SHOW_UNIQUE_SOLUTIONS_AND_EXIT = False
MIN_FRAMES_BETWEEN_NEW_SOLUTIONS = 5

# Newton's method parameters.
MAX_ITERS = 1000
EPSILON = 0.0001

# Image post-proc parameters.
BLACKOUT_ITERS = 1000
BLENDING_DECAY_FACTOR = 4.0
DEFAULT_COLOURS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
           'tab:pink', 'tab:olive', 'tab:cyan', 'tab:gray', 'tab:brown']
FRAME_COUNT_PADDING = '{:06d}'


import os
from pathlib import Path
BUILD_DIR = Path(os.path.realpath(__file__)).parent.parent/'build'
