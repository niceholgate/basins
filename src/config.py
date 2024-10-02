DISABLE_JIT = False
SAVE_PNG_FRAMES = True
SEARCH_X_LIMS = (-1000, 1000)
SEARCH_Y_LIMS = (-1000, 1000)
MAX_SEARCH_POINTS = 500
MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION = 50
SHOW_UNIQUE_SOLUTIONS_AND_EXIT = False

MAX_ITERS = 1000
EPSILON = 0.0001
BLACKOUT_ITERS = 1000
BLENDING_DECAY_FACTOR = 4.0
DEFAULT_COLOURS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
           'tab:pink', 'tab:olive', 'tab:cyan', 'tab:gray', 'tab:brown']
FRAME_COUNT_PADDING = '{:06d}'

NUMBA_TARGET = 'CPU' # setting cuda is not doing anything yet