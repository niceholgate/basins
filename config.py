DISABLE_JIT = False
SEARCH_X_LIMS = [-1000, 1000]
SEARCH_Y_LIMS = [-1000, 1000]
MAX_SEARCH_POINTS = 500
MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION = 50
SHOW_UNIQUE_SOLUTIONS_AND_EXIT = False

MAX_ITERS = 1000
EPSILON = 0.0001
X_PIXELS = int(3440/4)
Y_PIXELS = int(1440/4)
BLACKOUT_ITERS = 1000
BLENDING_DECAY_FACTOR = 4.0
COLOURS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

NUMBA_TARGET = 'CPU' # setting cuda is not doing anything