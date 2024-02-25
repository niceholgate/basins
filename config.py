import sympy as sp
from datetime import datetime

start_time = datetime.now()
symbols = sp.symbols('x y d')
x, y, d = symbols
f_sym = [y**2+2*x*y+2*x**2-4-0.5*sp.sin(15*(x+d)), y-10*x**2+3+d]

X_PIXELS = int(3440/8)
Y_PIXELS = int(1440/8)
COLOUR_SET = 1
ANIMATE = True

DELTA = 2
FRAMES = 5
FPS = 10

DISABLE_JIT = False
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