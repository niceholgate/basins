import numpy as np
import sympy as sp
import matplotlib
from typing import Dict, Tuple, List
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import sys
from numba import jit

SEARCH_X_LIMS = [-1000, 1000]
SEARCH_Y_LIMS = [-1000, 1000]
MAX_SEARCH_POINTS = 500
MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION = 50
MAX_ITERS = 1000
EPSILON = 0.0001
X_PIXELS = int(3440)
Y_PIXELS = int(1440)


def points_approx_equal(p1, p2, EPSILON):
    return np.linalg.norm([p1[x]-p2[x], p1[y]-p2[y]]) < 2*EPSILON

def get_F_sym(f, g):
    return [f, g]

def get_J_sym(expressions, symbols):
    return [[sp.diff(exp, sym) for sym in symbols] for exp in expressions]

# @jit(forceobj=True)
# def get_J_num(j_sym, substitutions):
#     all_free_symbols = set()
#     for row in j_sym:
#         for expr in row:
#             for sym in expr.free_symbols:
#                 all_free_symbols.add(sym)
#
#     all_substitutions = [(symbol, substitutions[symbol])
#                          for symbol in all_free_symbols
#                          if symbol in substitutions.keys()]
#     arr = sp.Array(j_sym).subs(all_substitutions)
#     if len(arr.free_symbols) != 0:
#         raise Exception(f'Did not provide substitutions for these free symbols: {arr.free_symbols}')
#     return np.array(arr).astype(np.float64)

# @jit(forceobj=True)
# def get_F_num(f_sym, substitutions):
#     arr = np.zeros([2, 1])
#     for i in [0, 1]:
#         all_substitutions = []
#         for symbol in f_sym[i].free_symbols:
#             if symbol in substitutions:
#                 all_substitutions.append((symbol, substitutions[symbol]))
#             else:
#                 raise Exception(f'Did not provide substitutions for symbol: {symbol}')
#         arr[i, 0] = f_sym[i].subs(all_substitutions)
#
#     return arr

# @jit
def newton_solve(f_lambda, j_lambda, starting_guess, ordered_syms, epsilon, max_iters) -> Tuple[Dict[sp.Symbol, float], List[float]]:
    current_guess = starting_guess
    delta_norm_history = []
    delta_norm = 1000
    n_iters = 0
    while delta_norm > epsilon and n_iters < max_iters:
        current_guess, delta_norm = newton_iteration(f_lambda, j_lambda, current_guess, ordered_syms)
        delta_norm_history.append(delta_norm)
        n_iters += 1

    return current_guess, delta_norm_history

# @jit
def newton_iteration(f_lambda, j_lambda, guess, ordered_syms) -> Tuple[Dict[sp.Symbol, float], float]:
    f_num = f_lambda(guess[ordered_syms[0]], guess[ordered_syms[1]])
    j_num = j_lambda(guess[ordered_syms[0]], guess[ordered_syms[1]])
    delta = np.linalg.solve(np.array(j_num), -np.array(f_num)).flatten()
    new_guess = {sym: guess[sym] + delta[i] for i, sym in enumerate(ordered_syms)}
    delta_norm = np.linalg.norm(delta)
    return new_guess, delta_norm

def print_worker(worker, thing):
    print(f'worker {worker} is doing task {thing}')

x, y = sp.symbols('x y')
f_sym = get_F_sym(y**2+x**2-9, y-sp.sin(2*x)+2)
j_sym = get_J_sym(f_sym, [x, y])
f_lambda = sp.lambdify((x, y), f_sym, 'numpy')
j_lambda = sp.lambdify((x, y), j_sym, 'numpy')
# soln, delta_norm_history = newton_solve(f_sym, j_sym, {x: -5, y: 1}, [x, y], 0.00001, 1000)

# Do a randomised search to find unique solutions, stopping early if new unique solutions stop being found
converged_search_points_since_last_new_soln = 0
unique_solns = []
x_randoms = SEARCH_X_LIMS[0] + (SEARCH_X_LIMS[1]-SEARCH_X_LIMS[0])*np.random.random(MAX_SEARCH_POINTS)
y_randoms = SEARCH_Y_LIMS[0] + (SEARCH_Y_LIMS[1]-SEARCH_Y_LIMS[0])*np.random.random(MAX_SEARCH_POINTS)
point_count = 0
for x_rand, y_rand in zip(x_randoms, y_randoms):
    point_count += 1
    print(f'Searching point number {point_count}')
    soln, delta_norm_history = newton_solve(f_lambda, j_lambda, {x: x_rand, y: y_rand}, [x, y], EPSILON, MAX_ITERS)
    converged = len(delta_norm_history) < MAX_ITERS
    if converged:
        if not unique_solns:
            unique_solns.append(soln)
        if any([points_approx_equal(existing_soln, soln, EPSILON) for existing_soln in unique_solns]):
            converged_search_points_since_last_new_soln += 1
        else:
            converged_search_points_since_last_new_soln = 0
            unique_solns.append(soln)
        if converged_search_points_since_last_new_soln >= MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION:
            print(f'End search with {len(unique_solns)} unique solutions after reaching the limit of {MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION} consecutive converged search points since the last new unique solution')
            break

# Collapse to a grid of final image's aspect ratio and with some borders around the unique solutions
if len(unique_solns) < 2:
    print('Found fewer than 2 solutions, cannot generate an image')
    sys.exit(0)
soln_xs, soln_ys = [soln[x] for soln in unique_solns], [soln[y] for soln in unique_solns]
x_min, x_mean, x_max = np.min(soln_xs), np.mean(soln_xs), np.max(soln_xs)
y_min, y_mean, y_max = np.min(soln_ys), np.mean(soln_ys), np.max(soln_ys)
# Need to handle cases where solutions are collinear in x or y directions
if y_min == y_max:
    x_range = (x_max - x_min) * 4
    y_range = x_range * Y_PIXELS / X_PIXELS
elif x_min == x_max:
    y_range = (y_max - y_min) * 4
    x_range = y_range * X_PIXELS / Y_PIXELS
else:
    x_range = (x_max - x_min) * 4
    y_range = (y_max - y_min) * 4

x_coords = np.linspace(x_mean-x_range/2, x_mean+x_range/2, X_PIXELS)
y_coords = np.linspace(y_mean-y_range/2, y_mean+y_range/2, Y_PIXELS)

plt.figure()
x_solns = [soln[x] for soln in unique_solns]
y_solns = [soln[y] for soln in unique_solns]
plt.scatter(x_solns, y_solns)
plt.show()

N_THREADS = 1
# Split rows between threads
thread_rows = [[] for thread in range(N_THREADS)]
for row in range(Y_PIXELS):
    thread_rows[row % N_THREADS].append(row)

# Find the converged solution for each pixel - record it and the number of iterations
# Determine the base colours for each unique solution, and how they will be modified by number of iterations
soln_grid = [[] for row in range(Y_PIXELS)]
solutions = np.zeros([Y_PIXELS, X_PIXELS], dtype=int)
iterations = np.zeros([Y_PIXELS, X_PIXELS], dtype=int)
# @jit
def solve_rows(rows):
    for j in rows:
        print(f'Now calculating pixels for row {j+1} of {len(rows)}')
        y_init = y_coords[j]
        for i, x_init in enumerate(x_coords):
            soln, delta_norm_history = newton_solve(f_lambda, j_lambda, {x: x_init, y: y_init}, [x, y], EPSILON, MAX_ITERS)
            iters = len(delta_norm_history)
            iterations[j, i] = iters
            if iters < MAX_ITERS:
                match = None
                for soln_idx, unique_soln in enumerate(unique_solns):
                    if points_approx_equal(unique_soln, soln, EPSILON):
                        match = soln_idx + 1
                        break
                if match:
                    solutions[j, i] = match
                else:
                    print(f'WARNING: Image will ignore a novel solution found on the grid: {soln}')
                    solutions[j, i] = 0
            else:
                solutions[j, i] = 0

start = time.time()
# with concurrent.futures.ProcessPoolExecutor(max_workers=N_THREADS) as executor:
#     executor.map(solve_rows, thread_rows, list(range(N_THREADS)))
solve_rows(list(range(Y_PIXELS)))
duration = time.time()-start
print(f'Solution duration with {N_THREADS} threads was {duration}')

colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
pixel_grid = np.zeros([Y_PIXELS, X_PIXELS, 4], dtype=np.uint8)
for j in range(Y_PIXELS):
    for i in range(X_PIXELS):
        colour_idx = solutions[j, i] % len(colours)
        pixel_grid[j, i, :] = [int(x*255) for x in matplotlib.colors.to_rgba(colours[colour_idx])]
        pixel_grid[j, i, 3] -= min(iterations[j, i]*2, 100)
plt.figure()
plt.imshow(pixel_grid)
plt.show()

from PIL import Image

pilimg=Image.fromarray(pixel_grid[:,:,:], 'RGBA')
pilimg.save("img_path7.png")

