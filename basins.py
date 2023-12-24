import numpy as np
import sympy as sp
import numba as nb
import time
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from typing import Tuple, List
from PIL import Image
from datetime import datetime
from pathlib import Path

nb.config.DISABLE_JIT = False

SEARCH_X_LIMS = [-1000, 1000]
SEARCH_Y_LIMS = [-1000, 1000]
MAX_SEARCH_POINTS = 500
MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION = 50
MAX_ITERS = 1000
EPSILON = 0.0001
X_PIXELS = int(3440)
Y_PIXELS = int(1440)
target = 'CPU' # setting cuda is not doing anything

@nb.njit(target_backend=target)
def points_approx_equal(p1, p2, EPSILON):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < 2*EPSILON

def get_F_sym(f, g):
    return [f, g]

def get_J_sym(expressions, symbols):
    return [[sp.diff(exp, sym) for sym in symbols] for exp in expressions]

@nb.njit(target_backend=target)
def newton_solve(f_lambda, j_lambda, starting_guess, epsilon, max_iters) -> Tuple[Tuple[float, float], List[float]]:
    current_guess = starting_guess
    delta_norm_history = []
    delta_norm = 1000
    n_iters = 0
    while delta_norm > epsilon and n_iters < max_iters:
        current_guess, delta_norm = newton_iteration(f_lambda, j_lambda, current_guess)
        delta_norm_history.append(delta_norm)
        n_iters += 1

    return current_guess, delta_norm_history

@nb.njit(target_backend=target)
def newton_iteration(f_lambda, j_lambda, guess) -> Tuple[Tuple[float, float], float]:
    f_num = f_lambda(guess[0], guess[1])
    j_num = j_lambda(guess[0], guess[1])
    delta = np.linalg.solve(np.array(j_num), -np.array(f_num)).flatten()
    new_guess = (guess[0] + delta[0], guess[1] + delta[1])
    delta_norm = np.linalg.norm(delta)
    return new_guess, delta_norm


x, y = sp.symbols('x y')
f_sym = get_F_sym(y**2+x**2-9, y-sp.sin(2*x)+2)
j_sym = get_J_sym(f_sym, [x, y])
f_lambda = nb.njit(sp.lambdify((x, y), f_sym, 'numpy'), target_backend=target)
j_lambda = nb.njit(sp.lambdify((x, y), j_sym, 'numpy'), target_backend=target)

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
    soln, delta_norm_history = newton_solve(f_lambda, j_lambda, (x_rand, y_rand), EPSILON, MAX_ITERS)
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
soln_xs, soln_ys = [soln[0] for soln in unique_solns], [soln[1] for soln in unique_solns]
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
x_solns = [soln[0] for soln in unique_solns]
y_solns = [soln[1] for soln in unique_solns]
plt.scatter(x_solns, y_solns)
plt.show()


# Find the converged solution for each pixel - record it and the number of iterations
# Determine the base colours for each unique solution, and how they will be modified by number of iterations
@nb.njit(target_backend=target)
def solve_grid(unique_solutions):
    solutions_local = np.zeros((Y_PIXELS, X_PIXELS), dtype=np.int_)
    iterations_local = np.zeros((Y_PIXELS, X_PIXELS), dtype=np.int_)
    for j in range(Y_PIXELS):
        print(f'Now calculating pixels for row {j+1} of {Y_PIXELS}')
        y_init = y_coords[j]
        for i, x_init in enumerate(x_coords):
            # print(f'Now calculating pixel for row {j + 1} of {Y_PIXELS}, col {i + 1} of {X_PIXELS}')
            soln, delta_norm_history = newton_solve(f_lambda, j_lambda, (x_init, y_init), EPSILON, MAX_ITERS)
            iters = len(delta_norm_history)
            iterations_local[j, i] = iters
            if iters < MAX_ITERS:
                match = -1
                for soln_idx, unique_soln in enumerate(unique_solutions):
                    if points_approx_equal(unique_soln, soln, EPSILON):
                        match = soln_idx + 1
                        break
                if match != -1:
                    solutions_local[j, i] = match
                else:
                    print(f'WARNING: Image will ignore a novel solution found on the grid: {soln}')
                    solutions_local[j, i] = 0
            else:
                solutions_local[j, i] = 0
    return solutions_local, iterations_local
start = time.time()
solutions, iterations = solve_grid(tuple(unique_solns))
duration = time.time()-start
print(f'Solution duration was {duration}')

def smooth_grid(solutions):
    # smoothed = np.zeros((Y_PIXELS, X_PIXELS), dtype=np.int_)
    smoothed_count = 0
    smooth_solutions = solutions.copy()
    for j in range(1, smooth_solutions.shape[0]-1):
        for i in range(1, smooth_solutions.shape[1]-1):
            adjacent_solutions = {}
            diagonal_solutions = {}
            for delta_j in [-1, 0, 1]:
                for delta_i in [-1, 0, 1]:
                    soln = smooth_solutions[j + delta_j, i + delta_i]
                    if delta_j == delta_i == 0:
                        break
                    elif delta_j == 0 or delta_i == 0:
                        if soln not in adjacent_solutions:
                            adjacent_solutions[soln] = 1
                        else:
                            adjacent_solutions[soln] += 1
                    else:
                        if soln not in diagonal_solutions:
                            diagonal_solutions[soln] = 1
                        else:
                            diagonal_solutions[soln] += 1

            # Smooth pixel if its only equivalent neighbour is a diagonal (or no equivalent neighbours)
            if smooth_solutions[j, i] not in adjacent_solutions\
                    and (diagonal_solutions.get(smooth_solutions[j, i], 0) < 2):
                # It becomes the same solution as its most prevalent neighbour
                current_max = 0
                current_soln = smooth_solutions[j, i]
                for neighbour in set(adjacent_solutions.keys()).union(diagonal_solutions.keys()):
                    count = adjacent_solutions.get(neighbour, 0) + diagonal_solutions.get(neighbour, 0)
                    if count > current_max:
                        current_max = count
                        current_soln = neighbour
                smooth_solutions[j, i] = current_soln
                smoothed_count += 1

    print(f'Smoothed {smoothed_count}/{Y_PIXELS*X_PIXELS} pixels')
    return smooth_solutions

solutions_smooth = smooth_grid(solutions)



images_dir = Path().cwd()/f'images/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
images_dir.mkdir(parents=True)
colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
pixel_grid = np.zeros([Y_PIXELS, X_PIXELS, 3], dtype=np.uint8)

# Generate images with multiple colour schemes
for colour_offset in range(len(colours)):
    for j in range(Y_PIXELS):
        for i in range(X_PIXELS):
            colour_idx = (solutions_smooth[j, i] + colour_offset) % len(colours)
            pixel_grid[j, i, :] = [int(x*255) for x in matplotlib.colors.to_rgb(colours[colour_idx])]
    Image.fromarray(pixel_grid, 'RGBA').save(images_dir / f'{colour_offset}.png')


# TODO: use the iterations counts to blend solution colours