import config
import newton
import smoothing
import blending

import numpy as np
import sympy as sp
import numba as nb
import sys
import matplotlib
from PIL import Image
from datetime import datetime
from pathlib import Path

nb.config.DISABLE_JIT = config.DISABLE_JIT
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


@nb.njit(target_backend=config.NUMBA_TARGET)
def points_approx_equal(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < 2 * config.EPSILON


x, y = sp.symbols('x y')
f_sym = [y**2+x**2-9, y-2*sp.cos(2*x)*sp.exp(-0.01*(x+y)**2)+2]
j_sym = [[sp.diff(exp, sym) for sym in [x, y]] for exp in f_sym]
f_lambda = nb.njit(sp.lambdify((x, y), f_sym, 'numpy'), target_backend=config.NUMBA_TARGET)
j_lambda = nb.njit(sp.lambdify((x, y), j_sym, 'numpy'), target_backend=config.NUMBA_TARGET)

# Do a randomised search to find unique solutions, stopping early if new unique solutions stop being found
converged_search_points_since_last_new_soln = 0
unique_solns = []
x_randoms = config.SEARCH_X_LIMS[0] + (config.SEARCH_X_LIMS[1]-config.SEARCH_X_LIMS[0])*np.random.random(config.MAX_SEARCH_POINTS)
y_randoms = config.SEARCH_Y_LIMS[0] + (config.SEARCH_Y_LIMS[1]-config.SEARCH_Y_LIMS[0])*np.random.random(config.MAX_SEARCH_POINTS)
point_count = 0
for x_rand, y_rand in zip(x_randoms, y_randoms):
    point_count += 1
    print(f'Searching point number {point_count}')
    soln, delta_norm_history = newton.solve(f_lambda, j_lambda, (x_rand, y_rand))
    converged = len(delta_norm_history) < config.MAX_ITERS
    if converged:
        if not unique_solns:
            unique_solns.append(soln)
        if any([points_approx_equal(existing_soln, soln) for existing_soln in unique_solns]):
            converged_search_points_since_last_new_soln += 1
        else:
            converged_search_points_since_last_new_soln = 0
            unique_solns.append(soln)
        if converged_search_points_since_last_new_soln >= config.MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION:
            print(f'End search with {len(unique_solns)} unique solutions after reaching the limit of '
                  f'{config.MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION} consecutive converged search points since the last new unique solution')
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
    y_range = x_range * config.Y_PIXELS / config.X_PIXELS
elif x_min == x_max:
    y_range = (y_max - y_min) * 4
    x_range = y_range * config.X_PIXELS / config.Y_PIXELS
else:
    x_range = (x_max - x_min) * 4
    y_range = (y_max - y_min) * 4

x_coords = np.linspace(x_mean-x_range/2, x_mean+x_range/2, config.X_PIXELS)
y_coords = np.linspace(y_mean-y_range/2, y_mean+y_range/2, config.Y_PIXELS)

if config.SHOW_UNIQUE_SOLUTIONS_AND_EXIT:
    plt.figure()
    x_solns = [soln[0] for soln in unique_solns]
    y_solns = [soln[1] for soln in unique_solns]
    plt.scatter(x_solns, y_solns)
    plt.show(block=True)
    sys.exit(0)



# Find the converged solution for each pixel - record it and the number of iterations
# Determine the base colours for each unique solution, and how they will be modified by number of iterations
@nb.njit(target_backend=config.NUMBA_TARGET)
def solve_grid(unique_solutions):
    solutions_local = np.zeros((config.Y_PIXELS, config.X_PIXELS), dtype=np.int_)
    iterations_local = np.zeros((config.Y_PIXELS, config.X_PIXELS), dtype=np.int_)
    for j in range(config.Y_PIXELS):
        print(f'Now calculating pixels for row {j+1} of {config.Y_PIXELS}')
        y_init = y_coords[j]
        for i, x_init in enumerate(x_coords):
            soln, delta_norm_hist = newton.solve(f_lambda, j_lambda, (x_init, y_init))
            iters = len(delta_norm_hist)
            iterations_local[j, i] = iters
            if iters < config.MAX_ITERS:
                match = -1
                for soln_idx, unique_soln in enumerate(unique_solutions):
                    if points_approx_equal(unique_soln, soln):
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


solutions, iterations = solve_grid(tuple(unique_solns))
smoothed_solutions = smoothing.smooth_grid(solutions)


images_dir = Path().cwd()/f'images/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
images_dir.mkdir(parents=True)
colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
pixel_grid = np.zeros((config.Y_PIXELS, config.X_PIXELS, 3), dtype=np.uint8)


# Generate images with multiple colour schemes
# for colour_offset in range(len(colours)):
decay_facs = np.array([4.0])
blending_arrays = blending.create_blending_arrays(iterations, decay_facs)
for colour_offset in range(len(colours)):
    for j in range(config.Y_PIXELS):
        for i in range(config.X_PIXELS):
            colour_idx = (smoothed_solutions[j, i] + colour_offset) % len(colours)
            pixel_grid[j, i, :] = [int(x*255) for x in matplotlib.colors.to_rgb(colours[colour_idx])]
    blended_pixel_grid = blending.blend_grid(pixel_grid, blending_arrays, 0)
    Image.fromarray(blended_pixel_grid, 'RGB').save(images_dir / f'{colour_offset}_blended.png')

