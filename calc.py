import config, newton

import numpy as np
import numba as nb
import sys
import matplotlib.pyplot as plt

nb.config.DISABLE_JIT = config.DISABLE_JIT


# Do a randomised search to find unique solutions, stopping early if new unique solutions stop being found
def find_all_unique_solutions(f_lambda, j_lambda, delta=0):
    unique_solns = []
    x_randoms = config.SEARCH_X_LIMS[0] + (config.SEARCH_X_LIMS[1]-config.SEARCH_X_LIMS[0])*np.random.random(config.MAX_SEARCH_POINTS)
    y_randoms = config.SEARCH_Y_LIMS[0] + (config.SEARCH_Y_LIMS[1]-config.SEARCH_Y_LIMS[0])*np.random.random(config.MAX_SEARCH_POINTS)
    point_count = 0
    converged_search_points_since_last_new_soln = 0
    for x_rand, y_rand in zip(x_randoms, y_randoms):
        point_count += 1
        # print(f'Searching point number {point_count}')
        soln, delta_norm_history = newton.solve(f_lambda, j_lambda, (x_rand, y_rand), delta)
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
                # print(f'End search with {len(unique_solns)} unique solutions after reaching the limit of '
                #       f'{config.MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION} consecutive converged search points since the last new unique solution')
                break

    if config.SHOW_UNIQUE_SOLUTIONS_AND_EXIT:
        plt.figure()
        x_solns = [soln[0] for soln in unique_solns]
        y_solns = [soln[1] for soln in unique_solns]
        plt.scatter(x_solns, y_solns)
        plt.show(block=True)
        sys.exit(0)

    if len(unique_solns) < 2:
        print('Found fewer than 2 unique solutions, cannot generate an image')
        sys.exit(0)

    return sorted(unique_solns)


def get_image_pixel_coords(unique_solns):
    # Collapse to a grid of final image's aspect ratio and with some borders around the unique solutions
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

    return np.linspace(x_mean-x_range/2, x_mean+x_range/2, config.X_PIXELS),\
           np.linspace(y_mean-y_range/2, y_mean+y_range/2, config.Y_PIXELS)

@nb.njit(target_backend=config.NUMBA_TARGET)
def points_approx_equal(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < 2 * config.EPSILON



# Find the converged solution for each pixel - record it and the number of iterations
# Determine the base colours for each unique solution, and how they will be modified by number of iterations
@nb.njit(target_backend=config.NUMBA_TARGET)
def solve_grid(unique_solutions, x_coords, y_coords, f_lambda, j_lambda, delta=0):
    solutions_local = np.zeros((config.Y_PIXELS, config.X_PIXELS), dtype=np.int_)
    iterations_local = np.zeros((config.Y_PIXELS, config.X_PIXELS), dtype=np.int_)
    for j in range(config.Y_PIXELS):
        # print(f'Now calculating pixels for row {j+1} of {config.Y_PIXELS}')
        y_init = y_coords[j]
        for i, x_init in enumerate(x_coords):
            soln, delta_norm_hist = newton.solve(f_lambda, j_lambda, (x_init, y_init), delta)
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
