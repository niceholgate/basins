import src.config as cfg
import src.utils as utils

import numpy as np
import numpy.typing as npt
import numba as nb
import sys
from typing import Tuple, Callable, List

nb.config.DISABLE_JIT = cfg.DISABLE_JIT


# @nb.njit(target_backend=cfg.NUMBA_TARGET)
def find_unique_solutions(f_lamb: Callable, j_lamb: Callable, delta: float) -> npt.NDArray:
    """Do a randomised search to find unique solutions, stopping early if new unique solutions stop being found."""
    unique_solns: List[npt.NDArray] = []
    x_randoms = cfg.SEARCH_X_LIMS[0] + (cfg.SEARCH_X_LIMS[1]-cfg.SEARCH_X_LIMS[0])*np.random.random(cfg.MAX_SEARCH_POINTS)
    y_randoms = cfg.SEARCH_Y_LIMS[0] + (cfg.SEARCH_Y_LIMS[1]-cfg.SEARCH_Y_LIMS[0])*np.random.random(cfg.MAX_SEARCH_POINTS)
    randoms = np.array([x_randoms, y_randoms])
    point_count = 0
    converged_search_points_since_last_new_soln = 0
    for idx in range(randoms.shape[1]):
        point_count += 1
        soln, delta_norm_history = newton_solve(f_lamb, j_lamb, randoms[:, idx], delta)
        converged = len(delta_norm_history) < cfg.MAX_ITERS
        if converged:
            if not unique_solns:
                unique_solns.append(soln)
            if any([_points_approx_equal(existing_soln, soln) for existing_soln in unique_solns]):
                converged_search_points_since_last_new_soln += 1
            else:
                converged_search_points_since_last_new_soln = 0
                unique_solns.append(soln)
            if converged_search_points_since_last_new_soln >= cfg.MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION:
                print(f'End search with {len(unique_solns)} unique solutions after reaching the limit of '
                      f'{cfg.MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION} consecutive converged search points'
                      f' since the last new unique solution.')
                break

    # Temporarily convert the solutions to tuples to sort them (ensures the random search returns the same result each
    # time for a given system of equations) then put them into one 2D array
    unique_solns_arr = np.array(sorted([tuple(s) for s in unique_solns]))

    if len(unique_solns) < 2:
        print('Found fewer than 2 unique solutions, cannot generate an image')
        sys.exit(0)

    if cfg.SHOW_UNIQUE_SOLUTIONS_AND_EXIT:
        utils.plot_unique_solutions(unique_solns_arr)
        sys.exit(0)

    return unique_solns_arr


# @nb.njit(target_backend=cfg.NUMBA_TARGET)
def get_image_pixel_coords(y_pixels: int, x_pixels: int, unique_solns: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """Get the coordinates of all the pixels in the image, ensuring that all the known solutions to the system of
    equations fall safely within the bounds of the image."""
    # Collapse to a grid of final image's aspect ratio and with some borders around the unique solutions
    x_min, x_mean, x_max = unique_solns[:, 0].min(), unique_solns[:, 0].mean(), unique_solns[:, 0].max()
    y_min, y_mean, y_max = unique_solns[:, 1].min(), unique_solns[:, 1].mean(), unique_solns[:, 1].max()

    # Need to handle cases where solutions are collinear in x or y directions
    if y_min == y_max:
        x_range = (x_max - x_min) * 4
        y_range = x_range * y_pixels / x_pixels
    elif x_min == x_max:
        y_range = (y_max - y_min) * 4
        x_range = y_range * x_pixels / y_pixels
    else:
        x_range = (x_max - x_min) * 4
        y_range = (y_max - y_min) * 4

    return np.linspace(x_mean-x_range/2, x_mean+x_range/2, x_pixels),\
           np.linspace(y_mean-y_range/2, y_mean+y_range/2, y_pixels)

def get_index_of_matching_unique_soln(unique_solutions: npt.NDArray, soln: npt.NDArray) -> int:
    match = -1
    for unique_soln_idx in range(unique_solutions.shape[0]):
        if _points_approx_equal(unique_solutions[unique_soln_idx, :], soln):
            match = unique_soln_idx + 1
            break
    return match


@nb.njit(target_backend=cfg.NUMBA_TARGET)
def solve_grid(unique_solutions: npt.NDArray, x_coords: npt.NDArray, y_coords: npt.NDArray, f_lambda: Callable,
               j_lambda: Callable, delta: float) -> Tuple[npt.NDArray, npt.NDArray]:
    """Find which unique solution is reached for each pixel, and how many Newton's method iterations it took."""
    solutions_local = np.zeros((y_coords.shape[0], x_coords.shape[0]), dtype=np.int_)
    iterations_local = np.zeros((y_coords.shape[0], x_coords.shape[0]), dtype=np.int_)
    for j in range(y_coords.shape[0]):
        y_init = y_coords[j]
        for i, x_init in enumerate(x_coords):
            soln, delta_norm_hist = newton_solve(f_lambda, j_lambda, np.array([x_init, y_init]), delta)
            iters = len(delta_norm_hist)
            iterations_local[j, i] = iters
            if iters < cfg.MAX_ITERS:
                match = get_index_of_matching_unique_soln(unique_solutions, soln)
                if match != -1:
                    solutions_local[j, i] = match
                else:
                    print(f'WARNING: Image will ignore a novel solution found on the grid: {soln}')
                    solutions_local[j, i] = 0
            else:
                solutions_local[j, i] = 0

    return solutions_local, iterations_local


@nb.njit(target_backend=cfg.NUMBA_TARGET)
def solve_grid_quadtrees(unique_solutions: npt.NDArray, x_coords: npt.NDArray, y_coords: npt.NDArray, f_lambda: Callable,
               j_lambda: Callable, delta: float) -> Tuple[npt.NDArray, npt.NDArray]:
    """Find which unique solution is reached for each pixel, and how many Newton's method iterations it took."""
    solutions_local = np.zeros((y_coords.shape[0], x_coords.shape[0]), dtype=np.int_)
    iterations_local = np.zeros((y_coords.shape[0], x_coords.shape[0]), dtype=np.int_)
    # for j in range(y_coords.shape[0]):
    #     y_init = y_coords[j]
    #     for i, x_init in enumerate(x_coords):


    # Create a stack of QuadTrees?
    stack = [QuadTree((0, x_coords.shape[0]), (0, y_coords.shape[0]))]

    # Iterate around the edges of the current QuadTree calculating the solution, and note if they are all identical
    qt = stack.pop()
    last_soln = None
    identical_solns = True
    for j in range(qt.y_lims[0], qt.y_lims[1]+1):
        y_init = y_coords[j]
        for i, x_init in qt.x_lims:
            soln, delta_norm_hist = newton_solve(f_lambda, j_lambda, np.array([x_init, y_init]), delta)
            iters = len(delta_norm_hist)
            iterations_local[j, i] = iters
            if iters < cfg.MAX_ITERS:
                match = get_index_of_matching_unique_soln(unique_solutions, soln)
                if match != -1:
                    solutions_local[j, i] = match
                    if identical_solns:
                        if last_soln is None:
                            last_soln = match
                        else:
                            if match != last_soln:
                                identical_solns = False

                else:
                    print(f'WARNING: Image will ignore a novel solution found on the grid: {soln}')
                    solutions_local[j, i] = 0
                    identical_solns = False
                    last_soln = None
            else:
                solutions_local[j, i] = 0
                identical_solns = False
                last_soln = None


    return solutions_local, iterations_local




@nb.njit(target_backend=cfg.NUMBA_TARGET)
def newton_solve(f_lam: Callable, j_lam: Callable, starting_guess: npt.NDArray, d: float) -> Tuple[npt.NDArray, npt.NDArray]:
    """Perform Newton's method until either the solution converges or the maximum iterations are exceeded."""
    current_guess = starting_guess
    delta_norm_hist = []
    delta_norm = np.float_(1000.0)
    n_iters = 0
    while delta_norm > cfg.EPSILON and n_iters < cfg.MAX_ITERS:
        f_num = f_lam(current_guess[0], current_guess[1], d)
        j_num = j_lam(current_guess[0], current_guess[1], d)
        delta = np.linalg.solve(np.array(j_num), -np.array(f_num)).flatten()
        current_guess = current_guess + delta
        delta_norm = np.linalg.norm(delta)
        delta_norm_hist.append(delta_norm)
        n_iters += 1

    return current_guess, np.array(delta_norm_hist)


@nb.njit(target_backend=cfg.NUMBA_TARGET)
def _points_approx_equal(p1: npt.NDArray, p2: npt.NDArray) -> bool:
    return bool(np.linalg.norm(p1 - p2) < 2 * cfg.EPSILON)


class QuadTree:
    x_lims: Tuple[int, int]
    y_lims: Tuple[int, int]
    nw: 'QuadTree'
    ne: 'QuadTree'
    sw: 'QuadTree'
    se: 'QuadTree'

    def __init__(self, x_lims: Tuple[int, int], y_lims: Tuple[int, int]):
        self.x_lims = x_lims
        self.y_lims = y_lims

    def iterate_boundary_coordinates(self):
        for x in range(self.x_lims[0], self.x_lims[1] + 1):
            yield self.y_lims[0], x
        for y in range(self.y_lims[0], self.y_lims[1] + 1):
            yield y, self.x_lims[1]
        for x in reversed(range(self.x_lims[0], self.x_lims[1] + 1)):
            yield self.y_lims[1], x
        for y in reversed(range(self.y_lims[0], self.y_lims[1] + 1)):
            yield y, self.x_lims[0]

    def subdivide(self):
        x_mid = np.ceil(np.mean(self.x_lims))
        y_mid = np.ceil(np.mean(self.y_lims))

        # Cannot subdivide a point
        if self.x_lims[0] == self.x_lims[1] and self.y_lims[0] == self.y_lims[1]:
            return
        # For a single column, can only subdivide in y direction
        if self.x_lims[0] == self.x_lims[1]:
            self.nw = QuadTree((x_mid, x_mid), (self.y_lims[0], y_mid))
            self.sw = QuadTree()
        # For a single row, can only subdivide in x direction


        self.nw = QuadTree((self.x_lims[0], x_mid), (self.y_lims[0], y_mid))
        self.ne = QuadTree