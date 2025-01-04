import src.config as cfg
from src.solving import solve

import os
import sys
import numpy as np
import numpy.typing as npt
from numba.pycc import CC
from numba import int32, float64, types
from typing import Tuple, Callable, List, Optional
import taichi as ti

MODULE_NAME = 'solving_interface'
MODULE_PATH = os.path.realpath(__file__)
cc = CC(MODULE_NAME)
try:
    sys.path.append(str(cfg.BUILD_DIR))
    import solving_interface
    PRECOMPILED = True
except:
    PRECOMPILED = False


def find_unique_solutions_wrapper(f_lambda: Callable, j_lambda: Callable, delta: float, search_limits: npt.NDArray) -> Optional[npt.NDArray]:
    if PRECOMPILED and cfg.ENABLE_NUMBA:
        return solving_interface.find_unique_solutions(f_lambda, j_lambda, delta, search_limits)
    return find_unique_solutions(f_lambda, j_lambda, delta, search_limits)


find_unique_solutions_spec = (
    types.List(float64)(float64, float64, float64).as_type(),
    types.List(types.List(float64))(float64, float64, float64).as_type(),
    float64,
    float64[:]
)
@cc.export('find_unique_solutions', find_unique_solutions_spec)
def find_unique_solutions(f_lambda: Callable, j_lambda: Callable, delta: float, search_limits: npt.NDArray) -> Optional[npt.NDArray]:

    """Do a randomised search to find unique solutions, stopping early if new unique solutions stop being found."""
    unique_solns: List[npt.NDArray] = []
    x_randoms = search_limits[0] + (search_limits[1] - search_limits[0]) * np.random.random(cfg.MAX_SEARCH_POINTS)
    y_randoms = search_limits[2] + (search_limits[3] - search_limits[2]) * np.random.random(cfg.MAX_SEARCH_POINTS)
    # randoms = np.array([x_randoms, y_randoms], dtype=np.float64)
    point_count = 0
    converged_search_points_since_last_new_soln = 0
    for idx in range(x_randoms.shape[0]):
        point_count += 1
        soln, iters = solve.newton_solve(f_lambda, j_lambda, np.array([x_randoms[idx], y_randoms[idx]], dtype=np.float64), delta)
        # print(f'Solution: {soln} and iters: {iters} with initial guess: {[x_randoms[idx], y_randoms[idx]]}')
        if iters < cfg.MAX_ITERS:
            if not unique_solns:
                unique_solns.append(soln)

            any_equal = False
            for existing_soln in unique_solns:
                if solve.points_approx_equal(existing_soln, soln):
                    any_equal = True
                    break
            if any_equal:
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
    unique_solns_arr = np.array(sorted([(s[0], s[1]) for s in unique_solns]), np.float64)

    if len(unique_solns) < 2:
        # print('Found fewer than 2 unique solutions, cannot generate an image')
        return None

    return unique_solns_arr


def get_image_pixel_coords_wrapper(y_pixels: int, x_pixels: int, unique_solutions: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    if PRECOMPILED and cfg.ENABLE_NUMBA:
        return solving_interface.get_image_pixel_coords(y_pixels, x_pixels, unique_solutions)
    return get_image_pixel_coords(y_pixels, x_pixels, unique_solutions)


@cc.export('get_image_pixel_coords', (int32, int32, float64[:, :]))
def get_image_pixel_coords(y_pixels: int, x_pixels: int, unique_solutions: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """Get the coordinates of all the pixels in the image, ensuring that all the known solutions to the system of
    equations fall safely within the bounds of the image."""
    # Collapse to a grid of final image's aspect ratio and with some borders around the unique solutions
    x_min, x_mean, x_max = unique_solutions[:, 0].min(), unique_solutions[:, 0].mean(), unique_solutions[:, 0].max()
    y_min, y_mean, y_max = unique_solutions[:, 1].min(), unique_solutions[:, 1].mean(), unique_solutions[:, 1].max()
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

    return np.linspace(x_mean - x_range / 2, x_mean + x_range / 2, x_pixels), \
           np.linspace(y_mean - y_range / 2, y_mean + y_range / 2, y_pixels)


def solve_grid_wrapper(f_lambda: Callable, j_lambda: Callable, x_coords: npt.NDArray, y_coords: npt.NDArray, delta: float, unique_solutions: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    if cfg.ENABLE_TAICHI:
        return solve_grid_taichi(f_lambda, j_lambda, x_coords, y_coords, delta, unique_solutions)
    if PRECOMPILED and cfg.ENABLE_NUMBA:
        return solving_interface.solve_grid(f_lambda, j_lambda, x_coords, y_coords, delta, unique_solutions)
    return solve_grid(f_lambda, j_lambda, x_coords, y_coords, delta, unique_solutions)


solve_grid_spec = (
    types.List(float64)(float64, float64, float64).as_type(),
    types.List(types.List(float64))(float64, float64, float64).as_type(),
    float64[:],
    float64[:],
    float64,
    float64[:, :]
)
@cc.export('solve_grid', solve_grid_spec)
def solve_grid(f_lambda: Callable, j_lambda: Callable, x_coords: npt.NDArray, y_coords: npt.NDArray, delta: float, unique_solutions: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    solver = solve.create_solver(f_lambda, j_lambda, x_coords, y_coords, delta, unique_solutions)
    solver.solve_grid()
    return solver.solutions_grid, solver.iterations_grid


def solve_grid_taichi(f_lambda: Callable, j_lambda: Callable, x_coords: npt.NDArray, y_coords: npt.NDArray, delta: float, unique_solutions: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    solver = solve.create_solver_taichi(ti.func(f_lambda), ti.func(j_lambda), x_coords, y_coords, delta, unique_solutions)
    solver.solve_grid()
    return solver.solutions_grid.to_numpy(), solver.iterations_grid.to_numpy()


def solve_grid_quadtrees_wrapper(f_lambda: Callable, j_lambda: Callable, x_coords: npt.NDArray, y_coords: npt.NDArray, delta: float, unique_solutions: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    if PRECOMPILED and cfg.ENABLE_NUMBA:
        return solving_interface.solve_grid_quadtrees(f_lambda, j_lambda, x_coords, y_coords, delta, unique_solutions)
    return solve_grid_quadtrees(f_lambda, j_lambda, x_coords, y_coords, delta, unique_solutions)


@cc.export('solve_grid_quadtrees', solve_grid_spec)
def solve_grid_quadtrees(f_lambda: Callable, j_lambda: Callable, x_coords: npt.NDArray, y_coords: npt.NDArray, delta: float, unique_solutions: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    solver = solve.create_solver(f_lambda, j_lambda, x_coords, y_coords, delta, unique_solutions)
    solver.solve_grid_quadtrees()
    return solver.solutions_grid, solver.iterations_grid
