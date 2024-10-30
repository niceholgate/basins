import src.config as cfg
import src.utils as utils
from src.solving import solve

import os
import sys
from pathlib import Path
import numpy as np
import numpy.typing as npt
from numba.pycc import CC
from numba import int32, float64, types
from typing import Tuple, Callable, List, Optional
import taichi as ti

MODULE_NAME = 'solving_interface'
cc = CC(MODULE_NAME)
try:
    src_dir = Path(os.path.realpath(__file__)).parent
    build_dir = src_dir.parent/'build'
    sys.path.append(str(build_dir))
    import solving_interface
    PRECOMPILED = True
except:
    PRECOMPILED = False


def find_unique_solutions_wrapper(f_lambda: Callable, j_lambda: Callable, delta: float, search_limits: npt.NDArray) -> Optional[npt.NDArray]:
    if PRECOMPILED and cfg.ENABLE_AOT:
        return solving_interface.find_unique_solutions(f_lambda, j_lambda, delta, search_limits)
    print('Used pythonic find_unique_solutions')
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
    if PRECOMPILED and cfg.ENABLE_AOT:
        return solving_interface.get_image_pixel_coords(y_pixels, x_pixels, unique_solutions)
    print('Used pythonic get_image_pixel_coords')
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
    if PRECOMPILED and cfg.ENABLE_AOT:
        return solving_interface.solve_grid(f_lambda, j_lambda, x_coords, y_coords, delta, unique_solutions)
    print('Used pythonic solve_grid')
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
    solver = solve.create_solver(ti.func(f_lambda), ti.func(j_lambda), x_coords, y_coords, delta, unique_solutions)
    print('starting solve')
    solver.solve_grid()
    print('finishing solve')
    return solver.solutions_grid, solver.iterations_grid


def solve_grid_quadtrees_wrapper(f_lambda: Callable, j_lambda: Callable, x_coords: npt.NDArray, y_coords: npt.NDArray, delta: float, unique_solutions: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    if PRECOMPILED and cfg.ENABLE_AOT:
        return solving_interface.solve_grid_quadtrees(f_lambda, j_lambda, x_coords, y_coords, delta, unique_solutions)
    print('Used pythonic solve_grid_quadtrees')
    return solve_grid_quadtrees(f_lambda, j_lambda, x_coords, y_coords, delta, unique_solutions)


@cc.export('solve_grid_quadtrees', solve_grid_spec)
def solve_grid_quadtrees(f_lambda: Callable, j_lambda: Callable, x_coords: npt.NDArray, y_coords: npt.NDArray, delta: float, unique_solutions: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    solver = solve.create_solver(f_lambda, j_lambda, x_coords, y_coords, delta, unique_solutions)
    solver.solve_grid_quadtrees()
    return solver.solutions_grid, solver.iterations_grid


# TODO: one script that recreates all of the precompiled modules
compiled_module_file_exists = any([x for x in cfg.BUILD_DIR.glob(f'{MODULE_NAME}*') if x.is_file()])
if cfg.ENABLE_AOT:
    if compiled_module_file_exists:
        print(f'Using existing numba Ahead-Of-Time compiled files for module: {MODULE_NAME}')
    else:
        print(f'Performing Ahead-Of-Time numba compilation for module: {MODULE_NAME}')
        cc.compile()
        src_dir = Path(os.path.realpath(__file__)).parent
        compiled_module_file = [x for x in src_dir.glob(f'{MODULE_NAME}*') if x.is_file()][0]
        utils.mkdir_if_nonexistent(cfg.BUILD_DIR)
        file_dest = cfg.BUILD_DIR / compiled_module_file.name
        file_dest.unlink(missing_ok=True)
        compiled_module_file.rename(file_dest)
        PRECOMPILED = True

