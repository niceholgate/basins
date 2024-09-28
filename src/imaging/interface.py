import src.config as cfg
import src.utils as utils
from src.solving.solve import Solver

import os
import sys
import matplotlib
from numba.pycc import CC
from numba import int32, float64, types
from PIL import Image
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Union, List, Dict


MODULE_NAME = 'imaging_interface'
cc = CC(MODULE_NAME)
try:
    src_dir = Path(os.path.realpath(__file__)).parent
    build_dir = src_dir.parent/'build'
    sys.path.append(str(build_dir))
    import imaging_interface
    PRECOMPILED = True
except:
    PRECOMPILED = False


# TODO: remove this from interface
def save_still(images_dir: Path, solver: Solver, smoothing: bool = True, blending: bool = True, colour_set: Union[int, List[str]] = 0, frame: int = 0):
    """Generate an image with the specified colour scheme, or a range of colour schemes if none specified."""
    pixel_grid = np.zeros((solver.solutions_grid.shape[0], solver.solutions_grid.shape[1], 3), dtype=np.uint8)
    smoothed_solutions = smooth_grid(solver.solutions_grid) if smoothing else solver.solutions_grid
    blending_arrays = create_blending_arrays(solver.iterations_grid) if blending else []

    unique_solns_this_delta = solver.unique_solutions

    if not (isinstance(colour_set, list) and len(colour_set) == unique_solns_this_delta.shape[0]):
        idx = colour_set if isinstance(colour_set, int) else 0
        colour_set = []
        while len(colour_set) < unique_solns_this_delta.shape[0]:
            colour_set.append(cfg.DEFAULT_COLOURS[idx % len(cfg.DEFAULT_COLOURS)])
            idx += 1
    rgb_colours = [matplotlib.colors.to_rgb(colour) for colour in colour_set]

    for j in range(solver.solutions_grid.shape[0]):
        for i in range(solver.solutions_grid.shape[1]):
            if solver.iterations_grid[j, i] < cfg.BLACKOUT_ITERS:
                pixel_grid[j, i, :] = [int(x*255) for x in rgb_colours[smoothed_solutions[j, i]-1]]
    blended_pixel_grid = blend_grid(pixel_grid, blending_arrays, 0) if blending else pixel_grid
    np.savetxt(images_dir / utils.get_frame_filename(frame, 'txt'), blended_pixel_grid.reshape([blended_pixel_grid.shape[0],
               blended_pixel_grid.shape[1]*blended_pixel_grid.shape[2]]), fmt='%u')
    if cfg.SAVE_PNG_FRAMES:
        Image.fromarray(blended_pixel_grid, 'RGB').save(images_dir / utils.get_frame_filename(frame, 'png'))


def smooth_grid_wrapper(solutions: npt.NDArray) -> npt.NDArray:
    if PRECOMPILED:
        return imaging_interface.smooth_grid(solutions)
    return smooth_grid(solutions)


smooth_grid_spec = (
    int32[:, :]
)
@cc.export('smooth_grid', smooth_grid_spec)
def smooth_grid(solutions: npt.NDArray) -> npt.NDArray:
    """Smooth a grid of solutions (happens before colouration).
    Just a crude algo to slightly reduce noise in very unstable areas."""
    smoothed_count = 0
    smoothed_solutions = solutions.copy()
    for j in range(1, smoothed_solutions.shape[0]-1):
        for i in range(1, smoothed_solutions.shape[1]-1):
            adjacent_solutions: Dict[int, int] = {}
            diagonal_solutions: Dict[int, int] = {}
            for delta_j in [-1, 0, 1]:
                for delta_i in [-1, 0, 1]:
                    soln = smoothed_solutions[j + delta_j, i + delta_i]
                    if delta_j == delta_i == 0:
                        break
                    elif delta_j == 0 or delta_i == 0:
                        adjacent_solutions[soln] = adjacent_solutions[soln] + 1 if soln in adjacent_solutions else 1
                    else:
                        diagonal_solutions[soln] = diagonal_solutions[soln] + 1 if soln in diagonal_solutions else 1

            # Smooth pixel if its only equivalent neighbour is a diagonal (or no equivalent neighbours)
            if smoothed_solutions[j, i] not in adjacent_solutions\
                    and (diagonal_solutions.get(smoothed_solutions[j, i], 0) < 2):
                # It becomes the same solution as its most prevalent neighbour
                current_max = 0
                current_soln = smoothed_solutions[j, i]
                for neighbour in set(adjacent_solutions.keys()).union(set(diagonal_solutions.keys())):
                    count = adjacent_solutions.get(neighbour, 0) + diagonal_solutions.get(neighbour, 0)
                    if count > current_max:
                        current_max = count
                        current_soln = neighbour
                smoothed_solutions[j, i] = current_soln
                smoothed_count += 1
    return smoothed_solutions


def blend_grid_wrapper(pixel_grid: npt.NDArray, blending_arrays: List[npt.NDArray], decay_fac_idx: int) -> npt.NDArray:
    if PRECOMPILED:
        return imaging_interface.blend_grid(pixel_grid, blending_arrays, decay_fac_idx)
    return blend_grid(pixel_grid, blending_arrays, decay_fac_idx)


blend_grid_spec = (
    float64[:, :, :],
    types.List(float64[:, :]),
    int32
)
@cc.export('blend_grid', blend_grid_spec)
def blend_grid(pixel_grid: npt.NDArray, blending_arrays: List[npt.NDArray], decay_fac_idx: int) -> npt.NDArray:
    """Blend a grid of pixels (happens after colouration). Blends more strongly where the iterations count is high,
    as this is strongly associated with more noise."""
    # TODO: a brightness gradient within large basins would look nice i.e. each colour region gets darker towards the middle (high distance from any other solution)
    blended_grid = np.zeros(pixel_grid.shape)
    next_arr_idx = 0
    for j in range(blended_grid.shape[0]):
        for i in range(blended_grid.shape[1]):
            blending_array = blending_arrays[next_arr_idx]
            j_coords = blending_array[:, 0].astype(np.int_)
            i_coords = blending_array[:, 1].astype(np.int_)
            for point in range(blending_array.shape[0]):
                blended_grid[j, i, :] += pixel_grid[j_coords[point], i_coords[point], :] * blending_array[point, decay_fac_idx + 2]
            next_arr_idx += 1
    return blended_grid.astype(np.uint8)


def create_blending_arrays_wrapper(iterations: npt.NDArray) -> List[npt.NDArray]:
    if PRECOMPILED:
        return imaging_interface.create_blending_arrays(iterations)
    return create_blending_arrays(iterations)


create_blending_arrays_spec = (
    int32[:, :]
)
@cc.export('create_blending_arrays', create_blending_arrays_spec)
def create_blending_arrays(iterations: npt.NDArray) -> List[npt.NDArray]:
    cbrt_iterations = np.cbrt(iterations)
    blending_arrays = []
    # Need to iterate through the arrays the same way as they are created
    # TODO: any loss of performance by just making the arrays in the blend_grid loops?
    for j in range(iterations.shape[0]):
        for i in range(iterations.shape[1]):
            blending_arrays.append(create_blending_array(j, i, iterations[j, i], cbrt_iterations[j, i]))
    return blending_arrays


def create_blending_array_wrapper(y_pixels: int, x_pixels: int, j: int, i: int, iterations: float, cbrt_iterations: float) -> npt.NDArray:
    if PRECOMPILED:
        return imaging_interface.create_blending_array(y_pixels, x_pixels, j, i, iterations, cbrt_iterations)
    return create_blending_array(y_pixels, x_pixels, j, i, iterations, cbrt_iterations)


create_blending_array_spec = (
    int32,
    int32,
    int32,
    int32,
    float64,
    float64
)
@cc.export('create_blending_array', create_blending_array_spec)
def create_blending_array(y_pixels: int, x_pixels: int, j: int, i: int, iterations: float, cbrt_iterations: float) -> npt.NDArray:
    half_width = int(cbrt_iterations) if iterations > 8 else 0
    j_range = range(max(0, j - half_width), min(y_pixels, j + half_width + 1))
    i_range = range(max(0, i - half_width), min(x_pixels, i + half_width + 1))
    n_rows = len(j_range) * len(i_range)
    weights = np.zeros((n_rows, 3), dtype=np.float_)
    row = 0
    for j_neighbour in j_range:
        for i_neighbour in i_range:
            dist = np.abs(j_neighbour-j) + np.abs(i_neighbour-i)
            # Each row is a neighbour to be blended (including the central point itself)
            # Rows consist of j_coords, i_coords, weights (one per decay factor)
            weights[row, 0:2] = [j_neighbour, i_neighbour]
            weights[row, 2] = np.exp(-cfg.BLENDING_DECAY_FACTOR*dist/(half_width+1))
            row += 1
    # Normalize the weights
    weights[:, 2:] = weights[:, 2:] / weights[:, 2:].sum(axis=0)
    return weights


# TODO: one script that recreates all of the precompiled modules
compiled_module_file_exists = any([x for x in cfg.BUILD_DIR.glob(f'{MODULE_NAME}*') if x.is_file()])
if not compiled_module_file_exists:
    cc.compile()
    src_dir = Path(os.path.realpath(__file__)).parent
    compiled_module_file = [x for x in src_dir.glob(f'{MODULE_NAME}*') if x.is_file()][0]
    utils.mkdir_if_nonexistent(cfg.BUILD_DIR)
    file_dest = cfg.BUILD_DIR / compiled_module_file.name
    file_dest.unlink(missing_ok=True)
    compiled_module_file.rename(file_dest)