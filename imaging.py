import config as cfg

import ffmpeg
import os
import matplotlib
import numpy as np
import numpy.typing as npt
import numba as nb
from PIL import Image
from pathlib import Path
from typing import List, Union

nb.config.DISABLE_JIT = cfg.DISABLE_JIT


def save_still(solutions: npt.NDArray, iterations: npt.NDArray, unique_solns: npt.NDArray, images_dir: Path,
               smoothing: bool = True, blending: bool = True, colour_set: Union[int, List[str]] = 0, frame: int = 0):
    """Generate an image with the specified colour scheme, or a range of colour schemes if none specified."""
    if not images_dir.exists():
        images_dir.mkdir(parents=True)

    pixel_grid = np.zeros((cfg.Y_PIXELS, cfg.X_PIXELS, 3), dtype=np.uint8)
    smoothed_solutions = _smooth_grid(solutions) if smoothing else solutions
    blending_arrays = _create_blending_arrays(iterations) if blending else []

    frame_formatted = cfg.FRAME_COUNT_PADDING.format(frame)

    if not (isinstance(colour_set, list) and len(colour_set) == unique_solns.shape[0]):
        idx = colour_set if isinstance(colour_set, int) else 0
        colour_set = []
        while len(colour_set) < unique_solns.shape[0]:
            colour_set.append(cfg.DEFAULT_COLOURS[idx % len(cfg.DEFAULT_COLOURS)])
            idx += 1
    rgb_colours = [matplotlib.colors.to_rgb(colour) for colour in colour_set]

    for j in range(cfg.Y_PIXELS):
        for i in range(cfg.X_PIXELS):
            if iterations[j, i] < cfg.BLACKOUT_ITERS:
                pixel_grid[j, i, :] = [int(x*255) for x in rgb_colours[smoothed_solutions[j, i]-1]]
    blended_pixel_grid = _blend_grid(pixel_grid, blending_arrays, 0) if blending else pixel_grid
    Image.fromarray(blended_pixel_grid, 'RGB').save(images_dir / f'frame-{frame_formatted}.png')


def stills_to_video(images_dir: Path, fps: int):
    """Use ffmpeg (via ffmpeg-python package) to assemble the image frames into a video."""
    images = [img for img in os.listdir(images_dir) if img.endswith(".png")]
    image_name_root = images[0].split('-')[0]
    ffmpeg.input(images_dir/f'{image_name_root}-{cfg.FRAME_COUNT_PADDING.strip("{").strip("}").replace(":","%")}.png',
                 pattern_type='sequence', framerate=fps)\
        .output(str(images_dir/f'video{fps}.mp4'))\
        .global_args('-loglevel', 'error')\
        .run()


@nb.njit(target_backend=cfg.NUMBA_TARGET)
def _smooth_grid(solutions: npt.NDArray) -> npt.NDArray:
    """Smooth a grid of solutions (happens before colouration).
    Just a crude algo to slightly reduce noise in very unstable areas."""
    smoothed_count = 0
    smoothed_solutions = solutions.copy()
    for j in range(1, smoothed_solutions.shape[0]-1):
        for i in range(1, smoothed_solutions.shape[1]-1):
            adjacent_solutions = {}
            diagonal_solutions = {}
            for delta_j in [-1, 0, 1]:
                for delta_i in [-1, 0, 1]:
                    soln = smoothed_solutions[j + delta_j, i + delta_i]
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


@nb.njit(target_backend=cfg.NUMBA_TARGET)
def _blend_grid(pixel_grid: npt.NDArray, blending_arrays: List[npt.NDArray], decay_fac_idx: int):
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


@nb.njit(target_backend=cfg.NUMBA_TARGET)
def _create_blending_arrays(iterations: npt.NDArray) -> List[npt.NDArray]:
    cbrt_iterations = np.cbrt(iterations)
    blending_arrays = []
    # Need to iterate through the arrays the same way as they are created
    # TODO: any loss of performance by just making the arrays in the blend_grid loops?
    for j in range(iterations.shape[0]):
        for i in range(iterations.shape[1]):
            blending_arrays.append(_create_blending_array(j, i, iterations[j, i], cbrt_iterations[j, i]))
    return blending_arrays


@nb.njit(target_backend=cfg.NUMBA_TARGET)
def _create_blending_array(j: int, i: int, iterations: float, cbrt_iterations: float) -> npt.NDArray:
    half_width = int(cbrt_iterations) if iterations > 8 else 0
    j_range = range(max(0, j - half_width), min(cfg.Y_PIXELS, j + half_width + 1))
    i_range = range(max(0, i - half_width), min(cfg.X_PIXELS, i + half_width + 1))
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
