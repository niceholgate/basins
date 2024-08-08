import src.config as cfg
import src.utils as utils
from pydantic import ValidationError
from src.solver import Solver
from src.request_types import StillRequest, AnimationRequest

import ffmpeg
import os
import io
import matplotlib
import numpy as np
import numpy.typing as npt
import numba as nb
from PIL import Image
from typing import List, Union, Dict, Optional
from pathlib import Path

nb.config.DISABLE_JIT = cfg.DISABLE_JIT


def save_still(images_dir: Path, solver: Solver, smoothing: bool = True, blending: bool = True, colour_set: Union[int, List[str]] = 0, frame: int = 0):
    """Generate an image with the specified colour scheme, or a range of colour schemes if none specified."""
    pixel_grid = np.zeros((solver.solutions_grid.shape[0], solver.solutions_grid.shape[1], 3), dtype=np.uint8)
    smoothed_solutions = _smooth_grid(solver.solutions_grid) if smoothing else solver.solutions_grid
    blending_arrays = _create_blending_arrays(solver.iterations_grid) if blending else []

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
    blended_pixel_grid = _blend_grid(pixel_grid, blending_arrays, 0) if blending else pixel_grid
    np.savetxt(images_dir / utils.get_frame_filename(frame, 'txt'), blended_pixel_grid.reshape([blended_pixel_grid.shape[0],
               blended_pixel_grid.shape[1]*blended_pixel_grid.shape[2]]), fmt='%u')
    if cfg.SAVE_PNG_FRAMES:
        Image.fromarray(blended_pixel_grid, 'RGB').save(images_dir / utils.get_frame_filename(frame, 'png'))


def png_to_mp4(images_dir: Path, fps: int):
    """Use ffmpeg (via ffmpeg-python package) to assemble the image frames into a video."""
    images = [img for img in os.listdir(images_dir) if img.endswith(".png")]
    image_name_root = images[0].split('-')[0]
    ffmpeg.input(images_dir/f'{image_name_root}-{cfg.FRAME_COUNT_PADDING.strip("{").strip("}").replace(":","%")}.png',
                 pattern_type='sequence', framerate=fps)\
        .output(str(images_dir/f'video{fps}.mp4'))\
        .global_args('-loglevel', 'error')\
        .run()


def rgb_to_mp4(images_dir: Path, fps: int):
    images = [img for img in os.listdir(images_dir) if img.endswith(".txt")]
    rgb_frame_data_path = images_dir / images[0]
    first_frame = load_rgb_file(rgb_frame_data_path, 3)
    height, width, channels = first_frame.shape

    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=fps)
        .output(str(images_dir/f'video-rgb{fps}.mp4'))
        .overwrite_output()
        .run_async(pipe_stdin=True, overwrite_output=True, pipe_stderr=True)
    )
    for i, image in enumerate(images):
        try:
            rgb_frame_data_path = images_dir / image
            frame = load_rgb_file(rgb_frame_data_path, 3) if i > 0 else first_frame
            process.stdin.write(
                frame.astype(np.uint8).tobytes()
            )
        except Exception as e: # should probably be an exception related to process.stdin.write, then after catch RGB load exception
            for line in io.TextIOWrapper(process.stderr, encoding="utf-8"): # I didn't know how to get the stderr from the process, but this worked for me
                print(line) # <-- print all the lines in the processes stderr after it has errored
            process.stdin.close()
            process.wait()
            return # cant run anymore so end the for loop and the function execution
    out, err = process.communicate()


def load_rgb_file(rgb_frame_data_path: Path, output_array_dim: int = 2) -> np.ndarray:
    array_2d = np.genfromtxt(rgb_frame_data_path, dtype=np.int_)
    if output_array_dim == 3:
        height, width_x_channels = array_2d.shape
        return array_2d.reshape([height, int(width_x_channels/3), 3])
    return array_2d


def load_run_data(uuid: str) -> Optional[Dict]:
    directory = utils.get_images_dir(uuid)
    logs_files = list(directory.glob(f'*.log'))
    if logs_files:
        file_path = logs_files[0]
        with open(file_path, 'r') as log:
            first_line = log.readline().strip('\n')
            json_str = first_line.split(' - ')[1]

            try:
                if file_path.name.startswith('animation'):
                    return AnimationRequest.model_validate_json(json_str).model_dump()
                else:
                    return StillRequest.model_validate_json(json_str).model_dump()
            except ValidationError as e:
                return {'message': 'error parsing!'}
    return None



@nb.njit(target_backend=cfg.NUMBA_TARGET)
def _smooth_grid(solutions: npt.NDArray) -> npt.NDArray:
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
def _create_blending_array(y_pixels: int, x_pixels: int, j: int, i: int, iterations: float, cbrt_iterations: float) -> npt.NDArray:
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
