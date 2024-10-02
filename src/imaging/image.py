import src.config as cfg
import src.utils as utils
from pydantic import ValidationError
from src.api.requests import StillRequest, AnimationRequest

import ffmpeg
import os
import io
import numpy as np
import numpy.typing as npt
import numba as nb
from typing import Dict, Optional
from pathlib import Path
import matplotlib
from PIL import Image
from typing import Union, List

import src.imaging.interface as imaging_interface

nb.config.DISABLE_JIT = not cfg.ENABLE_JIT


def save_still(images_dir: Path, solutions_grid: npt.NDArray, iterations_grid: npt.NDArray, unique_solutions: npt.NDArray, smoothing: bool = True, blending: bool = True, colour_set: Union[int, List[str]] = 0, frame: int = 0):
    """Generate an image with the specified colour scheme, or a range of colour schemes if none specified."""
    pixel_grid = np.zeros((solutions_grid.shape[0], solutions_grid.shape[1], 3), dtype=np.uint8)
    smoothed_solutions = imaging_interface.smooth_grid_wrapper(solutions_grid) if smoothing else solutions_grid
    blending_arrays = create_blending_arrays(iterations_grid) if blending else []

    if not (isinstance(colour_set, list) and len(colour_set) == unique_solutions.shape[0]):
        idx = colour_set if isinstance(colour_set, int) else 0
        colour_set = []
        while len(colour_set) < unique_solutions.shape[0]:
            colour_set.append(cfg.DEFAULT_COLOURS[idx % len(cfg.DEFAULT_COLOURS)])
            idx += 1
    rgb_colours = [matplotlib.colors.to_rgb(colour) for colour in colour_set]
    print(colour_set)
    print(unique_solutions)

    for j in range(solutions_grid.shape[0]):
        for i in range(solutions_grid.shape[1]):
            if iterations_grid[j, i] < cfg.BLACKOUT_ITERS:
                pixel_grid[j, i, :] = [int(x*255) for x in rgb_colours[smoothed_solutions[j, i]-1]]
    blended_pixel_grid = imaging_interface.blend_grid_wrapper(pixel_grid, blending_arrays, 0) if blending else pixel_grid
    np.savetxt(images_dir / utils.get_frame_filename(frame, 'txt'), blended_pixel_grid.reshape([blended_pixel_grid.shape[0],
               blended_pixel_grid.shape[1]*blended_pixel_grid.shape[2]]), fmt='%u')
    if cfg.SAVE_PNG_FRAMES:
        Image.fromarray(blended_pixel_grid, 'RGB').save(images_dir / utils.get_frame_filename(frame, 'png'))


def create_blending_arrays(iterations: npt.NDArray) -> List[npt.NDArray]:
    cbrt_iterations = np.cbrt(iterations)
    blending_arrays = []
    # Need to iterate through the arrays the same way as they are created
    for j in range(iterations.shape[0]):
        for i in range(iterations.shape[1]):
            blending_arrays.append(imaging_interface.create_blending_array_wrapper(j, i, iterations[j, i], cbrt_iterations[j, i]))
    return blending_arrays


def png_to_mp4(images_dir: Path, fps: int):
    """Use ffmpeg (via ffmpeg-python package) to assemble the image frames into a video."""
    images = sorted([img for img in os.listdir(images_dir) if img.endswith(".png")])
    image_name_root = images[0].split('-')[0]
    ffmpeg.input(images_dir/f'{image_name_root}-{cfg.FRAME_COUNT_PADDING.strip("{").strip("}").replace(":","%")}.png',
                 pattern_type='sequence', framerate=fps)\
        .output(str(images_dir/f'video{fps}.mp4'))\
        .global_args('-loglevel', 'error')\
        .run()


def rgb_to_mp4(images_dir: Path, fps: int):
    images = sorted([img for img in os.listdir(images_dir) if img.endswith(".txt")])
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

