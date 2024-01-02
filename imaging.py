import config

import matplotlib
import numpy as np
import numba as nb
from PIL import Image

nb.config.DISABLE_JIT = config.DISABLE_JIT


# Generate image an image with the specified colour scheme, or a range of colour schemes if none specified
def save_stills(solutions, iterations, images_dir, colour_offset=None, frame=0):
    if not images_dir.exists():
        images_dir.mkdir(parents=True)
    pixel_grid = np.zeros((config.Y_PIXELS, config.X_PIXELS, 3), dtype=np.uint8)
    blending_arrays = create_blending_arrays(iterations)
    smoothed_solutions = smooth_grid(solutions)
    rgb_colours = [matplotlib.colors.to_rgb(config.COLOURS[colour_idx]) for colour_idx in range(len(config.COLOURS))]
    colour_offsets = range(len(config.COLOURS)) if colour_offset is None else [colour_offset]
    for colour_offset in colour_offsets:
        for j in range(config.Y_PIXELS):
            for i in range(config.X_PIXELS):
                if iterations[j, i] < config.BLACKOUT_ITERS:
                    colour_idx = (smoothed_solutions[j, i] + colour_offset) % len(config.COLOURS)
                    pixel_grid[j, i, :] = [int(x*255) for x in rgb_colours[colour_idx]]
        print(f'Blending image {colour_offset + 1} of {len(config.COLOURS)}')
        blended_pixel_grid = blend_grid(pixel_grid, blending_arrays, 0)
        Image.fromarray(blended_pixel_grid, 'RGB').save(images_dir / f'{colour_offset}-{frame}.png')


# Smooth a grid of solutions (happens before colouration)
@nb.njit(target_backend=config.NUMBA_TARGET)
def smooth_grid(solutions):
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

    print(f'Smoothed {smoothed_count}/{solutions.shape[0]*solutions.shape[1]} pixels')
    return smoothed_solutions


# TODO: a brightness gradient within large basins would look nice. i.e. each colour region gets darker towards the middle (high distance from any other solution)
# Blend a grid of pixels (happens after colouration)
@nb.njit(target_backend=config.NUMBA_TARGET)
def blend_grid(pixel_grid, blending_arrays, decay_fac_idx):
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


@nb.njit(target_backend=config.NUMBA_TARGET)
def create_blending_arrays(iterations):
    cbrt_iterations = np.cbrt(iterations)
    blending_arrays = []
    # Need to iterate through the arrays the same way as they are created
    # TODO: any loss of performance by just making the arrays in the blend_grid loops?
    for j in range(iterations.shape[0]):
        for i in range(iterations.shape[1]):
            blending_arrays.append(create_blending_array(j, i, iterations[j, i], cbrt_iterations[j, i], np.array([config.BLENDING_DECAY_FACTOR])))
    return blending_arrays


@nb.njit(target_backend=config.NUMBA_TARGET)
def create_blending_array(j, i, iterations, cbrt_iterations, decay_facs):
    half_width = int(cbrt_iterations) if iterations > 8 else 0
    j_range = range(max(0, j - half_width), min(config.Y_PIXELS, j + half_width + 1))
    i_range = range(max(0, i - half_width), min(config.X_PIXELS, i + half_width + 1))
    n_rows = len(j_range) * len(i_range)
    weights = np.zeros((n_rows, 2 + len(decay_facs)), dtype=np.float_)
    row = 0
    for j_neighbour in j_range:
        for i_neighbour in i_range:
            dist = np.abs(j_neighbour-j) + np.abs(i_neighbour-i)
            # Each row is a neighbour to be blended (including the central point itself)
            # Rows consist of j_coords, i_coords, weights (one per decay factor)
            weights[row, 0:2] = [j_neighbour, i_neighbour]
            weights[row, 2:] = np.exp(-decay_facs*dist/(half_width+1))
            row += 1
    # Normalize the weights
    weights[:, 2:] = weights[:, 2:] / weights[:, 2:].sum(axis=0)
    return weights
