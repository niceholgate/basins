import config

import numpy as np
import numba as nb
nb.config.DISABLE_JIT = config.DISABLE_JIT


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
def create_blending_arrays(iterations, decay_facs):
    cbrt_iterations = np.cbrt(iterations)
    blending_weight_arrays = []
    for j in range(iterations.shape[0]):
        for i in range(iterations.shape[1]):
            blending_weight_arrays.append(create_blending_array(j, i, iterations, cbrt_iterations[j, i], decay_facs))
    return blending_weight_arrays


@nb.njit(target_backend=config.NUMBA_TARGET)
def create_blending_array(j, i, iterations, cbrt_iterations, decay_facs):
    half_width = int(cbrt_iterations) if iterations[j, i] > 8 else 0
    j_range = range(max(0, j - half_width), min(iterations.shape[0], j + half_width+1))
    i_range = range(max(0, i - half_width), min(iterations.shape[1], i + half_width + 1))
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
