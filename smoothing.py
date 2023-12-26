import config

import numba as nb
nb.config.DISABLE_JIT = config.DISABLE_JIT


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