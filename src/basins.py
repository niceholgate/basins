import config as cfg, calc, imaging, utils

import sys
import numpy as np
import sympy as sp
import numba as nb


@utils.timed
def produce_image_timed(unique_solns, x_coords, y_coords, f_lambda, j_lambda, delta, i):
    solutions, iterations = calc.solve_grid(unique_solns, x_coords, y_coords, f_lambda, j_lambda, delta)
    imaging.save_still(solutions, iterations, unique_solns,
                       smoothing=False, blending=False, colour_set=cfg.COLOUR_SET, frame=i)

# TODO:
#  1. add a CLI
#  2. add logging
#  3. Consolidate input validations in one place
#  4. Animations can pan/zoom the grid

# Sympy computes the partial derivatives of each equation with respect to x and y to obtain Jacobian matrix,
# then "lambdifies" them into Python functions with position args x, y, d.
j_sym = [[sp.diff(exp, sym) for sym in cfg.symbols[:2]] for exp in cfg.f_sym]
f_lambda = nb.njit(sp.lambdify(cfg.symbols, cfg.f_sym, 'numpy'), target_backend=cfg.NUMBA_TARGET)
j_lambda = nb.njit(sp.lambdify(cfg.symbols, j_sym, 'numpy'), target_backend=cfg.NUMBA_TARGET)

deltas = np.linspace(0, cfg.DELTA, cfg.FRAMES)
first_frame_unique_solns = calc.find_unique_solutions(f_lambda, j_lambda, deltas[0])
x_coords, y_coords = calc.get_image_pixel_coords(first_frame_unique_solns)

if cfg.ANIMATE:
    # Terminate with error if system of equations does not include a d term
    if all([cfg.symbols[2] not in exp.free_symbols for exp in cfg.f_sym]):
        print('For animations, must include at least one "d" term (delta to perturb the equation solutions)')
        sys.exit(0)
    # Assume that if the same number of solutions is found each time, the sorted solutions will
    # correspond to each other in sequence between different deltas
    unique_solns_per_delta = [first_frame_unique_solns]
    expected_number_of_solns = len(unique_solns_per_delta[0])
    for delta in deltas[1:]:
        this_delta_unique_solns = calc.find_unique_solutions(f_lambda, j_lambda, delta)
        if len(this_delta_unique_solns) > expected_number_of_solns:
            print(f'Terminating because number of solutions increased from {expected_number_of_solns}'
                  f' to {len(this_delta_unique_solns)} for delta={delta}')
            sys.exit(0)
        unique_solns_per_delta.append(this_delta_unique_solns)
    total_duration = 0.0
    for i, delta in enumerate(deltas):
        print(f'Now solving the grid for frame {i + 1} of {len(deltas)} (delta={delta})...')
        total_duration += produce_image_timed(unique_solns_per_delta[i], x_coords, y_coords, f_lambda, j_lambda, delta, i)
        utils.print_time_remaining_estimate(i, len(deltas), total_duration)
    imaging.stills_to_video()
else:
    produce_image_timed(first_frame_unique_solns, x_coords, y_coords, f_lambda, j_lambda, 0, 0)

should_exit = 'no'
while should_exit != 'exit':
    should_exit = input('Input "exit" to exit...')
