import config, calc, imaging

import numpy as np
import sympy as sp
import numba as nb
import sys
from pathlib import Path
from datetime import datetime

x, y, d = sp.symbols('x y d')
f_sym = [y**2+2*x*y+2*x**2-4-0.5*sp.sin(15*(x+d)), y-10*x**2+3+d]
j_sym = [[sp.diff(exp, sym) for sym in [x, y]] for exp in f_sym]
f_lambda = nb.njit(sp.lambdify(sp.symbols('x y d'), f_sym, 'numpy'), target_backend=config.NUMBA_TARGET)
j_lambda = nb.njit(sp.lambdify((x, y, d), j_sym, 'numpy'), target_backend=config.NUMBA_TARGET)

deltas = np.linspace(0, 2, 30)

# Assume that if the same number of solutions is found each time, the sorted solutions will
# correspond to each other in sequence between different deltas
unique_solns_per_delta = []
expected_number_of_solns = len(calc.find_all_unique_solutions(f_lambda, j_lambda, deltas[0]))
for delta in deltas:
    this_delta_unique_solns = calc.find_all_unique_solutions(f_lambda, j_lambda, delta)
    if len(this_delta_unique_solns) > expected_number_of_solns:
        print(f'Terminating because number of solutions increased from {expected_number_of_solns}'
              f' to {len(this_delta_unique_solns)} for delta={delta}')
        sys.exit(0)
    unique_solns_per_delta.append(this_delta_unique_solns)

x_coords, y_coords = calc.get_image_pixel_coords(unique_solns_per_delta[0])

total_duration = 0.0
images_dir = Path().cwd() / f'images/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
for i, delta in enumerate(deltas):
    start = datetime.now()
    print(f'Now solving the grid for frame {i+1} of {len(deltas)} (delta={delta})...')
    solutions, iterations = calc.solve_grid(unique_solns_per_delta[i], x_coords, y_coords, f_lambda, j_lambda, delta)
    imaging.save_still(solutions, iterations, images_dir, smoothing=False, blending=False, colour_offset=2, frame=i)
    total_duration += (datetime.now()-start).total_seconds()
    if i != len(deltas)-1:
        mean_duration = '{:.2f}'.format(total_duration/(i+1))
        est_mins_remaining = '{:.2f}'.format(total_duration/(i+1)*(len(deltas)-i-1)/60)
        print(f'Mean frame generation time is {mean_duration} seconds;'
              f' estimate {est_mins_remaining} minutes remaining for video generation')

imaging.stills_to_video(images_dir, 30)
