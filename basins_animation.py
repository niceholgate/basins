import config, calc
import imaging

import numpy as np
import sympy as sp
import numba as nb
import sys
from pathlib import Path
from datetime import datetime

x, y, d = sp.symbols('x y d')
f_sym = [y**2+x**2-9-d, y-2*sp.cos(2*x)*sp.exp(-0.01*(x+y)**2)+2]
j_sym = [[sp.diff(exp, sym) for sym in [x, y]] for exp in f_sym]
f_lambda = nb.njit(sp.lambdify((x, y, d), f_sym, 'numpy'), target_backend=config.NUMBA_TARGET)
j_lambda = nb.njit(sp.lambdify((x, y, d), j_sym, 'numpy'), target_backend=config.NUMBA_TARGET)

deltas = np.linspace(0, 1, 101)

# Assume that if the same number of solutions is found each time, the sorted solutions will correspond to each other in sequence
# between different deltas
unique_solns_per_delta = []
expected_number_of_solns = len(calc.find_all_unique_solutions(f_lambda, j_lambda, deltas[0]))
for delta in deltas:
    this_delta_unique_solns = calc.find_all_unique_solutions(f_lambda, j_lambda, delta)
    if len(this_delta_unique_solns) > expected_number_of_solns:
        print(f'Terminating because number of solutions increased from {expected_number_of_solns} to {len(this_delta_unique_solns)} for delta={delta}')
        sys.exit(0)
    unique_solns_per_delta.append(this_delta_unique_solns)

x_coords, y_coords = calc.get_image_pixel_coords(unique_solns_per_delta[0])

images_dir = Path().cwd() / f'images/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
for i, delta in enumerate(deltas):
    print(f'Now solving the grid for frame {i+1} of {len(deltas)} (delta={delta})...')
    solutions, iterations = calc.solve_grid(tuple(unique_solns_per_delta[i]), x_coords, y_coords, f_lambda, j_lambda, delta)
    imaging.save_stills(solutions, iterations, images_dir, colour_offset=2, frame=i)

