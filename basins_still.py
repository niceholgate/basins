import config, calc
import imaging

import sympy as sp
import numba as nb
from pathlib import Path
from datetime import datetime

x, y, d = sp.symbols('x y d')
f_sym = [y**2+x**2-9, y-2*sp.cos(2*x)*sp.exp(-0.01*(x+y)**2)+2]
j_sym = [[sp.diff(exp, sym) for sym in [x, y]] for exp in f_sym]
f_lambda = nb.njit(sp.lambdify((x, y, d), f_sym, 'numpy'), target_backend=config.NUMBA_TARGET)
j_lambda = nb.njit(sp.lambdify((x, y, d), j_sym, 'numpy'), target_backend=config.NUMBA_TARGET)


unique_solns = calc.find_all_unique_solutions(f_lambda, j_lambda)
x_coords, y_coords = calc.get_image_pixel_coords(unique_solns)
solutions, iterations = calc.solve_grid(tuple(unique_solns), x_coords, y_coords, f_lambda, j_lambda)

images_dir = Path().cwd() / f'images/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
imaging.save_stills(solutions, iterations, images_dir, colour_offset=1)








