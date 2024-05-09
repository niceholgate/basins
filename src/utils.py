import src.config as cfg

import numpy.typing as npt
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import numba as nb
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple


SYMBOLS = sp.symbols('x y d')


def get_lambdas(expressions: List[str]) -> Tuple[Callable, Callable]:
    """
    # Sympy computes the partial derivatives of each equation with respect to x and y to obtain Jacobian matrix,
    # then "lambdifies" them into Python functions with position args x, y, d.
    """
    f_sym = [sp.parsing.sympy_parser.parse_expr(ex) for ex in expressions]
    j_sym = [[sp.diff(exp, sym) for sym in SYMBOLS[:2]] for exp in f_sym]
    f_lambda = nb.njit(sp.lambdify(SYMBOLS, f_sym, 'numpy'), target_backend=cfg.NUMBA_TARGET)
    j_lambda = nb.njit(sp.lambdify(SYMBOLS, j_sym, 'numpy'), target_backend=cfg.NUMBA_TARGET)
    return f_lambda, j_lambda


def get_images_dir(start_time: str, uuid: str) -> Path:
    images_dir = Path().cwd() / f'images/{start_time}_{uuid}'
    if not images_dir.exists():
        images_dir.mkdir(parents=True)
    return images_dir


def timed(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        start = datetime.now()
        func(*args, **kwargs)
        return (datetime.now() - start).total_seconds()
    return wrapper


def print_time_remaining_estimate(i: int, N: int, duration_so_far: float) -> None:
    if i != N - 1:
        mean_duration = '{:.2f}'.format(duration_so_far / (i + 1))
        est_mins_remaining = '{:.2f}'.format(duration_so_far / (i + 1) * (N - i - 1) / 60)
        print(f'Mean frame generation time is {mean_duration} seconds;'
              f' estimate {est_mins_remaining} minutes remaining for video generation')


def plot_unique_solutions(unique_solns: npt.NDArray) -> None:
    plt.figure()
    plt.scatter(unique_solns[:, 0], unique_solns[:, 1])
    plt.show(block=True)
