import config

import numba as nb
import numpy as np
from typing import Tuple, List

nb.config.DISABLE_JIT = config.DISABLE_JIT


@nb.njit(target_backend=config.NUMBA_TARGET)
def solve(f_lam, j_lam, starting_guess) -> Tuple[Tuple[float, float], List[float]]:
    current_guess = starting_guess
    delta_norm_hist = []
    delta_norm = 1000
    n_iters = 0
    while delta_norm > config.EPSILON and n_iters < config.MAX_ITERS:
        current_guess, delta_norm = iteration(f_lam, j_lam, current_guess)
        delta_norm_hist.append(delta_norm)
        n_iters += 1

    return current_guess, delta_norm_hist


@nb.njit(target_backend=config.NUMBA_TARGET)
def iteration(f_lam, j_lam, guess) -> Tuple[Tuple[float, float], float]:
    f_num = f_lam(guess[0], guess[1])
    j_num = j_lam(guess[0], guess[1])
    delta = np.linalg.solve(np.array(j_num), -np.array(f_num)).flatten()
    new_guess = (guess[0] + delta[0], guess[1] + delta[1])
    delta_norm = np.linalg.norm(delta)
    return new_guess, delta_norm