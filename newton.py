import config as cfg

import numba as nb
import numpy as np
import numpy.typing as npt
from typing import Tuple, Callable

nb.config.DISABLE_JIT = cfg.DISABLE_JIT


@nb.njit(target_backend=cfg.NUMBA_TARGET)
def solve(f_lam: Callable, j_lam: Callable, starting_guess: npt.NDArray, d: float) -> Tuple[npt.NDArray, npt.NDArray]:
    """Perform Newton's method until either the solution converges or the maximum iterations are exceeded."""
    current_guess = starting_guess
    delta_norm_hist = []
    delta_norm = np.float_(1000.0)
    n_iters = 0
    while delta_norm > cfg.EPSILON and n_iters < cfg.MAX_ITERS:
        f_num = f_lam(current_guess[0], current_guess[1], d)
        j_num = j_lam(current_guess[0], current_guess[1], d)
        delta = np.linalg.solve(np.array(j_num), -np.array(f_num)).flatten()
        current_guess = current_guess + delta
        delta_norm = np.linalg.norm(delta)
        delta_norm_hist.append(delta_norm)
        n_iters += 1

    return current_guess, np.array(delta_norm_hist)
