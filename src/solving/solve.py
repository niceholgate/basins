import src.config as cfg
from .quad_tree import QuadTree

import numpy as np
import numpy.typing as npt
import numba as nb
from numba.experimental import jitclass
from numba import int32, float64, types
from typing import Tuple, Callable, Optional

nb.config.DISABLE_JIT = not cfg.ENABLE_JIT


@nb.njit(target_backend=cfg.NUMBA_TARGET)
def points_approx_equal(p1: npt.NDArray, p2: npt.NDArray, epsilon: float = cfg.EPSILON) -> bool:
    return bool(np.linalg.norm(p1 - p2) < 2 * epsilon)


@nb.njit(target_backend=cfg.NUMBA_TARGET)
def newton_solve(f_lam: Callable, j_lam: Callable, starting_guess: npt.NDArray, d: float) -> Tuple[
    npt.NDArray, int]:
    """Perform Newton's method until either the solution converges or the maximum iterations are exceeded."""
    current_guess = starting_guess
    # delta_norm_hist = []
    delta_norm = np.float_(1000.0)
    n_iters = 0
    while delta_norm > cfg.EPSILON and n_iters < cfg.MAX_ITERS:
        f_num = np.array(f_lam(current_guess[0], current_guess[1], d), dtype=np.float_)
        j_num = np.array(j_lam(current_guess[0], current_guess[1], d), dtype=np.float_)
        delta = np.linalg.solve(j_num, -f_num).flatten()
        current_guess = current_guess + delta
        delta_norm = np.linalg.norm(delta)
        # delta_norm_hist.append(delta_norm)
        n_iters += 1

    return current_guess, n_iters


create_solver_spec = (
    types.List(float64)(float64, float64, float64).as_type(),
    types.List(types.List(float64))(float64, float64, float64).as_type(),
    float64[:],
    float64[:],
    float64,
    float64[:, :]
)
@nb.njit(target_backend=cfg.NUMBA_TARGET)
def create_solver(f_lambda, j_lambda, x_coords, y_coords, delta, unique_solutions):
    return Solver(f_lambda, j_lambda, x_coords, y_coords, delta, unique_solutions)


solution_context_spec = (
    ('f_lambda', types.List(float64)(float64, float64, float64).as_type()),
    ('j_lambda', types.List(types.List(float64))(float64, float64, float64).as_type()),
    ('x_coords', float64[:]),
    ('y_coords', float64[:]),
    ('solutions_grid', int32[:, :]),
    ('iterations_grid', int32[:, :]),
    ('delta', float64),
    ('unique_solutions', float64[:, :])
)
@jitclass(solution_context_spec)
class Solver(object):
    def __init__(self, f_lambda, j_lambda, x_coords, y_coords, delta, unique_solutions):
        self.f_lambda = f_lambda
        self.j_lambda = j_lambda
        self.delta = delta
        self.unique_solutions = unique_solutions
        self.x_coords, self.y_coords = x_coords, y_coords
        self.solutions_grid = -np.ones((len(y_coords), len(x_coords)), dtype=np.int_)
        self.iterations_grid = np.zeros((len(y_coords), len(x_coords)), dtype=np.int_)

    def solve_grid(self) -> None:
        """Find which unique solution is reached for each pixel, and how many Newton's method iterations it took."""
        for j in nb.prange(self.y_coords.shape[0]):
            for i in nb.prange(self.x_coords.shape[0]):
                self._set_pixel_values_if_unset(j, i)

    def solve_grid_quadtrees(self) -> None:
        """Find which unique solution is reached for each pixel, and how many Newton's method iterations it took."""

        # Create the top QuadTree that encompasses the entire grid
        top_qt = QuadTree((0, self.x_coords.shape[0]-1), (0, self.y_coords.shape[0]-1), None)
        qt: Optional[QuadTree] = None
        qts = []
        for child in top_qt.get_children():
            qts.append(child)

        # Depth-First Search through nested QuadTrees until every pixel has been filled in on the grids
        while True:
            if qt is None:
                qt = top_qt

            # Iterate around the boundary of the QuadTree calculating the solution, and note if they are all identical
            last_boundary_soln = -93736
            unique_boundary_soln = True
            for i, j in qt.boundary_coordinates_generator():
                self._set_pixel_values_if_unset(j, i)
                if unique_boundary_soln:
                    if last_boundary_soln != -93736:
                        unique_boundary_soln = self.solutions_grid[j, i] == last_boundary_soln
                    last_boundary_soln = self.solutions_grid[j, i]

            # If they are all identical around the boundary, check some random interior points to infer if
            # the entire region encompassed by the QuadTree is uniform...
            if unique_boundary_soln:
                unique_interior_soln = True
                n_interior_pixels = max((qt.x_lims[1] - qt.x_lims[0] - 1), 0) *\
                                    max((qt.y_lims[1] - qt.y_lims[0] - 1), 0)
                pixels_checked = 0
                #TODO: if there are fewer than X pixels, explicitly check all of the interior pixels instead of random ones
                while pixels_checked < 20 and pixels_checked < n_interior_pixels:
                    rand_i = qt.random_interior_x()
                    rand_j = qt.random_interior_y()
                    self._set_pixel_values_if_unset(rand_j, rand_i)
                    unique_interior_soln = self.solutions_grid[rand_j, rand_i] == last_boundary_soln
                    if not unique_interior_soln:
                        break
                    pixels_checked += 1

                # ... and if so, then fill the whole QuadTree area with that solution, and mark it as terminal
                # (this will cause the QuadTree to have no children, so its next DFS node is its parent).
                if unique_interior_soln:
                    self.solutions_grid[qt.y_lims[0]:qt.y_lims[1] + 1, qt.x_lims[0]:qt.x_lims[1] + 1] = last_boundary_soln
                    iters = self.iterations_grid[qt.y_lims[0]:qt.y_lims[1] + 1, qt.x_lims[0]:qt.x_lims[1] + 1]
                    # TODO: is this slow?
                    # known_iters = iters[iters != 0]
                    # self.iterations_grid[qt.y0:qt.y1 + 1, qt.x0:qt.x1 + 1] = int(known_iters.mean())
                    sum_iters = 0
                    count_iters = 0
                    for _, value in np.ndenumerate(iters):
                        if value > 0:
                            count_iters += 1
                            sum_iters += value
                    mean_iters = int(sum_iters/count_iters)
                    self.iterations_grid[qt.y_lims[0]:qt.y_lims[1] + 1, qt.x_lims[0]:qt.x_lims[1] + 1] = mean_iters
                    qt.terminal = True

            # Once the DFS ends, then we must have finished the whole grid.
            if len(qts) == 0:
                break

            # Set the next QuadTree on which to perform calculations.
            qt = qts.pop()

            # Add its children to the stack
            for child in qt.get_children():
                qts.append(child)

    def _get_index_of_matching_unique_soln(self, soln: npt.NDArray) -> int:
        match = -1
        for unique_soln_idx in range(self.unique_solutions.shape[0]):
            if points_approx_equal(self.unique_solutions[unique_soln_idx, :], soln):
                match = unique_soln_idx + 1
                break
        return match

    def _set_pixel_values_if_unset(self, j: int, i: int):
        # Set the values for this pixel if it hasn't been attempted yet (-1)
        if self.solutions_grid[j, i] == -1:
            solution_pixel, iterations_pixel = newton_solve(self.f_lambda, self.j_lambda, np.array([self.x_coords[i], self.y_coords[j]]), self.delta)
            self.iterations_grid[j, i] = iterations_pixel
            if iterations_pixel < cfg.MAX_ITERS:
                match = self._get_index_of_matching_unique_soln(solution_pixel)
                if match != -1:
                    self.solutions_grid[j, i] = match
                else:
                    self.solutions_grid[j, i] = 0
                    # print(f'WARNING: Image will ignore a novel solution found on the grid: {solution_pixel}')
            else:
                self.solutions_grid[j, i] = 0
                # print(f'WARNING: Maximum iterations were exceeded for a pixel')
