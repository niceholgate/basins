import src.config as cfg
from src.quad_tree import QuadTree

import numpy as np
import numpy.typing as npt
import numba as nb
from numba.experimental import jitclass
from numba import int32, float64, types
from typing import Tuple, Callable, List, Optional

nb.config.DISABLE_JIT = not cfg.ENABLE_JIT


@nb.njit(target_backend=cfg.NUMBA_TARGET)
def newton_solve(f_lam: Callable, j_lam: Callable, starting_guess: npt.NDArray, d: float) -> Tuple[npt.NDArray, int]:
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


@nb.njit(target_backend=cfg.NUMBA_TARGET)
def points_approx_equal(p1: npt.NDArray, p2: npt.NDArray) -> bool:
    return bool(np.linalg.norm(p1 - p2) < 2 * cfg.EPSILON)


solution_context_spec = [
    ('f_lambda', types.List(float64)(float64, float64, float64).as_type()),
    ('j_lambda', types.List(types.List(float64))(float64, float64, float64).as_type()),
    ('x_coords', float64[:]),
    ('y_coords', float64[:]),
    ('solutions_grid', int32[:, :]),
    ('iterations_grid', int32[:, :]),
    ('delta', float64),
    ('unique_solutions', float64[:, :])
]
@jitclass(solution_context_spec)
class Solver(object):
    def __init__(self, f_lambda, j_lambda, x_pixels, y_pixels, delta):
        self.f_lambda = f_lambda
        self.j_lambda = j_lambda
        self.delta = delta
        self.unique_solutions = self._find_unique_solutions()
        self.x_coords, self.y_coords = self._get_image_pixel_coords(y_pixels, x_pixels)
        self.solutions_grid = -np.ones((y_pixels, x_pixels), dtype=np.int_)
        self.iterations_grid = np.zeros((y_pixels, x_pixels), dtype=np.int_)

    def solve_grid(self) -> None:
        """Find which unique solution is reached for each pixel, and how many Newton's method iterations it took."""
        for j in range(self.y_coords.shape[0]):
            for i in range(self.x_coords.shape[0]):
                self._set_pixel_values_if_unset(j, i)

    def solve_grid_quadtrees(self) -> None:
        """Find which unique solution is reached for each pixel, and how many Newton's method iterations it took."""

        # Create the top QuadTree that encompasses the entire grid
        top_qt = QuadTree(0, self.x_coords.shape[0]-1, 0, self.y_coords.shape[0]-1, None)
        qt: Optional[QuadTree] = None

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
                n_interior_pixels = max((qt.x1 - qt.x0 - 1), 0) *\
                                    max((qt.y1 - qt.y0 - 1), 0)
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
                    self.solutions_grid[qt.y0:qt.y1 + 1, qt.x0:qt.x1 + 1] = last_boundary_soln
                    iters = self.iterations_grid[qt.y0:qt.y1 + 1, qt.x0:qt.x1 + 1]
                    known_iters = iters[iters != 0]
                    self.iterations_grid[qt.y0:qt.y1 + 1, qt.x0:qt.x1 + 1] = int(known_iters.mean())
                    qt.terminal = True

            # Set the next QuadTree on which to perform calculations.
            qt = qt.get_next_node_dfs()

            # Once the DFS ends, then we must have finished the whole grid.
            if qt is None:
                break

    def _find_unique_solutions(self) -> Optional[npt.NDArray]:
        """Do a randomised search to find unique solutions, stopping early if new unique solutions stop being found."""
        unique_solns: List[npt.NDArray] = []
        x_randoms = cfg.SEARCH_X_LIMS[0] + (cfg.SEARCH_X_LIMS[1] - cfg.SEARCH_X_LIMS[0]) * np.random.random(
            cfg.MAX_SEARCH_POINTS)
        y_randoms = cfg.SEARCH_Y_LIMS[0] + (cfg.SEARCH_Y_LIMS[1] - cfg.SEARCH_Y_LIMS[0]) * np.random.random(
            cfg.MAX_SEARCH_POINTS)
        # randoms = np.array([x_randoms, y_randoms], dtype=np.float_)
        point_count = 0
        converged_search_points_since_last_new_soln = 0
        for idx in range(x_randoms.shape[0]):
            point_count += 1
            soln, iters = newton_solve(self.f_lambda, self.j_lambda, np.array([x_randoms[idx], y_randoms[idx]], dtype=np.float64), self.delta)
            if iters < cfg.MAX_ITERS:
                if not unique_solns:
                    unique_solns.append(soln)

                any_equal = False
                for existing_soln in unique_solns:
                    if points_approx_equal(existing_soln, soln):
                        any_equal = True
                        break
                if any_equal:
                    converged_search_points_since_last_new_soln += 1
                else:
                    converged_search_points_since_last_new_soln = 0
                    unique_solns.append(soln)

                if converged_search_points_since_last_new_soln >= cfg.MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION:
                    print(f'End search with {len(unique_solns)} unique solutions after reaching the limit of '
                          f'{cfg.MAX_CONVERGED_SEARCH_POINTS_SINCE_LAST_NEW_SOLUTION} consecutive converged search points'
                          f' since the last new unique solution.')
                    break

        # Temporarily convert the solutions to tuples to sort them (ensures the random search returns the same result each
        # time for a given system of equations) then put them into one 2D array
        unique_solns_arr = np.array(sorted([(s[0], s[1]) for s in unique_solns]), np.float_)

        if len(unique_solns) < 2:
            print('Found fewer than 2 unique solutions, cannot generate an image')
            return None

        return unique_solns_arr

    def _get_image_pixel_coords(self, y_pixels: int, x_pixels: int) -> Tuple[npt.NDArray, npt.NDArray]:
        """Get the coordinates of all the pixels in the image, ensuring that all the known solutions to the system of
        equations fall safely within the bounds of the image."""
        # Collapse to a grid of final image's aspect ratio and with some borders around the unique solutions
        x_min, x_mean, x_max = self.unique_solutions[:, 0].min(), self.unique_solutions[:, 0].mean(), self.unique_solutions[:, 0].max()
        y_min, y_mean, y_max = self.unique_solutions[:, 1].min(), self.unique_solutions[:, 1].mean(), self.unique_solutions[:, 1].max()
        # Need to handle cases where solutions are collinear in x or y directions
        if y_min == y_max:
            x_range = (x_max - x_min) * 4
            y_range = x_range * y_pixels / x_pixels
        elif x_min == x_max:
            y_range = (y_max - y_min) * 4
            x_range = y_range * x_pixels / y_pixels
        else:
            x_range = (x_max - x_min) * 4
            y_range = (y_max - y_min) * 4

        return np.linspace(x_mean - x_range / 2, x_mean + x_range / 2, x_pixels), \
               np.linspace(y_mean - y_range / 2, y_mean + y_range / 2, y_pixels)

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
                    print(f'WARNING: Image will ignore a novel solution found on the grid: {solution_pixel}')
            else:
                self.solutions_grid[j, i] = 0
                print(f'WARNING: Maximum iterations were exceeded for a pixel')


